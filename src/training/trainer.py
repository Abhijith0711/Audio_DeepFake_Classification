import json
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import calculate_metrics
from src.evaluation.plotting import save_training_curves
from src.models.ft_rcl_detector import FTRCLDetector
from src.models.mobilenet_v3_audio import MobileNetV3SmallAudio
from src.robustness.conditions import all_training_conditions
from src.robustness.degrade import degrade
from src.training.losses.ft_rcl import ft_rcl_loss
from src.training.losses.star_edge import abr_loss, channel_aux_loss, stc_loss
from src.training.run_artifacts import (
    append_train_log,
    print_metrics_table,
    save_training_history,
)
from src.training.schedulers.robustness_curriculum import max_severity_for_epoch, severity_eta

logger = logging.getLogger(__name__)

FAMILY_TO_ID = {
    "resample": 0,
    "mp3_codec": 1,
    "additive_noise": 2,
    "telephony": 3,
}


class Trainer:
    def __init__(
        self,
        config,
        project_root,
        experiment_dir,
        figure_dir,
        frontend,
        device,
        mode="M0",
    ):
        self.config = config
        self.project_root = project_root
        self.experiment_dir = experiment_dir
        self.figure_dir = figure_dir
        self.frontend = frontend
        self.device = device
        self.mode = mode

        star = config.get("star_edge", {})
        self.use_star = bool(star.get("enabled", False))
        use_cad = bool(star.get("use_cad", False)) if self.use_star else False

        if mode == "M0":
            self.model = MobileNetV3SmallAudio(num_classes=2).to(device)
            self.subband = None
        else:
            self.model = FTRCLDetector(num_classes=2, use_cad=use_cad).to(device)
            self.subband = self.model.subband_pool

        self.channel_head = None
        if use_cad:
            self.channel_head = nn.Linear(8, len(FAMILY_TO_ID)).to(device)

        self.history = []
        self.best_score = -1e9
        self.best_epoch = 0

        train_cfg = config["training"]
        self.amp_enabled = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters())
            + (list(self.channel_head.parameters()) if self.channel_head else []),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        ft_cfg = config.get("ft_rcl", {})
        self.beta = ft_cfg.get("beta", 0.5)
        self.gamma_pc = ft_cfg.get("gamma_pc", 1.0)
        self.gamma_ftsc = ft_cfg.get("gamma_ftsc", 0.5)
        self.type_alpha = ft_cfg.get(
            "type_alpha",
            {"speech": 1.0, "singing": 1.0, "music": 0.3, "sound": 0.5},
        )
        self.lambda_delta = config.get("selection", {}).get("lambda_delta", 0.5)
        self.use_j_score = config.get("selection", {}).get("use_j_score", mode not in ("M0",))

        curriculum = ft_cfg.get("curriculum", {})
        self.epoch_milestones = curriculum.get("epoch_milestones", [1, 6, 13])
        self.severity_milestones = curriculum.get("severity_milestones", [1, 3, 4])

        self.star_cfg = star

    def autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        )

    def _forward_clean(self, waveforms):
        x = self.frontend(waveforms)

        if self.mode == "M0":
            with self.autocast():
                return self.model(x)

        if isinstance(self.model, FTRCLDetector):
            with self.autocast():
                logits, signature, u = self.model(
                    x,
                    waveforms=waveforms,
                    current_epoch=getattr(self, "current_epoch", 1),
                )
            return logits, signature, u

        raise RuntimeError("Unexpected model type")

    def train_one_epoch(self, loader, criterion, epoch):
        self.current_epoch = epoch
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        paired = self.mode != "M0"

        for batch in tqdm(loader, desc="Training", leave=False):
            if paired:
                clean_w, deg_w, labels, types, meta = batch
                labels = labels.to(self.device, non_blocking=True)
                clean_w = clean_w.to(self.device, non_blocking=True)
                deg_w = deg_w.to(self.device, non_blocking=True)

                max_sev = max_severity_for_epoch(epoch, self.epoch_milestones, self.severity_milestones)

                if isinstance(meta, list):
                    severities = [m["severity"] for m in meta]
                    eta = float(np.mean([severity_eta(s, max_sev) for s in severities]))
                    types_list = types
                else:
                    eta = severity_eta(meta.get("severity", 1), max_sev)
                    types_list = types

                self.optimizer.zero_grad(set_to_none=True)

                with self.autocast():
                    if self.mode == "M1":
                        x_deg = self.frontend(deg_w)
                        logits, _, _ = self.model(x_deg, waveforms=deg_w, current_epoch=epoch)
                        loss = criterion(logits, labels)
                    else:
                        x_clean = self.frontend(clean_w)
                        x_deg = self.frontend(deg_w)
                        logits_c, sig_c, u_c = self.model(
                            x_clean,
                            waveforms=clean_w,
                            current_epoch=epoch,
                        )
                        logits_d, sig_d, u_d = self.model(
                            x_deg,
                            waveforms=deg_w,
                            current_epoch=epoch,
                        )

                        loss, _ = ft_rcl_loss(
                            logits_c,
                            logits_d,
                            labels,
                            sig_c,
                            sig_d,
                            types_list,
                            eta if self.mode == "M3" else 1.0,
                            self.type_alpha,
                            beta=self.beta,
                            gamma_pc=self.gamma_pc if self.mode in ("M2", "M3") else 0.0,
                            gamma_ftsc=self.gamma_ftsc if self.mode == "M3" else 0.0,
                            criterion=criterion,
                        )

                        if self.use_star and self.mode == "M3":
                            star = self.star_cfg
                            if star.get("use_abr", True):
                                p_c = torch.softmax(logits_c, dim=1)[:, 1]
                                p_d = torch.softmax(logits_d, dim=1)[:, 1]
                                loss = loss + star.get("rho", 0.2) * abr_loss(p_c, p_d, labels)

                            if star.get("use_cad", False) and self.channel_head is not None:
                                family_ids = torch.tensor(
                                    [FAMILY_TO_ID.get(m["condition_family"], 0) for m in meta],
                                    device=self.device,
                                    dtype=torch.long,
                                )
                                loss = loss + channel_aux_loss(
                                    self.channel_head(u_d),
                                    family_ids,
                                    criterion,
                                )

                            if star.get("use_stc", False) and torch.rand(1).item() < 0.5:
                                p_c = torch.softmax(logits_c, dim=1)[:, 1]
                                p_d = torch.softmax(logits_d, dim=1)[:, 1]
                                loss = loss + star.get("delta", 0.3) * stc_loss([sig_c, sig_d], [p_c, p_d])

                        logits = logits_c

            else:
                waveforms, labels, _meta = batch
                labels = labels.to(self.device, non_blocking=True)
                waveforms = waveforms.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                with self.autocast():
                    logits = self._forward_clean(waveforms)
                    loss = criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            running_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)

        return running_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def evaluate_loader(self, loader, criterion=None, description="Evaluating"):
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        all_labels = []
        all_scores = []
        all_meta = []

        for batch in tqdm(loader, desc=description, leave=False):
            if len(batch) == 3:
                waveforms, labels, meta = batch
            else:
                waveforms, labels, meta = batch[0], batch[1], batch[2]

            labels = labels.to(self.device, non_blocking=True)
            waveforms = waveforms.to(self.device, non_blocking=True)
            x = self.frontend(waveforms)

            with self.autocast():
                if self.mode == "M0":
                    logits = self.model(x)
                else:
                    logits, _, _ = self.model(x)

                if criterion is not None:
                    loss = criterion(logits, labels)
                    total_loss += float(loss.item()) * x.size(0)

            scores = torch.softmax(logits.float(), dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())

            if isinstance(meta, dict):
                all_meta.append(meta)
            elif isinstance(meta, list):
                all_meta.extend(meta)
            else:
                for item in meta:
                    all_meta.append(item if isinstance(item, dict) else {"file_id": str(item)})

            total_items += x.size(0)

        metrics, _ = calculate_metrics(all_labels, all_scores)
        metrics["loss"] = total_loss / max(total_items, 1)
        return metrics, np.asarray(all_labels), np.asarray(all_scores), all_meta

    @torch.no_grad()
    def estimate_dev_delta_macro_f1(self, dev_loader, max_batches=5):
        if self.mode == "M0" or max_batches <= 0:
            return 0.0

        conditions = [c for c in all_training_conditions() if c.family != "identity"][:2]
        clean_scores = []
        clean_labels = []
        deg_scores = []
        deg_labels = []

        for batch_idx, batch in enumerate(dev_loader):
            if batch_idx >= max_batches:
                break

            waveforms, labels, meta = batch
            waveforms = waveforms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            x = self.frontend(waveforms)

            if self.mode == "M0":
                logits = self.model(x)
            else:
                logits, _, _ = self.model(x)

            scores = torch.softmax(logits.float(), dim=1)[:, 1]
            clean_scores.extend(scores.cpu().numpy().tolist())
            clean_labels.extend(labels.cpu().numpy().tolist())

            degraded = []
            waveforms_np = waveforms.cpu().numpy()

            for i in range(waveforms_np.shape[0]):
                cond = conditions[i % len(conditions)]
                file_id = meta[i]["file_id"] if isinstance(meta, list) else meta["file_id"]
                deg, _ = degrade(
                    waveforms_np[i],
                    self.config["audio"]["sample_rate"],
                    cond,
                    mode="deterministic",
                    source_id=file_id,
                )
                degraded.append(deg)

            deg_w = torch.tensor(np.stack(degraded), device=self.device, dtype=torch.float32)
            x_deg = self.frontend(deg_w)

            if self.mode == "M0":
                logits_d = self.model(x_deg)
            else:
                logits_d, _, _ = self.model(x_deg)

            scores_d = torch.softmax(logits_d.float(), dim=1)[:, 1]
            deg_scores.extend(scores_d.cpu().numpy().tolist())
            deg_labels.extend(labels.cpu().numpy().tolist())

        clean_metrics, _ = calculate_metrics(clean_labels, clean_scores)
        deg_metrics, _ = calculate_metrics(deg_labels, deg_scores)
        return float(clean_metrics["macro_f1"] - deg_metrics["macro_f1"])

    def fit(self, train_loader, dev_loader, criterion, epochs, patience):
        epochs_without_improvement = 0
        best_path = self.experiment_dir / "best.pt"
        last_path = self.experiment_dir / "last.pt"

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            if hasattr(train_loader.dataset, "set_max_severity"):
                max_sev = max_severity_for_epoch(epoch, self.epoch_milestones, self.severity_milestones)
                train_loader.dataset.set_max_severity(max_sev)

            train_loss, train_acc = self.train_one_epoch(train_loader, criterion, epoch)
            dev_metrics, _, _, _ = self.evaluate_loader(dev_loader, criterion, "Dev")

            mean_delta = 0.0
            if self.use_j_score:
                mean_delta = self.estimate_dev_delta_macro_f1(
                    dev_loader,
                    max_batches=self.config.get("selection", {}).get("delta_max_batches", 5),
                )

            j_score = dev_metrics["macro_f1"] - self.lambda_delta * mean_delta
            selection_score = j_score if self.use_j_score else dev_metrics["macro_f1"]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_metrics["loss"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_auroc": dev_metrics["auroc"],
                "dev_eer": dev_metrics["eer"],
                "dev_accuracy": dev_metrics["accuracy"],
                "mean_delta_macro_f1": mean_delta,
                "best_J": selection_score,
                "lr": self.optimizer.param_groups[0]["lr"],
                "mode": self.mode,
            }

            self.history.append(row)
            append_train_log(self.experiment_dir, row)
            save_training_history(self.experiment_dir, self.history)

            logger.info(
                "Epoch %s/%s train_loss=%.4f dev_macro_f1=%.4f J=%.4f",
                epoch,
                epochs,
                train_loss,
                dev_metrics["macro_f1"],
                selection_score,
            )

            self.scheduler.step(dev_metrics["macro_f1"])

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
                "mode": self.mode,
                "mel_mean": self.frontend.mel_mean,
                "mel_std": self.frontend.mel_std,
                "best_score": selection_score,
            }

            if self.channel_head is not None:
                checkpoint["channel_head_state_dict"] = self.channel_head.state_dict()

            torch.save(checkpoint, last_path)

            if selection_score > self.best_score:
                self.best_score = selection_score
                self.best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(checkpoint, best_path)
                logger.info("  -> saved new best checkpoint (score=%.4f)", selection_score)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %s", epoch)
                break

        history_df = pd.DataFrame(self.history)
        save_training_curves(history_df, self.figure_dir, self.experiment_dir.name)

        return best_path, last_path
