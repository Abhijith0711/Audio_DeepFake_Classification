import logging
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.metrics import calculate_metrics
from src.evaluation.plotting import save_training_curves
from src.evaluation.prediction_analysis import type_average_f1
from src.models.ft_rcl_detector import FTRCLDetector
from src.models.mobilenet_v3_audio import MobileNetV3SmallAudio
from src.training.losses.ft_rcl import ft_rcl_loss
from src.training.run_artifacts import (
    append_train_log,
    save_training_history,
)
from src.training.schedulers.robustness_curriculum import (
    max_severity_for_epoch,
    severity_eta,
    should_early_stop,
)

logger = logging.getLogger(__name__)


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

        if mode == "M0":
            self.model = MobileNetV3SmallAudio(num_classes=2).to(device)
            self.subband = None
        else:
            self.model = FTRCLDetector(num_classes=2).to(device)
            self.subband = self.model.subband_pool

        self.history = []
        self.best_score = -1e9
        self.best_epoch = 0

        train_cfg = config["training"]
        self.min_epoch_for_early_stop = int(train_cfg.get("min_epoch_for_early_stop", 1))
        self.amp_enabled = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
        self.gate = ft_cfg.get("gate", "full")
        self.type_alpha = ft_cfg.get(
            "type_alpha",
            {"speech": 1.0, "singing": 1.0, "music": 0.3, "sound": 0.5},
        )
        curriculum = ft_cfg.get("curriculum", {})
        self.epoch_milestones = curriculum.get("epoch_milestones", [1, 6, 13])
        self.severity_milestones = curriculum.get("severity_milestones", [1, 3, 4])

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

                # Forward passes run under autocast; every loss term is computed in FP32 below.
                with self.autocast():
                    if self.mode == "M1":
                        x_deg = self.frontend(deg_w)
                        logits, _, _ = self.model(x_deg, waveforms=deg_w, current_epoch=epoch)
                    else:
                        x_clean = self.frontend(clean_w)
                        x_deg = self.frontend(deg_w)
                        logits_c, sig_c, _ = self.model(
                            x_clean,
                            waveforms=clean_w,
                            current_epoch=epoch,
                        )
                        logits_d, sig_d, _ = self.model(
                            x_deg,
                            waveforms=deg_w,
                            current_epoch=epoch,
                        )

                if self.mode == "M1":
                    loss = criterion(logits.float(), labels)
                else:
                    logits_c, logits_d = logits_c.float(), logits_d.float()
                    sig_c, sig_d = sig_c.float(), sig_d.float()

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
                        gate=self.gate,
                    )

                    logits = logits_c

            else:
                waveforms, labels, _meta = batch
                labels = labels.to(self.device, non_blocking=True)
                waveforms = waveforms.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                with self.autocast():
                    logits = self._forward_clean(waveforms)

                loss = criterion(logits.float(), labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss ({loss.item()}) in mode {self.mode}, epoch {epoch}."
                )

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

        labels_np = np.asarray(all_labels)
        scores_np = np.asarray(all_scores)
        metrics, _ = calculate_metrics(labels_np, scores_np)
        metrics["loss"] = total_loss / max(total_items, 1)
        types = np.asarray([str(item.get("type", "unknown")) for item in all_meta])
        if len(types) == len(labels_np) and len(labels_np):
            metrics["type_avg_macro_f1"] = type_average_f1(labels_np, scores_np, types, 0.5)
        return metrics, labels_np, scores_np, all_meta

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

            selection_score = dev_metrics["macro_f1"]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_metrics["loss"],
                "dev_macro_f1": dev_metrics["macro_f1"],
                "dev_type_avg_macro_f1": dev_metrics.get("type_avg_macro_f1"),
                "dev_auroc": dev_metrics["auroc"],
                "dev_eer": dev_metrics["eer"],
                "dev_accuracy": dev_metrics["accuracy"],
                "selection_score": selection_score,
                "lr": self.optimizer.param_groups[0]["lr"],
                "mode": self.mode,
            }

            self.history.append(row)
            append_train_log(self.experiment_dir, row)
            save_training_history(self.experiment_dir, self.history)

            logger.info(
                "Epoch %s/%s train_loss=%.4f dev_macro_f1=%.4f dev_eer=%.4f",
                epoch,
                epochs,
                train_loss,
                dev_metrics["macro_f1"],
                dev_metrics["eer"],
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

            torch.save(checkpoint, last_path)

            if selection_score > self.best_score:
                self.best_score = selection_score
                self.best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(checkpoint, best_path)
                logger.info("  -> saved new best checkpoint (dev_macro_f1=%.4f)", selection_score)
            else:
                epochs_without_improvement += 1

            if should_early_stop(
                epoch,
                epochs_without_improvement,
                patience,
                self.min_epoch_for_early_stop,
            ):
                logger.info("Early stopping at epoch %s", epoch)
                break

        history_df = pd.DataFrame(self.history)
        save_training_curves(history_df, self.figure_dir, self.experiment_dir.name)

        return best_path, last_path
