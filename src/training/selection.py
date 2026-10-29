"""Checkpoint selection for FT-RCL and the clean baseline.

The same score is used for every mode, so M0, M1, M2, and M3 stay comparable.
"""

import numpy as np
import torch

from src.datasets.atadd_manifest_dataset import stratified_indices
from src.evaluation.metrics import calculate_metrics
from src.evaluation.prediction_analysis import type_average_f1
from src.robustness.conditions import all_training_conditions
from src.robustness.degrade import degrade


def checkpoint_is_eligible(epoch, mode, eligible_from_final_stage, final_stage_epoch):
    """Whether this epoch may replace the saved checkpoint.

    M0 is always eligible. M1, M2, and M3 are eligible from epoch 1 unless
    ``eligible_from_final_stage`` is set, in which case eligibility starts at
    ``final_stage_epoch`` (the milestone where the hardest severity turns on).
    """
    if mode == "M0" or not eligible_from_final_stage:
        return True
    return int(epoch) >= int(final_stage_epoch)


def select_degraded_dev_indices(df, num_samples, seed):
    """Row positions for the degraded development subset.

    Drawn from the full development table, stratified by type and label, with a
    fixed seed. This is not the leading rows of a sorted manifest.
    """
    return stratified_indices(df, num_samples, seed=seed)


def selection_score(clean_score, degraded_score=None, use_degraded_dev=False):
    """Score used to keep a checkpoint. The same rule is used for every mode.

    ``clean_score`` is development Macro-F1, or type-averaged Macro-F1 when
    ``selection.clean_metric`` is ``type_avg_macro_f1``.

    With ``selection.use_degraded_dev`` off, the score is ``clean_score`` on the
    full development loader.

    With it on, both numbers are that same metric on one stratified subset of
    the full development list. The subset is drawn once (fixed seed, balanced by
    type and label) and is not the first rows of the sorted manifest. Each clip
    is also passed through one training degradation, assigned in a fixed
    round-robin over the non-identity conditions. The checkpoint score is the
    unweighted mean::

        score = (clean_score + degraded_score) / 2
    """
    clean = float(clean_score)
    if not use_degraded_dev:
        return clean
    if degraded_score is None:
        raise ValueError("use_degraded_dev requires a degraded development score")
    return 0.5 * (clean + float(degraded_score))


def metric_value(labels, scores, types, clean_metric):
    """Macro-F1 or type-averaged Macro-F1 at threshold 0.5."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    metrics, _ = calculate_metrics(labels, scores)
    types = np.asarray(types)
    if len(types) == len(labels) and len(labels):
        metrics["type_avg_macro_f1"] = type_average_f1(labels, scores, types, 0.5)
    if clean_metric not in ("macro_f1", "type_avg_macro_f1"):
        raise ValueError(
            f"selection.clean_metric must be 'macro_f1' or 'type_avg_macro_f1', got {clean_metric!r}"
        )
    value = metrics.get(clean_metric)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError(f"Could not compute selection metric {clean_metric}")
    return float(value)


def match_length(wave, length):
    wave = np.asarray(wave, dtype=np.float32)
    if len(wave) >= length:
        return wave[:length]
    if len(wave) == 0:
        return np.zeros(length, dtype=np.float32)
    return np.pad(wave, (0, length - len(wave)))


def cache_degraded_dev(dataset, num_samples, seed, conditions=None):
    """Clean and degraded waveforms for a fixed stratified development subset.

    ``conditions`` defaults to every non-identity training condition. Clips are
    assigned round-robin, in that order, so the assignment does not depend on
    the epoch.
    """
    if not hasattr(dataset, "df"):
        raise TypeError("Degraded development scoring needs a dataset with a manifest frame")

    indices = select_degraded_dev_indices(dataset.df, num_samples, seed)
    pool = list(conditions) if conditions is not None else [
        c for c in all_training_conditions() if c.family != "identity"
    ]
    if not pool:
        raise ValueError("Degraded development scoring needs at least one condition")

    clean_waves = []
    degraded_waves = []
    labels = []
    types = []

    for order, index in enumerate(indices):
        wave, label, meta = dataset[index]
        if torch.is_tensor(wave):
            wave_np = wave.detach().cpu().numpy()
        else:
            wave_np = np.asarray(wave)
        wave_np = np.asarray(wave_np, dtype=np.float32)
        condition = pool[order % len(pool)]
        source_id = str(meta.get("file_id", index))
        degraded, _meta = degrade(
            wave_np,
            dataset.sample_rate,
            condition,
            mode="deterministic",
            source_id=source_id,
        )
        clean_waves.append(torch.from_numpy(wave_np))
        degraded_waves.append(torch.from_numpy(match_length(degraded, len(wave_np))))
        labels.append(int(label))
        types.append(str(meta.get("type", "unknown")))

    return {
        "indices": indices,
        "clean_waves": clean_waves,
        "degraded_waves": degraded_waves,
        "labels": labels,
        "types": types,
    }
