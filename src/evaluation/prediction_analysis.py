"""Statistical analysis of saved evaluation predictions.

Reports pooled and type-averaged (AT-ADD) Macro-F1, per-type results at a single global
threshold, per-generator recall, expected calibration error, and bootstrap confidence
intervals (clip-level resampling, with paired differences against a reference model).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from src.datasets.atadd_manifest_dataset import find_project_root

TYPES = ["speech", "singing", "music", "sound"]


def eer_and_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", zero_division=0))


def type_average_f1(y, s, types, threshold):
    scores = []
    for t in TYPES:
        m = types == t
        if m.sum() and len(np.unique(y[m])) == 2:
            scores.append(macro_f1(y[m], (s[m] >= threshold).astype(int)))
    return float(np.mean(scores))


def ece(y, s, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - s[m].mean())
    return float(total)


def point_metrics(y, s, types):
    eer, thr = eer_and_threshold(y, s)
    return {
        "macro_f1": macro_f1(y, (s >= 0.5).astype(int)),
        "accuracy": float(((s >= 0.5) == y).mean()),
        "auroc": float(roc_auc_score(y, s)),
        "eer": eer,
        "eer_threshold": thr,
        "type_avg_macro_f1": type_average_f1(y, s, types, 0.5),
        "type_avg_macro_f1_at_global_eer_thr": type_average_f1(y, s, types, thr),
        "macro_f1_at_global_eer_thr": macro_f1(y, (s >= thr).astype(int)),
        "ece": ece(y, s),
        "brier": float(np.mean((s - y) ** 2)),
    }


def per_type_table(y, s, types, thr):
    rows = []
    for t in TYPES:
        m = types == t
        eer_t, thr_t = eer_and_threshold(y[m], s[m])
        pred05, predg = (s[m] >= 0.5).astype(int), (s[m] >= thr).astype(int)
        real, fake = y[m] == 0, y[m] == 1
        rows.append(
            {
                "type": t,
                "n": int(m.sum()),
                "macro_f1_0.5": macro_f1(y[m], pred05),
                "eer_own_threshold": eer_t,
                "auroc": float(roc_auc_score(y[m], s[m])),
                "macro_f1_global_thr": macro_f1(y[m], predg),
                "fpr_global_thr": float(predg[real].mean()),
                "fnr_global_thr": float(1 - predg[fake].mean()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap(preds, ref, reps, seed=0):
    """Paired clip-level bootstrap. `preds` maps model -> aligned DataFrame."""
    rng = np.random.default_rng(seed)
    any_df = next(iter(preds.values()))
    n = len(any_df)
    y = any_df["label"].to_numpy()
    types = any_df["type"].to_numpy()
    draws = {m: [] for m in preds}

    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        for m, df in preds.items():
            s = df["score"].to_numpy()[idx]
            eer, _ = eer_and_threshold(y[idx], s)
            draws[m].append(
                (
                    macro_f1(y[idx], (s >= 0.5).astype(int)),
                    eer,
                    float(roc_auc_score(y[idx], s)),
                    type_average_f1(y[idx], s, types[idx], 0.5),
                )
            )

    names = ["macro_f1", "eer", "auroc", "type_avg_macro_f1"]
    out = {}
    for m, d in draws.items():
        arr = np.array(d)
        out[m] = {k: [float(np.percentile(arr[:, j], 2.5)), float(np.percentile(arr[:, j], 97.5))] for j, k in enumerate(names)}
        if m != ref:
            diff = arr - np.array(draws[ref])
            out[m]["diff_vs_ref"] = {
                k: {
                    "mean": float(diff[:, j].mean()),
                    "ci": [float(np.percentile(diff[:, j], 2.5)), float(np.percentile(diff[:, j], 97.5))],
                    "p_not_better": float((diff[:, j] <= 0).mean()) if k != "eer" else float((diff[:, j] >= 0).mean()),
                }
                for j, k in enumerate(names)
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    parser.add_argument("--ref", default="F_M0")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--out", default="results/tables/final")
    args = parser.parse_args()

    root = find_project_root()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for exp_id in args.ids.split(","):
        df = pd.read_csv(root / "experiments" / exp_id / "eval_predictions_eval.csv").sort_values("filepath").reset_index(drop=True)
        preds[exp_id] = df

    summary, type_tables, gen_tables = {}, [], []
    for exp_id, df in preds.items():
        y, s, types = df["label"].to_numpy(), df["score"].to_numpy(), df["type"].to_numpy()
        summary[exp_id] = point_metrics(y, s, types)
        tt = per_type_table(y, s, types, summary[exp_id]["eer_threshold"])
        tt.insert(0, "model", exp_id)
        type_tables.append(tt)
        fake = df[df["label"] == 1]
        g = fake.groupby(["type", "generator"]).apply(
            lambda x: pd.Series({"n": len(x), "recall_0.5": (x["score"] >= 0.5).mean(), "recall_global_thr": (x["score"] >= summary[exp_id]["eer_threshold"]).mean(), "mean_score": x["score"].mean()}),
            include_groups=False,
        ).reset_index()
        g.insert(0, "model", exp_id)
        gen_tables.append(g)

    summary["_bootstrap_95ci"] = bootstrap(preds, args.ref, args.reps)
    (out_dir / "prediction_analysis.json").write_text(json.dumps(summary, indent=2))
    pd.concat(type_tables).to_csv(out_dir / "per_type.csv", index=False)
    pd.concat(gen_tables).to_csv(out_dir / "per_generator.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "_bootstrap_95ci"}, indent=1))


if __name__ == "__main__":
    main()
