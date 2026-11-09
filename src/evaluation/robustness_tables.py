"""Summarise condition-wise scores: metrics, drops from clean, aggregate drops with paired bootstrap
intervals, window-averaged scoring, and the agreement between assigned severity and measured drop."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from src.datasets.atadd_manifest_dataset import find_project_root
from src.evaluation.prediction_analysis import eer_and_threshold
from src.robustness.conditions import all_training_conditions

SEVERITY = {c.name: c.severity for c in all_training_conditions() if c.family != "identity"}


def metrics(y, s):
    return {
        "macro_f1": float(f1_score(y, (s >= 0.5).astype(int), average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(y, s)),
        "eer": eer_and_threshold(y, s)[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/tables/final")
    parser.add_argument("--ref", default="F_M0")
    parser.add_argument("--reps", type=int, default=300)
    args = parser.parse_args()
    out = find_project_root() / args.out

    df = pd.read_csv(out / "robustness_scores.csv")
    models = list(df["model"].unique())
    conds = [c for c in df["condition"].unique() if c != "clean"]
    kinds = df.drop_duplicates("condition").set_index("condition")["kind"].to_dict()

    # Align every (model, condition) by file id.
    wide = {m: {c: g.sort_values("file_id")["score"].to_numpy() for c, g in dm.groupby("condition")} for m, dm in df.groupby("model")}
    base = df[(df["model"] == models[0]) & (df["condition"] == "clean")].sort_values("file_id")
    y, types = base["label"].to_numpy(), base["type"].to_numpy()

    rows = []
    for m in models:
        clean = metrics(y, wide[m]["clean"])
        rows.append({"model": m, "condition": "clean", "kind": "clean", **clean, "d_macro_f1": 0, "d_auroc": 0, "d_eer": 0})
        for c in conds:
            r = metrics(y, wide[m][c])
            rows.append({"model": m, "condition": c, "kind": kinds[c], **r,
                         "d_macro_f1": clean["macro_f1"] - r["macro_f1"], "d_auroc": clean["auroc"] - r["auroc"], "d_eer": r["eer"] - clean["eer"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "robustness_summary.csv", index=False)

    # Per-type AUROC drop.
    trows = []
    for m in models:
        for c in conds:
            for t in sorted(set(types)):
                mask = types == t
                clean_a = roc_auc_score(y[mask], wide[m]["clean"][mask])
                cond_a = roc_auc_score(y[mask], wide[m][c][mask])
                trows.append({"model": m, "condition": c, "type": t, "d_auroc": clean_a - cond_a})
    pd.DataFrame(trows).to_csv(out / "robustness_by_type.csv", index=False)

    # Aggregate drop with paired bootstrap.
    groups = {"seen": [c for c in conds if kinds[c] == "seen"], "unseen": [c for c in conds if kinds[c] == "unseen"]}
    rng = np.random.default_rng(0)

    def mean_drop(m, idx, cs):
        yy = y[idx]
        return np.mean([roc_auc_score(yy, wide[m]["clean"][idx]) - roc_auc_score(yy, wide[m][c][idx]) for c in cs])

    agg = {"point": {}, "bootstrap": {}}
    for g, cs in groups.items():
        agg["point"][g] = {m: float(mean_drop(m, np.arange(len(y)), cs)) for m in models}
    draws = {g: {m: [] for m in models} for g in groups}
    for _ in range(args.reps):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        for g, cs in groups.items():
            for m in models:
                draws[g][m].append(mean_drop(m, idx, cs))
    for g in groups:
        agg["bootstrap"][g] = {}
        for m in models:
            d = np.array(draws[g][m])
            entry = {"ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
            if m != args.ref:
                diff = d - np.array(draws[g][args.ref])
                entry["diff_vs_ref"] = {"mean": float(diff.mean()), "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
            agg["bootstrap"][g][m] = entry

    # Severity-rank agreement (reference model, seen training conditions).
    train_conds = [c for c in groups["seen"] if c in SEVERITY]
    rho = {}
    for m in models:
        drops = [summary[(summary.model == m) & (summary.condition == c)]["d_auroc"].iloc[0] for c in train_conds]
        rho[m] = float(spearmanr([SEVERITY[c] for c in train_conds], drops).statistic)
    agg["severity_spearman"] = rho
    (out / "robustness_aggregate.json").write_text(json.dumps(agg, indent=2))

    # Window-averaged scoring.
    win = pd.read_csv(out / "multiwindow_scores.csv")
    wrows = []
    for m in models:
        w = win[win.model == m].sort_values("file_id")
        wrows.append({"model": m, "scoring": "single centre crop", **metrics(y, wide[m]["clean"])})
        wrows.append({"model": m, "scoring": "mean over up to 8 windows", **metrics(y, w["score"].to_numpy())})
    pd.DataFrame(wrows).to_csv(out / "multiwindow_summary.csv", index=False)
    print(json.dumps(agg["point"], indent=1), rho)


if __name__ == "__main__":
    main()
