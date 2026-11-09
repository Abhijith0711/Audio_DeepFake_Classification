"""Test whether simple acoustic cues alone separate real from fake clips on AT-ADD Track 2.

Each cue (native sample rate, duration, loudness, silence, bandwidth) is fed to a small
classifier trained on a stratified subset of the training list and tested on the evaluation
list. An AUROC far from 0.5 shows that the cue is a shortcut a detector could exploit.
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.datasets.atadd_manifest_dataset import audio_path_from_row, find_project_root, load_manifest_df, stratified_indices

FEATURE_GROUPS = {
    "duration": ["duration"],
    "native_sample_rate": ["native_sr"],
    "loudness": ["rms_db", "peak_db"],
    "silence": ["silence_ratio", "lead_trail_silence"],
    "bandwidth": ["rolloff95_hz", "hf_energy_ratio"],
    "all_cues": ["duration", "native_sr", "rms_db", "peak_db", "silence_ratio", "lead_trail_silence", "rolloff95_hz", "hf_energy_ratio"],
}


def extract(path):
    info = sf.info(path)
    wave, sr = sf.read(path, dtype="float32", frames=min(info.frames, int(10 * info.samplerate)))
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    eps = 1e-10
    rms = np.sqrt(np.mean(wave**2) + eps)
    frame = max(1, int(0.025 * sr))
    n = len(wave) // frame
    frames = wave[: n * frame].reshape(n, frame) if n else wave[None]
    frame_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + eps)
    silent = frame_db < (20 * np.log10(rms + eps) - 30)
    edge = max(1, int(0.1 * len(frame_db)))
    spec = np.abs(np.fft.rfft(wave * np.hanning(len(wave)))) ** 2
    cum = np.cumsum(spec) / (spec.sum() + eps)
    freqs = np.fft.rfftfreq(len(wave), 1 / sr)
    return {
        "duration": info.duration,
        "native_sr": info.samplerate,
        "rms_db": 20 * np.log10(rms),
        "peak_db": 20 * np.log10(np.max(np.abs(wave)) + eps),
        "silence_ratio": float(silent.mean()),
        "lead_trail_silence": float(silent[:edge].mean() + silent[-edge:].mean()),
        "rolloff95_hz": float(freqs[np.searchsorted(cum, 0.95)]),
        "hf_energy_ratio": float(spec[freqs > 4000].sum() / (spec.sum() + eps)),
    }


def features_for(df, root, audio_root, workers):
    paths = [str(audio_path_from_row(r, root, audio_root)) for _, r in df.iterrows()]
    with Pool(workers) as pool:
        feats = list(tqdm(pool.imap(extract, paths, chunksize=32), total=len(paths)))
    out = pd.DataFrame(feats)
    out["label"] = (df["label"] == "fake").astype(int).to_numpy()
    out["type"] = df["type"].to_numpy()
    return out


def eer(y, s):
    from src.evaluation.metrics import compute_eer

    return compute_eer(y, s)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="results/tables/final")
    args = parser.parse_args()

    root = find_project_root()
    audio_root = "AT_ADD_BTP"
    sets = {}
    for split in ("train", "eval"):
        df = load_manifest_df(root, f"data/manifests/{split}_manifest.csv", split)
        sets[split] = features_for(df.iloc[stratified_indices(df, args.n, seed=11)], root, audio_root, args.workers)

    train, test = sets["train"], sets["eval"]
    results = []
    for group, cols in FEATURE_GROUPS.items():
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(train[cols], train["label"])
        for scope in ["all"] + sorted(test["type"].unique()):
            m = test if scope == "all" else test[test["type"] == scope]
            if m["label"].nunique() < 2:
                continue
            p = clf.predict_proba(m[cols])[:, 1]
            results.append({"cue": group, "scope": scope, "auroc": roc_auc_score(m["label"], p), "eer": eer(m["label"], p), "n": len(m)})

    summary = test.groupby(["type", "label"])[["duration", "native_sr", "rms_db", "silence_ratio", "rolloff95_hz"]].mean()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "shortcut_probe.csv", index=False)
    summary.to_csv(out / "cue_statistics.csv")
    test.groupby(["label"])["native_sr"].value_counts().to_csv(out / "native_sample_rates.csv")
    print(pd.DataFrame(results).to_string())
    print(summary)


if __name__ == "__main__":
    main()
