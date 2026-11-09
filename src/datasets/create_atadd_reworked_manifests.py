"""Generate strictly speaker-disjoint AND generator-disjoint train, dev, and eval manifests.

This script creates the reworked AT-ADD Track 2 manifests inside:
    data/manifests/AT_ADDReworked/

Guarantees & Invariants:
1. Generator Disjointness (Fake Audio):
   - Train Fake: 21 generators (44,000 clips)
   - Dev Fake:    8 holdout generators (16,000 clips)
   - Eval Fake:  38 unseen generators (37,753 clips)
   - Train ∩ Dev = 0, Train ∩ Eval = 0, Dev ∩ Eval = 0 (100% disjoint)

2. Speaker Disjointness (Real Audio):
   - Speech (8,000 clips) and Singing (8,000 clips) speakers partitioned with zero overlap:
     * LJSpeech single-speaker (1,495 clips) assigned entirely to Train.
     * EchoFake speakers (163 speakers, 232 clips) assigned entirely to Eval.
     * Remaining 3,170 speech speakers partitioned using MILP to hit exact quotas
       (Train: 4,000, Dev: 2,000, Eval: 2,000) while maximizing affinity to original splits.
     * Singing speakers (22 speakers) partitioned strictly (Train: 7 speakers / 4,000 clips,
       Dev: 4 speakers / 2,000 clips, Eval: 11 speakers / 2,000 clips).
   - Train ∩ Dev = 0, Train ∩ Eval = 0, Dev ∩ Eval = 0 (100% disjoint)

3. Source ID Disjointness:
   - Zero shared source IDs across all splits (Train ∩ Dev = 0, Train ∩ Eval = 0, Dev ∩ Eval = 0).
   - Resolves the 1 cross-domain source_id ('tPVsTFGWU5Q') by swapping sound clips.

4. Domain Balance Quotas:
   - Train: 4,000 speech, 4,000 sound, 4,000 singing, 4,000 music (16,000 real) + 44,000 fake = 60,000 clips
   - Dev:    2,000 speech, 2,000 sound, 2,000 singing,   536 music ( 6,536 real) + 16,000 fake = 22,536 clips
   - Eval:   2,000 speech, 2,000 sound, 2,000 singing,   536 music ( 6,536 real) + 37,753 fake = 44,289 clips

5. Physical Disk Integrity:
   - 100% of audio files exist on physical disk under data/raw/AT_ADD/{folder}/{name}.
"""

import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def create_reworked_manifests():
    project_root = Path(__file__).resolve().parents[2]
    src_manifest_dir = project_root / "data" / "manifests" / "AT_ADD"
    raw_dir = project_root / "data" / "raw" / "AT_ADD"
    out_dir = project_root / "data" / "manifests" / "AT_ADDReworked"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Reading source manifests from: {src_manifest_dir}")
    print(f"Writing reworked manifests to: {out_dir}")

    # 1. Load source manifests from AT_ADD
    df_tr = pd.read_csv(src_manifest_dir / "train_manifest.csv")
    df_dev = pd.read_csv(src_manifest_dir / "dev_manifest.csv")
    df_ev = pd.read_csv(src_manifest_dir / "eval_manifest.csv")

    all_df = pd.concat([df_tr, df_dev, df_ev], ignore_index=True)
    print(f"Loaded {len(all_df):,} total clips across source manifests.")

    # =========================================================================
    # PART A: FAKE AUDIO (Generators already 100% disjoint)
    # =========================================================================
    train_fake = df_tr[df_tr["label"] == "fake"].copy()
    dev_fake = df_dev[df_dev["label"] == "fake"].copy()
    eval_fake = df_ev[df_ev["label"] == "fake"].copy()

    # =========================================================================
    # PART B: REAL AUDIO POOLING AND DISJOINT PARTITIONING
    # =========================================================================
    real_all = all_df[all_df["label"] == "real"].copy()

    # B1. Singing (8,000 clips: already 100% speaker disjoint across 22 speakers)
    # Train: 7 speakers (4,000 clips)
    # Dev:   4 speakers (2,000 clips)
    # Eval: 11 speakers (2,000 clips)
    singing_all = real_all[real_all["type"] == "singing"].copy()
    singing_spk_tr = set(df_tr[(df_tr["label"] == "real") & (df_tr["type"] == "singing")]["speaker_id"].unique())
    singing_spk_dev = set(df_dev[(df_dev["label"] == "real") & (df_dev["type"] == "singing")]["speaker_id"].unique())
    singing_spk_ev = set(df_ev[(df_ev["label"] == "real") & (df_ev["type"] == "singing")]["speaker_id"].unique())

    train_singing = singing_all[singing_all["speaker_id"].isin(singing_spk_tr)].copy()
    dev_singing = singing_all[singing_all["speaker_id"].isin(singing_spk_dev)].copy()
    eval_singing = singing_all[singing_all["speaker_id"].isin(singing_spk_ev)].copy()

    # B2. Music (5,072 clips: Train 4,000, Dev 536, Eval 536)
    train_music = df_tr[(df_tr["label"] == "real") & (df_tr["type"] == "music")].copy()
    dev_music = df_dev[(df_dev["label"] == "real") & (df_dev["type"] == "music")].copy()
    eval_music = df_ev[(df_ev["label"] == "real") & (df_ev["type"] == "music")].copy()

    # B3. Sound (8,000 clips: Train 4,000, Dev 2,000, Eval 2,000)
    # Resolve 1 shared source_id ('tPVsTFGWU5Q') between Eval Sound and Train Music:
    # Swap ATADD_T2_Eval_006558.flac (Eval Sound) with ATADD_T2_Train_049578.flac (Train Sound)
    train_sound = df_tr[(df_tr["label"] == "real") & (df_tr["type"] == "sound")].copy()
    dev_sound = df_dev[(df_dev["label"] == "real") & (df_dev["type"] == "sound")].copy()
    eval_sound = df_ev[(df_ev["label"] == "real") & (df_ev["type"] == "sound")].copy()

    sound_swap_train = train_sound[train_sound["name"] == "ATADD_T2_Train_049578.flac"].copy()
    sound_swap_eval = eval_sound[eval_sound["name"] == "ATADD_T2_Eval_006558.flac"].copy()

    train_sound = train_sound[train_sound["name"] != "ATADD_T2_Train_049578.flac"]
    train_sound = pd.concat([train_sound, sound_swap_eval], ignore_index=True)

    eval_sound = eval_sound[eval_sound["name"] != "ATADD_T2_Eval_006558.flac"]
    eval_sound = pd.concat([eval_sound, sound_swap_train], ignore_index=True)

    # B4. Speech (8,000 clips: Train 4,000, Dev 2,000, Eval 2,000)
    # Strictly partition all speakers using MILP:
    # - LJSpeech (single speaker Linda Johnson, 1,495 clips) -> Train
    # - EchoFake speakers (163 speakers, 232 clips) -> Eval (prevents source_id & speaker leakage)
    # - Remaining 3,170 speakers partitioned to hit exact quotas (Train: 2505, Dev: 2000, Eval: 1768)
    real_speech = real_all[real_all["type"] == "speech"].copy()
    real_speech.loc[real_speech["source"] == "LJSpeech", "speaker_id"] = "LJSpeech_single_speaker"

    ef_spks = set(real_speech[real_speech["source"] == "EchoFake"]["speaker_id"])

    spk_summary = []
    for spk, grp in real_speech.groupby("speaker_id"):
        spk_summary.append({
            "speaker_id": spk,
            "count": len(grp),
            "source": grp["source"].iloc[0],
            "is_echofake": spk in ef_spks,
            "is_ljspeech": spk == "LJSpeech_single_speaker",
        })
    df_spk = pd.DataFrame(spk_summary)

    ljs_count = df_spk.loc[df_spk["is_ljspeech"], "count"].sum()
    ef_count = df_spk.loc[df_spk["is_echofake"], "count"].sum()

    rem_spk = df_spk[~df_spk["is_ljspeech"] & ~df_spk["is_echofake"]].copy().reset_index(drop=True)
    n = len(rem_spk)
    counts = rem_spk["count"].values

    target_train = 4000 - ljs_count   # 2,505
    target_dev = 2000                 # 2,000
    target_eval = 2000 - ef_count     # 1,768

    # Objective: retain affinity to original splits
    spk_orig_split = {}
    for spk, grp in real_speech[~real_speech["source"].isin(["LJSpeech"])].groupby("speaker_id"):
        spk_orig_split[spk] = grp["split"].mode().iloc[0]

    c = np.zeros(2 * n)
    for i in range(n):
        orig = spk_orig_split[rem_spk.loc[i, "speaker_id"]]
        if orig == "train":
            c[i] = -1.0
        elif orig == "dev":
            c[n + i] = -1.0
        elif orig == "eval":
            c[i] = 1.0
            c[n + i] = 1.0

    A_eq = np.zeros((2, 2 * n))
    A_eq[0, :n] = counts
    A_eq[1, n:] = counts
    b_eq_l = np.array([target_train, target_dev], dtype=float)
    b_eq_u = np.array([target_train, target_dev], dtype=float)

    A_sum = np.zeros((n, 2 * n))
    for i in range(n):
        A_sum[i, i] = 1.0
        A_sum[i, n + i] = 1.0
    b_sum_l = np.zeros(n, dtype=float)
    b_sum_u = np.ones(n, dtype=float)

    A = np.vstack([A_eq, A_sum])
    b_l = np.concatenate([b_eq_l, b_sum_l])
    b_u = np.concatenate([b_eq_u, b_sum_u])

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones(2 * n)
    bounds = Bounds(np.zeros(2 * n), np.ones(2 * n))

    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
    assert res.success, f"MILP partition failed: {res.message}"

    x_sol = np.round(res.x[:n]).astype(int)
    y_sol = np.round(res.x[n:]).astype(int)

    rem_spk["assigned_split"] = np.where(x_sol == 1, "train", np.where(y_sol == 1, "dev", "eval"))

    ljs_df = df_spk[df_spk["is_ljspeech"]].copy().assign(assigned_split="train")
    ef_df = df_spk[df_spk["is_echofake"]].copy().assign(assigned_split="eval")

    df_all_spk = pd.concat([rem_spk, ljs_df, ef_df], ignore_index=True)

    # Restore LJSpeech speaker_id back to 'single-speaker'
    real_speech.loc[real_speech["source"] == "LJSpeech", "speaker_id"] = "single-speaker"
    df_all_spk.loc[df_all_spk["source"] == "LJSpeech", "speaker_id"] = "single-speaker"

    spk_to_split = dict(zip(df_all_spk["speaker_id"], df_all_spk["assigned_split"]))
    real_speech["assigned_split"] = real_speech.apply(
        lambda r: "train" if r["source"] == "LJSpeech" else spk_to_split[r["speaker_id"]],
        axis=1
    )

    train_speech = real_speech[real_speech["assigned_split"] == "train"].copy()
    dev_speech = real_speech[real_speech["assigned_split"] == "dev"].copy()
    eval_speech = real_speech[real_speech["assigned_split"] == "eval"].copy()

    # =========================================================================
    # PART C: COMBINE AND FORMAT MANIFESTS
    # =========================================================================
    train_real = pd.concat([train_speech, train_sound, train_singing, train_music], ignore_index=True)
    dev_real = pd.concat([dev_speech, dev_sound, dev_singing, dev_music], ignore_index=True)
    eval_real = pd.concat([eval_speech, eval_sound, eval_singing, eval_music], ignore_index=True)

    new_train = pd.concat([train_fake, train_real], ignore_index=True)
    new_train["split"] = "train"

    new_dev = pd.concat([dev_fake, dev_real], ignore_index=True)
    new_dev["split"] = "dev"

    new_eval = pd.concat([eval_fake, eval_real], ignore_index=True)
    new_eval["split"] = "eval"

    cols = [
        "name",
        "type",
        "label",
        "source",
        "source_id",
        "speaker_id",
        "generation_task",
        "generator",
        "caption",
        "split",
        "folder",
        "audio_path",
    ]

    for df in [new_train, new_dev, new_eval]:
        df["audio_path"] = "data/raw/AT_ADD/" + df["folder"] + "/" + df["name"]

    new_train = new_train[cols]
    new_dev = new_dev[cols]
    new_eval = new_eval[cols]

    print("\n--- NEW SPLIT ROW COUNTS ---")
    print(f"Train: {len(new_train):,} (Fake: {len(train_fake):,}, Real: {len(train_real):,})")
    print(f"Dev:   {len(new_dev):,} (Fake: {len(dev_fake):,}, Real: {len(dev_real):,})")
    print(f"Eval:  {len(new_eval):,} (Fake: {len(eval_fake):,}, Real: {len(eval_real):,})")

    # =========================================================================
    # PART D: COMPREHENSIVE VERIFICATION & AUDIT
    # =========================================================================
    # 1. Generator Disjointness (Fake Audio)
    g_tr = set(new_train[new_train["label"] == "fake"]["generator"].unique())
    g_dev = set(new_dev[new_dev["label"] == "fake"]["generator"].unique())
    g_ev = set(new_eval[new_eval["label"] == "fake"]["generator"].unique())

    print("\n=== GENERATOR DISJOINTNESS VERIFICATION (FAKE AUDIO) ===")
    print(f"Train fake generators ({len(g_tr)}): {sorted(g_tr)}")
    print(f"Dev fake generators   ({len(g_dev)}): {sorted(g_dev)}")
    print(f"Eval fake generators  ({len(g_ev)}): {sorted(g_ev)}")
    print(f"Train ∩ Dev:  {len(g_tr & g_dev)} overlapping generators")
    print(f"Train ∩ Eval: {len(g_tr & g_ev)} overlapping generators")
    print(f"Dev ∩ Eval:   {len(g_dev & g_ev)} overlapping generators")
    assert len(g_tr & g_dev) == 0, "Train and Dev share generators!"
    assert len(g_tr & g_ev) == 0, "Train and Eval share generators!"
    assert len(g_dev & g_ev) == 0, "Dev and Eval share generators!"
    print("PASS: 100% pairwise generator-disjoint!")

    # 2. Speaker Disjointness (Real Audio)
    s_tr = set(new_train[new_train["label"] == "real"]["speaker_id"].unique()) - {"-"}
    s_dev = set(new_dev[new_dev["label"] == "real"]["speaker_id"].unique()) - {"-"}
    s_ev = set(new_eval[new_eval["label"] == "real"]["speaker_id"].unique()) - {"-"}

    print("\n=== SPEAKER DISJOINTNESS VERIFICATION (REAL AUDIO) ===")
    print(f"Train unique real speakers: {len(s_tr)}")
    print(f"Dev unique real speakers:   {len(s_dev)}")
    print(f"Eval unique real speakers:  {len(s_ev)}")
    print(f"Train ∩ Dev:  {len(s_tr & s_dev)} overlapping speakers")
    print(f"Train ∩ Eval: {len(s_tr & s_ev)} overlapping speakers")
    print(f"Dev ∩ Eval:   {len(s_dev & s_ev)} overlapping speakers")
    assert len(s_tr & s_dev) == 0, "Train and Dev share speakers!"
    assert len(s_tr & s_ev) == 0, "Train and Eval share speakers!"
    assert len(s_dev & s_ev) == 0, "Dev and Eval share speakers!"
    print("PASS: 100% pairwise speaker-disjoint (real audio)!")

    # 2.5 Speaker Disjointness Across All Rows (Fake + Real)
    s_tr_all = set(new_train["speaker_id"].dropna().unique()) - {"-"}
    s_dev_all = set(new_dev["speaker_id"].dropna().unique()) - {"-"}
    s_ev_all = set(new_eval["speaker_id"].dropna().unique()) - {"-"}
    print(f"\nAll-label speaker overlap: Tr-Dev={len(s_tr_all & s_dev_all)}, Tr-Ev={len(s_tr_all & s_ev_all)}, Dev-Ev={len(s_dev_all & s_ev_all)}")
    assert len(s_tr_all & s_dev_all) == 0
    assert len(s_tr_all & s_ev_all) == 0
    assert len(s_dev_all & s_ev_all) == 0
    print("PASS: 100% pairwise speaker-disjoint across all labels!")

    # 3. Source ID Disjointness
    src_tr = set(new_train["source_id"].unique()) - {"-", "single-speaker"}
    src_dev = set(new_dev["source_id"].unique()) - {"-", "single-speaker"}
    src_ev = set(new_eval["source_id"].unique()) - {"-", "single-speaker"}

    print("\n=== SOURCE ID DISJOINTNESS VERIFICATION ===")
    print(f"Train ∩ Dev:  {len(src_tr & src_dev)} overlapping source_ids")
    print(f"Train ∩ Eval: {len(src_tr & src_ev)} overlapping source_ids")
    print(f"Dev ∩ Eval:   {len(src_dev & src_ev)} overlapping source_ids")
    assert len(src_tr & src_dev) == 0, "Train and Dev share source_ids!"
    assert len(src_tr & src_ev) == 0, "Train and Eval share source_ids!"
    assert len(src_dev & src_ev) == 0, "Dev and Eval share source_ids!"
    print("PASS: 100% pairwise source_id-disjoint!")

    # 4. Clip Disjointness
    c_tr = set(new_train["name"].unique())
    c_dev = set(new_dev["name"].unique())
    c_ev = set(new_eval["name"].unique())

    print("\n=== CLIP DISJOINTNESS VERIFICATION ===")
    print(f"Train ∩ Dev:  {len(c_tr & c_dev)} overlapping clips")
    print(f"Train ∩ Eval: {len(c_tr & c_ev)} overlapping clips")
    print(f"Dev ∩ Eval:   {len(c_dev & c_ev)} overlapping clips")
    assert len(c_tr & c_dev) == 0, "Train and Dev share clips!"
    assert len(c_tr & c_ev) == 0, "Train and Eval share clips!"
    assert len(c_dev & c_ev) == 0, "Dev and Eval share clips!"
    print("PASS: 100% pairwise clip-disjoint!")

    # 5. Physical Disk Integrity Check
    print("\n=== PHYSICAL DISK INTEGRITY CHECK ===")
    disk_files = {
        "train": set(os.listdir(raw_dir / "train")),
        "dev": set(os.listdir(raw_dir / "dev")),
        "eval": set(os.listdir(raw_dir / "eval")),
    }
    all_dfs = [("train", new_train), ("dev", new_dev), ("eval", new_eval)]
    for name, df in all_dfs:
        missing = [row["name"] for _, row in df.iterrows() if row["name"] not in disk_files[row["folder"]]]
        print(f"{name.capitalize()} split: {len(df):,} clips, {len(missing)} missing files")
        assert len(missing) == 0, f"Missing files found in {name}: {missing[:5]}"
    print("PASS: 100% of files physically exist on disk in their designated folders!")

    # =========================================================================
    # PART E: SAVE MANIFEST FILES AND CHECKSUMS
    # =========================================================================
    train_path = out_dir / "train_manifest.csv"
    dev_path = out_dir / "dev_manifest.csv"
    eval_path = out_dir / "eval_manifest.csv"

    new_train.to_csv(train_path, index=False)
    new_dev.to_csv(dev_path, index=False)
    new_eval.to_csv(eval_path, index=False)

    print(f"\nWrote {train_path} ({len(new_train):,} rows)")
    print(f"Wrote {dev_path} ({len(new_dev):,} rows)")
    print(f"Wrote {eval_path} ({len(new_eval):,} rows)")

    # Write SHA-256 Checksums
    sha_path = out_dir / "MANIFEST_SHA256.txt"
    with open(sha_path, "w", encoding="utf-8") as f:
        for p in [train_path, dev_path, eval_path]:
            h = compute_sha256(p)
            f.write(f"{h}  {p.name}\n")
            print(f"SHA256 ({p.name}): {h}")
    print(f"Wrote checksums to {sha_path}")
    print("\nSuccessfully generated all reworked manifests in data/manifests/AT_ADDReworked/!")


if __name__ == "__main__":
    create_reworked_manifests()
