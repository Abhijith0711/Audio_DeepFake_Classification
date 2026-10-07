#!/usr/bin/env python3

"""
AT-ADD padding and duration diagnostics.

Run from the project root:
    python check_padding_diagnostics.py

Or:
    python check_padding_diagnostics.py --project-root ~/Audio_DeepFake_Classification
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm


def parse_args():
    parser=argparse.ArgumentParser(description="AT-ADD padding and duration diagnostics.")
    parser.add_argument("--project-root",type=str,default=".")
    parser.add_argument("--segment-seconds",type=float,default=4.0)
    parser.add_argument("--output-dir",type=str,default=None)
    return parser.parse_args()


def load_manifest(project_root,split):
    path=project_root/"data"/"manifests"/f"{split}_manifest.csv"

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df=pd.read_csv(path)

    required={"name","label","split"}
    missing=required-set(df.columns)

    if missing:
        raise ValueError(f"{split} manifest is missing columns: {sorted(missing)}")

    return df


def get_duration(path):
    try:
        return float(sf.info(path).duration)
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return np.nan


def analyze_split(df,split,project_root,target_duration):
    records=[]

    for _,row in tqdm(df.iterrows(),total=len(df),desc=f"Analyzing {split}"):
        split_name=str(row["split"])
        filename=str(row["name"])
        path=project_root/"data"/"raw"/"AT_ADD"/split_name/filename

        duration=get_duration(path)

        if np.isnan(duration):
            needs_padding=np.nan
            padding_seconds=np.nan
            padding_ratio=np.nan
        else:
            needs_padding=duration<target_duration
            padding_seconds=max(0.0,target_duration-duration)
            padding_ratio=padding_seconds/target_duration

        records.append({
            "name":filename,
            "type":row.get("type","unknown"),
            "label":row.get("label","unknown"),
            "source":row.get("source","unknown"),
            "generator":row.get("generator","unknown"),
            "split":split_name,
            "duration":duration,
            "needs_padding":needs_padding,
            "padding_seconds":padding_seconds,
            "padding_ratio":padding_ratio
        })

    return pd.DataFrame(records)


def print_split_results(df,split):
    print()
    print("="*80)
    print(f"{split.upper()} - DURATION STATISTICS")
    print("="*80)
    print(df.groupby(["type","label"])["duration"].describe().round(2).to_string())

    print()
    print("="*80)
    print(f"{split.upper()} - PADDING FREQUENCY (%)")
    print("="*80)
    print((df.groupby(["type","label"])["needs_padding"].mean()*100).round(2).to_string())

    print()
    print("Overall padding frequency by label (%)")
    print((df.groupby("label")["needs_padding"].mean()*100).round(2).to_string())

    print()
    print("="*80)
    print(f"{split.upper()} - PADDING AMOUNT")
    print("="*80)
    print(df.groupby(["type","label"])["padding_seconds"].agg(["mean","median","max"]).round(2).to_string())

    print()
    print("Padding ratio by type and label")
    print(df.groupby(["type","label"])["padding_ratio"].agg(["mean","median","max"]).round(3).to_string())

    print()
    print("="*80)
    print(f"{split.upper()} - OVERALL LABEL DURATION")
    print("="*80)
    print(df.groupby("label")["duration"].agg(["mean","median","std","min","max"]).round(2).to_string())

    print()
    print("="*80)
    print(f"{split.upper()} - REAL SOURCE PADDING")
    print("="*80)

    real_df=df[df["label"].astype(str).str.lower()=="real"]

    if len(real_df)>0:
        source_stats=real_df.groupby(["type","source"])["needs_padding"].agg(["mean","sum","count"])
        source_stats["padding_percent"]=source_stats["mean"]*100
        source_stats=source_stats.drop(columns=["mean"])
        print(source_stats.round(2).sort_values("padding_percent",ascending=False).to_string())
    else:
        print("No real samples found.")


def compare_train_eval(train_df,eval_df):
    print()
    print("="*80)
    print("TRAIN VS EVAL PADDING COMPARISON")
    print("="*80)

    train_stats=train_df.groupby(["type","label"])["needs_padding"].mean()*100
    eval_stats=eval_df.groupby(["type","label"])["needs_padding"].mean()*100

    comparison=pd.DataFrame({
        "train_padding_percent":train_stats,
        "eval_padding_percent":eval_stats
    })

    comparison["eval_minus_train"]=comparison["eval_padding_percent"]-comparison["train_padding_percent"]

    print(comparison.round(2).to_string())


def check_missing_files(df,split,project_root):
    missing=0

    for _,row in df.iterrows():
        path=project_root/"data"/"raw"/"AT_ADD"/str(row["split"])/str(row["name"])

        if not path.exists():
            missing+=1

    print()
    print(f"{split.upper()} missing files: {missing}")


def main():
    args=parse_args()

    project_root=Path(args.project_root).expanduser().resolve()
    target_duration=args.segment_seconds

    if args.output_dir is None:
        output_dir=project_root/"results"/"tables"/"padding_diagnostics"
    else:
        output_dir=Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True,exist_ok=True)

    print("="*80)
    print("AT-ADD PADDING AND DURATION DIAGNOSTICS")
    print("="*80)
    print(f"Project root: {project_root}")
    print(f"Target duration: {target_duration:.2f} seconds")
    print(f"Output directory: {output_dir}")

    datasets={}

    for split in ["train","dev","eval"]:
        df=load_manifest(project_root,split)

        print()
        print(f"Loaded {split} manifest: {len(df)} rows")

        check_missing_files(df,split,project_root)

        result=analyze_split(df,split,project_root,target_duration)
        datasets[split]=result

        output_path=output_dir/f"{split}_duration_padding_diagnostics.csv"
        result.to_csv(output_path,index=False)

        print(f"Saved: {output_path}")
        print_split_results(result,split)

    compare_train_eval(datasets["train"],datasets["eval"])

    combined=pd.concat(datasets.values(),ignore_index=True)
    combined_path=output_dir/"all_splits_duration_padding_diagnostics.csv"
    combined.to_csv(combined_path,index=False)

    print()
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for split,df in datasets.items():
        valid=df["needs_padding"].notna().sum()
        padded=df["needs_padding"].sum()
        percent=padded/valid*100 if valid else np.nan
        print(f"{split}: {len(df)} files | {int(padded)} padded | {percent:.2f}% padded")

    print()
    print(f"Combined results: {combined_path}")
    print()
    print("Use the results to compare:")
    print("1. Real vs fake padding frequency within each audio type.")
    print("2. Train vs eval padding frequency.")
    print("3. Padding amount and padding ratio.")
    print("4. Padding frequency across real sources.")
    print("5. Do not conclude causation from correlation alone.")


if __name__=="__main__":
    main()
