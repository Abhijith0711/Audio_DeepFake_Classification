#!/usr/bin/env python3

"""
ASVspoof 2019 LA dataset preparation
Pipeline version: v1.0

This script creates reproducible metadata, split files and manifests.
It does not modify the raw audio or create spectrograms.
"""

from pathlib import Path
import argparse
import hashlib
import json
import csv
from datetime import datetime,timezone


# Keep the pipeline version fixed so later changes can be tracked
PIPELINE_VERSION="v1.0"

# Labels used throughout the project
LABEL_MAP={
    "bonafide":0,
    "spoof":1
}

# Use the official ASVspoof partitions instead of creating random splits
SPLITS=["train","dev","eval"]


def get_project_root():
    # The script is stored in src/preprocessing/
    return Path(__file__).resolve().parents[2]


def get_paths(project_root):
    data_root=project_root/"data"
    raw_root=data_root/"raw"/"ASVspoof2019_LA"
    protocol_root=raw_root/"ASVspoof2019_LA_cm_protocols"

    return {
        "project_root":project_root,
        "raw_root":raw_root,

        "train_audio":raw_root/"ASVspoof2019_LA_train"/"flac",
        "dev_audio":raw_root/"ASVspoof2019_LA_dev"/"flac",
        "eval_audio":raw_root/"ASVspoof2019_LA_eval"/"flac",

        "train_protocol":protocol_root/"ASVspoof2019.LA.cm.train.trn.txt",
        "dev_protocol":protocol_root/"ASVspoof2019.LA.cm.dev.trl.txt",
        "eval_protocol":protocol_root/"ASVspoof2019.LA.cm.eval.trl.txt",

        "metadata_dir":data_root/"metadata",
        "splits_dir":data_root/"splits",
        "manifests_dir":data_root/"manifests"/"ASVSpoof",

        "experiment_dir":project_root/"experiments"/"baselines"/"E1"
    }


def sha256_file(path):
    # Hash protocol files so we can identify exactly which version was used
    sha256=hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk=f.read(1024*1024)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def write_csv(path,rows,fieldnames):
    # Save generated metadata and manifests as CSV files
    path.parent.mkdir(parents=True,exist_ok=True)

    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_path(path,project_root):
    # Store project-relative paths so the repository can be moved safely
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def parse_protocol(protocol_path,audio_dir,split_name,project_root):
    """
    Parse one official ASVspoof 2019 LA CM protocol.

    The parser is intentionally tolerant of the common LA protocol
    layout:

        speaker_id file_id ... system_id key

    The first field is treated as speaker ID.
    The second field is treated as recording/file ID.
    The final field is treated as the class key.
    The field before the final field is retained as system/generator ID.

    If an evaluation protocol contains an unknown/unavailable key,
    label is recorded as -1 rather than inventing a class label.
    """
    rows=[]
    warnings=[]

    with protocol_path.open("r",encoding="utf-8") as f:
        for line_number,line in enumerate(f,start=1):
            parts=line.strip().split()

            if not parts:
                continue

            if len(parts)<2:
                warnings.append(
                    f"{protocol_path.name}: line {line_number} has fewer than 2 fields"
                )
                continue

            # The first two fields identify the speaker and recording
            speaker_id=parts[0]
            file_id=parts[1]

            # The last field contains the class key
            key=parts[-1].lower()

            # The field before the key identifies the spoofing system when available
            system_id=parts[-2] if len(parts)>=3 else "-"

            label=LABEL_MAP.get(key,-1)

            # Build the expected FLAC path without changing the raw audio
            audio_path=audio_dir/f"{file_id}.flac"

            rows.append({
                "split":split_name,
                "file_id":file_id,
                "audio_path":relative_path(audio_path,project_root),
                "speaker_id":speaker_id,
                "system_id":system_id,
                "key":key,
                "label":label,
                "audio_exists":int(audio_path.exists()),
                "protocol_file":protocol_path.name
            })

    return rows,warnings


def find_duplicates(rows):
    # Check that the same recording does not appear twice in a partition
    seen=set()
    duplicates=[]

    for row in rows:
        file_id=row["file_id"]
        if file_id in seen:
            duplicates.append(file_id)
        seen.add(file_id)
    return sorted(set(duplicates))

def get_class_counts(rows):
    # Count bonafide, spoof and unknown entries in each partition
    counts={
        "bonafide":0,
        "spoof":0,
        "unknown":0
    }

    for row in rows:
        if row["label"]==0:
            counts["bonafide"]+=1
        elif row["label"]==1:
            counts["spoof"]+=1
        else:
            counts["unknown"]+=1
    return counts


def get_speaker_overlap(split_rows):
    # Audit speaker overlap without changing the official partitions
    train_speakers={row["speaker_id"] for row in split_rows["train"]}
    dev_speakers={row["speaker_id"] for row in split_rows["dev"]}
    eval_speakers={row["speaker_id"] for row in split_rows["eval"]}

    return {
        "train_dev":len(train_speakers&dev_speakers),
        "train_eval":len(train_speakers&eval_speakers),
        "dev_eval":len(dev_speakers&eval_speakers)
    }


def prepare_dataset(project_root):
    paths=get_paths(project_root)

    print("="*70)
    print("ASVspoof 2019 LA Dataset Preparation")
    print(f"Pipeline version: {PIPELINE_VERSION}")
    print("="*70)

    # Check that the expected dataset structure exists
    required_paths={
        "train audio":paths["train_audio"],
        "dev audio":paths["dev_audio"],
        "eval audio":paths["eval_audio"],
        "train protocol":paths["train_protocol"],
        "dev protocol":paths["dev_protocol"],
        "eval protocol":paths["eval_protocol"]
    }
    print("\nChecking dataset structure...\n")

    missing=[]

    for name,path in required_paths.items():
        status="OK" if path.exists() else "MISSING"
        print(f"{name:<18}: {status:<8} {path}")

        if not path.exists():
            missing.append((name,path))

    if missing:
        print("\nERROR: Required dataset files/directories are missing.")
        print("Fix the dataset structure before running this script.")
        return

    # Create the directories used by the reproducibility pipeline
    paths["metadata_dir"].mkdir(parents=True,exist_ok=True)
    paths["splits_dir"].mkdir(parents=True,exist_ok=True)
    paths["manifests_dir"].mkdir(parents=True,exist_ok=True)
    paths["experiment_dir"].mkdir(parents=True,exist_ok=True)

    # Connect each official split to its protocol and audio directory
    protocol_information={
        "train":(paths["train_protocol"],paths["train_audio"]),
        "dev":(paths["dev_protocol"],paths["dev_audio"]),
        "eval":(paths["eval_protocol"],paths["eval_audio"])
    }

    split_rows={}
    all_rows=[]
    all_warnings=[]

    print("\nReading official protocol files...\n")

    for split_name in SPLITS:
        protocol_path,audio_dir=protocol_information[split_name]
        rows,warnings=parse_protocol(
            protocol_path,
            audio_dir,
            split_name,
            project_root
        )

        split_rows[split_name]=rows
        all_rows.extend(rows)
        all_warnings.extend(warnings)

        print(f"{split_name:<8}: {len(rows):>8} protocol entries")

    # Check for duplicate recording IDs in each official partition
    print("\nRunning duplicate audits...\n")

    duplicate_ids=[]

    for split_name in SPLITS:
        duplicates=find_duplicates(split_rows[split_name])

        if duplicates:
            print(
                f"WARNING: {split_name} contains "
                f"{len(duplicates)} duplicate file IDs."
            )

            duplicate_ids.extend(
                [f"{split_name}:{file_id}" for file_id in duplicates]
            )
        else:
            print(f"{split_name:<8}: no duplicate file IDs")

    # Check that every protocol entry has a corresponding FLAC file
    print("\nMissing audio files:")

    missing_audio={}

    for split_name in SPLITS:
        missing=[
            row["file_id"]
            for row in split_rows[split_name]
            if row["audio_exists"]==0
        ]

        missing_audio[split_name]=missing

        print(f"{split_name:<8}: {len(missing):>8}")

    # Check speaker overlap as part of the leakage audit
    speaker_overlap=get_speaker_overlap(split_rows)

    print("\nSpeaker identity overlap:")
    print("train/dev :",speaker_overlap["train_dev"])
    print("train/eval:",speaker_overlap["train_eval"])
    print("dev/eval  :",speaker_overlap["dev_eval"])

    # Display the class distribution before training
    print("\nClass counts:")

    class_summary={}

    for split_name in SPLITS:
        counts=get_class_counts(split_rows[split_name])
        class_summary[split_name]=counts

        print(
            f"{split_name:<8}: "
            f"bonafide={counts['bonafide']}, "
            f"spoof={counts['spoof']}, "
            f"unknown={counts['unknown']}"
        )

    # Save one combined metadata file
    metadata_fields=[
        "split",
        "file_id",
        "audio_path",
        "speaker_id",
        "system_id",
        "key",
        "label",
        "audio_exists",
        "protocol_file"
    ]

    metadata_path=paths["metadata_dir"]/"asvspoof2019_la_metadata.csv"
    write_csv(metadata_path,all_rows,metadata_fields)

    # Save the official train/dev/eval split information
    for split_name in SPLITS:
        split_path=paths["splits_dir"]/f"{split_name}.csv"
        write_csv(split_path,split_rows[split_name],metadata_fields)

    # Save simplified manifests for model training
    manifest_fields=[
        "file_id",
        "audio_path",
        "speaker_id",
        "system_id",
        "key",
        "label",
        "split"
    ]

    for split_name in SPLITS:
        manifest_rows=[]

        for row in split_rows[split_name]:
            manifest_rows.append({
                "file_id":row["file_id"],
                "audio_path":row["audio_path"],
                "speaker_id":row["speaker_id"],
                "system_id":row["system_id"],
                "key":row["key"],
                "label":row["label"],
                "split":row["split"]
            })

        manifest_path=paths["manifests_dir"]/f"{split_name}_manifest.csv"

        write_csv(manifest_path,manifest_rows,manifest_fields)

    # Hash the protocol files for experiment reproducibility
    protocol_hashes={}

    for split_name in SPLITS:
        protocol_path=protocol_information[split_name][0]
        protocol_hashes[protocol_path.name]=sha256_file(protocol_path)

    # Save a record of how this dataset preparation was performed
    preparation_record={
        "pipeline_name":"ASVspoof2019_LA_preparation",
        "pipeline_version":PIPELINE_VERSION,
        "created_utc":datetime.now(timezone.utc).isoformat(),

        "dataset":{
            "name":"ASVspoof 2019 Logical Access",
            "raw_root":str(paths["raw_root"]),
            "official_partitions_used":SPLITS
        },

        "processing":{
            "random_split":False,
            "uses_official_protocol_partitions":True,
            "audio_modified":False,
            "spectrograms_created":False,
            "sample_rate_conversion":False,
            "cropping":False,
            "padding":False,
            "augmentation":False
        },

        "label_definition":{
            "bonafide":0,
            "spoof":1,
            "unknown":-1
        },

        "counts":{
            split_name:len(split_rows[split_name])
            for split_name in SPLITS
        },

        "class_counts":class_summary,
        "speaker_overlap":speaker_overlap,

        "missing_audio":{
            split_name:len(missing_audio[split_name])
            for split_name in SPLITS
        },

        "duplicate_ids":duplicate_ids,
        "protocol_sha256":protocol_hashes,
        "warnings":all_warnings
    }

    preparation_path=paths["experiment_dir"]/"dataset_preparation.json"
    with preparation_path.open("w",encoding="utf-8") as f:
        json.dump(preparation_record,f,indent=2)

    print("\n"+"="*70)
    print("DATASET PREPARATION COMPLETE")
    print("="*70)

    print("\nCreated:")
    print("Metadata :",metadata_path)
    print("Splits   :",paths["splits_dir"])
    print("Manifests:",paths["manifests_dir"])
    print("Audit    :",preparation_path)

    print("\nRaw audio was not modified.")
    print("No spectrograms were generated.")
    print("Official train/dev/eval partitions were preserved.")

    if any(missing_audio.values()):
        print("\nWARNING: Missing audio files were detected.")

    if duplicate_ids:
        print("\nWARNING: Duplicate file IDs were detected.")

    if all_warnings:
        print("\nProtocol parsing warnings:")

        for warning in all_warnings:
            print("-",warning)


def main():
    parser=argparse.ArgumentParser(
        description="Prepare reproducible ASVspoof 2019 LA metadata and manifests."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None
    )

    args=parser.parse_args()

    if args.project_root is None:
        project_root=get_project_root()
    else:
        project_root=args.project_root.resolve()
    prepare_dataset(project_root)

if __name__=="__main__":
    main()
