import os
import shutil
import pandas as pd

PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))

MANIFEST_DIR=os.path.join(PROJECT_ROOT,"data","manifests")
SOURCE_DIR=os.path.join(PROJECT_ROOT,"data","raw","AT_ADD")
OUTPUT_DIR=os.path.join(PROJECT_ROOT,"data","shared","AT_ADD_BTP")

MANIFESTS={
    "train":"train_manifest.csv",
    "dev":"dev_manifest.csv",
    "eval":"eval_manifest.csv"
}


def load_manifest(split):
    path=os.path.join(MANIFEST_DIR,MANIFESTS[split])

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    df=pd.read_csv(path)

    if "name" not in df.columns:
        raise ValueError(f"'name' column missing from {path}")

    return df


def copy_split(df,split):
    source_dir=os.path.join(SOURCE_DIR,split)
    output_dir=os.path.join(OUTPUT_DIR,split)

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    os.makedirs(output_dir,exist_ok=True)

    copied=0
    missing=[]

    print(f"\n{split.upper()}")
    print(f"Source: {source_dir}")
    print(f"Destination: {output_dir}")
    print(f"Manifest entries: {len(df)}")

    for filename in df["name"]:
        filename=os.path.basename(str(filename))

        source=os.path.join(source_dir,filename)
        destination=os.path.join(output_dir,filename)

        if not os.path.isfile(source):
            missing.append(filename)
            continue

        if not os.path.isfile(destination):
            shutil.copy2(source,destination)
            copied+=1

    print(f"Copied: {copied}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nFirst missing files:")

        for filename in missing[:10]:
            print(filename)

        raise FileNotFoundError(f"{len(missing)} files from {split} manifest were not found.")

    return copied


def verify_split(df,split):
    output_dir=os.path.join(OUTPUT_DIR,split)

    missing=[]

    for filename in df["name"]:
        filename=os.path.basename(str(filename))
        path=os.path.join(output_dir,filename)

        if not os.path.isfile(path):
            missing.append(filename)

    if missing:
        raise RuntimeError(f"{len(missing)} files missing from output {split} dataset.")

    print(f"{split.upper()} verification: PASSED")


def main():
    print("==============================================")
    print("AT-ADD BTP DATASET CREATION")
    print("==============================================")

    print(f"\nOriginal dataset:")
    print(SOURCE_DIR)

    print(f"\nSharing dataset:")
    print(OUTPUT_DIR)

    print("\nThe original AT_ADD dataset will NOT be modified.")

    manifests={}

    for split in ["train","dev","eval"]:
        manifests[split]=load_manifest(split)

    print("\n==============================================")
    print("COPYING DATASET")
    print("==============================================")

    total=0

    for split in ["train","dev","eval"]:
        total+=copy_split(manifests[split],split)

    print("\n==============================================")
    print("VERIFYING DATASET")
    print("==============================================")

    for split in ["train","dev","eval"]:
        verify_split(manifests[split],split)

    print("\n==============================================")
    print("FINAL SUMMARY")
    print("==============================================")

    for split in ["train","dev","eval"]:
        print(f"{split}: {len(manifests[split])}")

    print(f"Total files: {total}")

    print("\nDataset created successfully.")
    print(f"Location: {OUTPUT_DIR}")


if __name__=="__main__":
    main()