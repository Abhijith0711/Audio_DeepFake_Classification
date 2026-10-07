import os
import pandas as pd

ROOT="data/raw/AT_ADD"
MANIFEST_DIR="data/manifests"

splits=["train","dev","eval"]

for split in splits:
    manifest_path=os.path.join(MANIFEST_DIR,f"{split}_manifest.csv")
    df=pd.read_csv(manifest_path)

    print(f"\n{split.upper()}")
    print(f"Manifest samples: {len(df)}")

    missing=[]
    found=0

    for name in df["name"]:
        path=os.path.join(ROOT,split,name)

        if os.path.isfile(path):
            found+=1
        else:
            missing.append(name)

    print(f"Files found: {found}")
    print(f"Files missing: {len(missing)}")

    if missing:
        print("First missing files:")
        for name in missing[:10]:
            print(name)

print("\nVerification complete.")
