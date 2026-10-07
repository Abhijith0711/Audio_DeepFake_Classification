import os
import pandas as pd

MANIFEST_DIR="data/manifests"
splits=["train","dev","eval"]

data={}

for split in splits:
    path=os.path.join(MANIFEST_DIR,f"{split}_manifest.csv")
    df=pd.read_csv(path)
    data[split]=df

print("="*70)
print("AT-ADD DATASET INTEGRITY CHECK")
print("="*70)

print("\n1. SPLIT COUNTS")
print("-"*70)

for split in splits:
    df=data[split]

    print(f"\n{split.upper()}")
    print(f"Total samples : {len(df)}")
    print(f"Real samples  : {(df['label']=="real").sum()}")
    print(f"Fake samples  : {(df['label']=="fake").sum()}")

print("\n\n2. AUDIO TYPE DISTRIBUTION")
print("-"*70)

for split in splits:
    df=data[split]

    print(f"\n{split.upper()}")
    print(df["type"].value_counts().sort_index())

print("\n\n3. REAL/FAKE DISTRIBUTION BY TYPE")
print("-"*70)

for split in splits:
    df=data[split]

    print(f"\n{split.upper()}")
    table=pd.crosstab(df["type"],df["label"])
    print(table)

print("\n\n4. GENERATOR DISTRIBUTION")
print("-"*70)

for split in splits:
    df=data[split]

    fake=df[df["label"]=="fake"]

    print(f"\n{split.upper()}")
    print(f"Number of fake generators: {fake['generator'].nunique()}")

    print("\nFake samples per generator:")
    print(fake["generator"].value_counts().sort_index())

print("\n\n5. GENERATOR LEAKAGE CHECK")
print("-"*70)

generators={}

for split in splits:
    fake=data[split]
    fake=fake[fake["label"]=="fake"]

    generators[split]=set(fake["generator"].dropna().unique())

for split in splits:
    print(f"{split.upper()} generators: {len(generators[split])}")

pairs=[
    ("train","dev"),
    ("train","eval"),
    ("dev","eval")
]

for a,b in pairs:
    overlap=generators[a]&generators[b]

    print(f"\n{a.upper()} ∩ {b.upper()}: {len(overlap)}")

    if overlap:
        print("Overlapping generators:")
        for generator in sorted(overlap):
            print(f"  {generator}")
    else:
        print("No generator overlap.")

print("\n\n6. DUPLICATE CHECK WITHIN SPLITS")
print("-"*70)

for split in splits:
    df=data[split]
    duplicates=df["name"].duplicated().sum()

    print(f"{split.upper()}: {duplicates} duplicate filenames")

print("\n\n7. CROSS-SPLIT DUPLICATE CHECK")
print("-"*70)

names={split:set(data[split]["name"]) for split in splits}

for a,b in pairs:
    overlap=names[a]&names[b]

    print(f"{a.upper()} ∩ {b.upper()}: {len(overlap)} duplicate filenames")

print("\n\n8. SOURCE DISTRIBUTION")
print("-"*70)

for split in splits:
    print(f"\n{split.upper()}")
    print(data[split]["source"].value_counts().head(20))

print("\n\n9. MISSING METADATA CHECK")
print("-"*70)

columns=[
    "name",
    "type",
    "label",
    "source",
    "source_id",
    "speaker_id",
    "generation_task",
    "generator"
]

for split in splits:
    df=data[split]

    print(f"\n{split.upper()}")

    for column in columns:
        missing=df[column].isna().sum()
        print(f"{column:18}: {missing}")

print("\n\n10. FINAL SUMMARY")
print("-"*70)

total=sum(len(data[split]) for split in splits)

print(f"Total selected samples : {total:,}")

for split in splits:
    df=data[split]

    print(
        f"{split.upper():5} : "
        f"{len(df):,} samples | "
        f"real={(df['label']=="real").sum():,} | "
        f"fake={(df['label']=="fake").sum():,}"
    )

print("\nIntegrity check complete.")
