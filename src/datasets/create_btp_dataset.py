import os
import hashlib
import pandas as pd


SEED=20261005

PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))
BASE_DIR=os.path.join(PROJECT_ROOT,"data")
META_DIR=os.path.join(BASE_DIR,"metadata/meta_label")
MANIFEST_DIR=os.path.join(BASE_DIR,"manifests")

TRAIN_META=os.path.join(META_DIR,"t2_train_meta.csv")
DEV_META=os.path.join(META_DIR,"t2_dev_meta.csv")
EVAL_META=os.path.join(META_DIR,"t2_eval_meta.csv")

FAKE_PER_GENERATOR=1000

TRAIN_REAL_PER_TYPE={
    "speech":4000,
    "sound":4000,
    "singing":4000,
    "music":4000
}

DEV_REAL_PER_TYPE={
    "speech":2000,
    "sound":2000,
    "singing":2000,
    "music":536
}

EVAL_REAL_PER_TYPE={
    "speech":2000,
    "sound":2000,
    "singing":2000,
    "music":536
}

AUDIO_TYPES=["speech","sound","singing","music"]


def load_metadata(path):
    print(f"\nLoading: {path}")
    df=pd.read_csv(path)
    print(f"Shape: {df.shape}")
    return df


def check_generator_availability(df,split_name):
    print("\n========================================")
    print(f"{split_name.upper()} GENERATOR AVAILABILITY")
    print("========================================")

    problems=[]

    for audio_type in AUDIO_TYPES:
        fake=df[(df["type"]==audio_type)&(df["label"]=="fake")].copy()
        generators=sorted(set(fake["generator"])-{"-"})

        print(f"\n{audio_type}: {len(generators)} generators")

        for generator in generators:
            count=len(fake[fake["generator"]==generator])

            if count<FAKE_PER_GENERATOR:
                problems.append((audio_type,generator,count))
                print(f"WARNING: {generator}: {count}")
            else:
                print(f"{generator}: {count}")

    return problems


def select_fake(df):
    selected=[]

    for audio_type in AUDIO_TYPES:
        type_df=df[(df["type"]==audio_type)&(df["label"]=="fake")].copy()
        generators=sorted(set(type_df["generator"])-{"-"})

        for generator in generators:
            generator_df=type_df[type_df["generator"]==generator].copy()

            available=len(generator_df)
            target=min(FAKE_PER_GENERATOR,available)

            if target<FAKE_PER_GENERATOR:
                print(f"WARNING: {audio_type} / {generator}: using {target} samples")

            generator_df=generator_df.sort_values("name",kind="mergesort")
            sampled=generator_df.sample(n=target,random_state=SEED)
            sampled=sampled.sort_values("name",kind="mergesort")

            selected.append(sampled)

    return pd.concat(selected,ignore_index=True)


def select_real(df,counts):
    selected=[]

    for audio_type in AUDIO_TYPES:
        type_df=df[(df["type"]==audio_type)&(df["label"]=="real")].copy()

        required=counts[audio_type]

        if len(type_df)<required:
            raise ValueError(f"Not enough real samples for {audio_type}: {len(type_df)} available, {required} required")

        type_df=type_df.sort_values("name",kind="mergesort")
        sampled=type_df.sample(n=required,random_state=SEED)
        sampled=sampled.sort_values("name",kind="mergesort")

        selected.append(sampled)

    return pd.concat(selected,ignore_index=True)


def create_split(df,real_counts,split_name):
    print("\n========================================")
    print(f"CREATING {split_name.upper()} SPLIT")
    print("========================================")

    fake=select_fake(df)
    real=select_real(df,real_counts)

    split=pd.concat([fake,real],ignore_index=True)

    split["split"]=split_name

    split=split.sort_values(["type","label","generator","name"],kind="mergesort")
    split=split.reset_index(drop=True)

    return split


def print_type_label_breakdown(df,split_name):
    print("\n========================================")
    print(f"{split_name.upper()} TYPE / LABEL BREAKDOWN")
    print("========================================")

    table=df.groupby(["type","label"]).size().unstack(fill_value=0)

    if "real" not in table.columns:
        table["real"]=0

    if "fake" not in table.columns:
        table["fake"]=0

    table["total"]=table["real"]+table["fake"]

    table=table[["real","fake","total"]]

    print(table.to_string())

    print(f"\nTotal {split_name}: {len(df)}")


def print_generator_breakdown(df,split_name):
    print("\n========================================")
    print(f"{split_name.upper()} FAKE GENERATOR BREAKDOWN")
    print("========================================")

    fake=df[df["label"]=="fake"]

    table=fake.groupby(["type","generator"]).size().reset_index(name="samples")

    for audio_type in AUDIO_TYPES:
        subset=table[table["type"]==audio_type]

        print(f"\n{audio_type.upper()}")

        if len(subset)==0:
            print("No fake samples.")
        else:
            print(subset[["generator","samples"]].to_string(index=False))


def print_real_breakdown(df,split_name):
    print("\n========================================")
    print(f"{split_name.upper()} REAL DATASET BREAKDOWN")
    print("========================================")

    real=df[df["label"]=="real"]

    table=real.groupby(["type","source"]).size().reset_index(name="samples")

    for audio_type in AUDIO_TYPES:
        subset=table[table["type"]==audio_type]

        print(f"\n{audio_type.upper()}")

        if len(subset)==0:
            print("No real samples.")
        else:
            print(subset[["source","samples"]].to_string(index=False))


def print_dataset_summary(train,dev,evaluate):
    print("\n\n")
    print("############################################################")
    print("# FINAL BTP DATASET BREAKDOWN")
    print("############################################################")

    total=len(train)+len(dev)+len(evaluate)

    print("\nTOTAL DATASET")
    print(f"Train: {len(train)}")
    print(f"Dev: {len(dev)}")
    print(f"Eval: {len(evaluate)}")
    print(f"Total: {total}")

    print("\nOVERALL TYPE DISTRIBUTION")

    combined=pd.concat([train,dev,evaluate],ignore_index=True)

    type_counts=combined["type"].value_counts().sort_index()

    for audio_type in AUDIO_TYPES:
        print(f"{audio_type}: {type_counts.get(audio_type,0)}")

    print("\nOVERALL LABEL DISTRIBUTION")

    label_counts=combined["label"].value_counts()

    print(f"Real: {label_counts.get('real',0)}")
    print(f"Fake: {label_counts.get('fake',0)}")

    print("\nSPLIT-WISE BREAKDOWN")

    for split_name,df in [("Train",train),("Dev",dev),("Eval",evaluate)]:
        print(f"\n{split_name}")

        table=df.groupby(["type","label"]).size().unstack(fill_value=0)

        if "real" not in table.columns:
            table["real"]=0

        if "fake" not in table.columns:
            table["fake"]=0

        table["total"]=table["real"]+table["fake"]

        print(table[["real","fake","total"]].to_string())

    print("\n############################################################")
    print("# MODEL EXPERIMENT DATA USAGE")
    print("############################################################")

    print("\nE1 - LIGHTWEIGHT CNN BASELINE")
    print("Training data: Train manifest")
    print("Validation data: Dev manifest")
    print("Evaluation data: Eval manifest")
    print("Input: Mel-spectrogram")
    print("Task: Binary real/fake classification")
    print("Audio types: Speech + Sound + Singing + Music")

    print("\nE2 - COMPACT CRNN")
    print("Training data: Same Train manifest")
    print("Validation data: Same Dev manifest")
    print("Evaluation data: Same Eval manifest")
    print("Input: Same preprocessing pipeline")
    print("Task: Binary real/fake classification")
    print("Purpose: Compare temporal modeling against E1")

    print("\nE3 - STRONG SSL REFERENCE")
    print("Training data: Same Train manifest")
    print("Validation data: Same Dev manifest")
    print("Evaluation data: Same Eval manifest")
    print("Purpose: Compare lightweight models against a stronger learned representation")

    print("\nE4 - UNSEEN GENERATOR GENERALIZATION")
    print("Training: Train generators")
    print("Validation: Dev generators")
    print("Evaluation: Completely unseen fake generators")
    print("Purpose: Measure generalization to unseen generation methods")

    print("\nIMPORTANT")
    print("The audio type is used for dataset analysis and evaluation.")
    print("It is NOT provided as an input feature to the detector.")
    print("The detector must classify real/fake without being told the audio type.")


def verify_no_generator_leakage(train,dev,evaluate):
    train_generators=set(train.loc[train["label"]=="fake","generator"])-{"-"}
    dev_generators=set(dev.loc[dev["label"]=="fake","generator"])-{"-"}
    eval_generators=set(evaluate.loc[evaluate["label"]=="fake","generator"])-{"-"}

    train_eval=train_generators&eval_generators
    dev_eval=dev_generators&eval_generators

    print("\n========================================")
    print("GENERATOR LEAKAGE CHECK")
    print("========================================")

    print(f"Train generators: {len(train_generators)}")
    print(f"Dev generators: {len(dev_generators)}")
    print(f"Eval generators: {len(eval_generators)}")

    print(f"Train ∩ Eval: {len(train_eval)}")
    print(f"Dev ∩ Eval: {len(dev_eval)}")

    if train_eval:
        raise ValueError(f"Generator leakage between train and eval: {sorted(train_eval)}")

    if dev_eval:
        raise ValueError(f"Generator leakage between dev and eval: {sorted(dev_eval)}")

    print("No generator leakage detected.")


def verify_duplicates(train,dev,evaluate):
    train_names=set(train["name"])
    dev_names=set(dev["name"])
    eval_names=set(evaluate["name"])

    train_dev=train_names&dev_names
    train_eval=train_names&eval_names
    dev_eval=dev_names&eval_names

    print("\n========================================")
    print("DUPLICATE CHECK")
    print("========================================")

    print(f"Train ∩ Dev: {len(train_dev)}")
    print(f"Train ∩ Eval: {len(train_eval)}")
    print(f"Dev ∩ Eval: {len(dev_eval)}")

    if train_dev or train_eval or dev_eval:
        raise ValueError("Duplicate audio filenames detected across splits.")

    print("No duplicate filenames detected.")


def calculate_sha256(path):
    sha256=hashlib.sha256()

    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            sha256.update(block)

    return sha256.hexdigest()


def save_manifest(df,name):
    path=os.path.join(MANIFEST_DIR,name)

    df.to_csv(path,index=False)

    print(f"\nSaved: {path}")
    print(f"Rows: {len(df)}")

    return path


def main():
    os.makedirs(MANIFEST_DIR,exist_ok=True)

    print("Loading metadata...")

    train_meta=load_metadata(TRAIN_META)
    dev_meta=load_metadata(DEV_META)
    eval_meta=load_metadata(EVAL_META)

    print("\nChecking generator availability...")

    for split_name,df in [("train",train_meta),("dev",dev_meta),("eval",eval_meta)]:
        check_generator_availability(df,split_name)

    train=create_split(train_meta,TRAIN_REAL_PER_TYPE,"train")
    dev=create_split(dev_meta,DEV_REAL_PER_TYPE,"dev")
    evaluate=create_split(eval_meta,EVAL_REAL_PER_TYPE,"eval")

    print_type_label_breakdown(train,"train")
    print_type_label_breakdown(dev,"dev")
    print_type_label_breakdown(evaluate,"eval")

    print_generator_breakdown(train,"train")
    print_generator_breakdown(dev,"dev")
    print_generator_breakdown(evaluate,"eval")

    print_real_breakdown(train,"train")
    print_real_breakdown(dev,"dev")
    print_real_breakdown(evaluate,"eval")

    verify_no_generator_leakage(train,dev,evaluate)
    verify_duplicates(train,dev,evaluate)

    print_dataset_summary(train,dev,evaluate)

    train_path=save_manifest(train,"train_manifest.csv")
    dev_path=save_manifest(dev,"dev_manifest.csv")
    eval_path=save_manifest(evaluate,"eval_manifest.csv")

    hash_path=os.path.join(MANIFEST_DIR,"MANIFEST_SHA256.txt")

    with open(hash_path,"w") as f:
        f.write(f"train_manifest.csv  {calculate_sha256(train_path)}\n")
        f.write(f"dev_manifest.csv    {calculate_sha256(dev_path)}\n")
        f.write(f"eval_manifest.csv   {calculate_sha256(eval_path)}\n")

    print(f"\nSHA256 hashes saved to: {hash_path}")

    print("\n========================================")
    print("DATASET CREATION COMPLETED")
    print("========================================")

    print("The manifests are ready.")
    print("Do NOT download the audio until the printed breakdown has been verified.")


if __name__=="__main__":
    main()
