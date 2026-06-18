# ─── IMPORTS ───

import os
import pandas as pd


# ─── CONFIGURATION ───

AMYLOID_MANIFEST = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\manifests\Fully_Processed_Amyloid_PET_Manifest_07May2026.csv"
TAU_MANIFEST = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\manifests\Fully_Processed_Tau_PET_Manifest_07May2026.csv"

OUTPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\filtered_manifests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AMYLOID_TRACER = "AV45"
TAU_TRACER = "AV1451"

BASELINE_VISIT = "bl"


# ─── MANIFEST FILTERING ───

def filter_manifest(manifest_path, tracer, output_path, modality_label):

    # LOAD MANIFESTS

    df = pd.read_csv(manifest_path)
    print(f"Total number of rows: {len(df)}")
    print(f"Total number of subjects:{df['subject_id'].nunique()}")

    # STEP 1: PARSE DATES

    df["image_date"] = pd.to_datetime(df["image_date"], dayfirst=True, errors="coerce")
    
    n_invalid = df["image_date"].isna().sum()
    if n_invalid > 0:
        print(f"Dropping {n_invalid} rows with unparseable dates.")
    
    df = df.dropna(subset=["image_date"])

    # STEP 2: FILTER TRACE

    df = df[df["image_description"].str.contains(tracer, na=False)].copy()
    print(f"After {tracer} tracer filter:")
    print(f"Total number of rows: {len(df)}")
    print(f"Total number of subjects: {df['subject_id'].nunique()}")

    # STEP 3: REMOVE DUPLICATE IMAGE IDS

    n_before_removed = len(df)

    df = df.drop_duplicates(subset=["image_id"])
    
    n_after_removed = n_before_removed - len(df)
    if n_after_removed > 0:
        print(f"Removing {n_after_removed} duplicate image_id entries.")

    # STEP 4: FILTER BASELINE VISIT 

    df = df[df["image_visit"] == BASELINE_VISIT].copy()
    print(f"After baseline visit filter:")
    print(f"Total number of rows: {len(df)}")
    print(f"Total number of subjects: {df['subject_id'].nunique()}")

    # STEP 5: MAINTAIN SINGLE SCAN PER SUBJECT 

    df = df.loc[df.groupby("subject_id")["image_date"].idxmin()].copy().reset_index(drop=True)

    # VALIDATION
    vc = df["subject_id"].value_counts()

    print(f"Final filtered manifest:")
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print(f"  Scans: {len(df)}")
    print(f"  Date range: from {df['image_date'].min().date()} to {df['image_date'].max().date()}")

    assert vc.max() == 1, (
        f"ERROR: Some subjects have more than one scan:"
        f"{vc[vc > 1]}"
    )
    assert vc.min() == 1, (
        "ERROR: Some subjects do not have any scan."
    )

    print(f"Validation passed.")

    # SAVE RESULT 

    df.to_csv(output_path, index=False)
    print(f"Final filtered manifest saved: {output_path}")

    return df


# ─── MAIN ORCHESTRATOR ───

def main():

    amyloid_output = os.path.join(OUTPUT_DIR, "amyloid_filtered_manifest.csv")
    amyloid_df = filter_manifest(
        manifest_path=AMYLOID_MANIFEST,
        tracer=AMYLOID_TRACER,
        output_path=amyloid_output,
        modality_label="AMYLOID",
    )

    tau_output = os.path.join(OUTPUT_DIR, "tau_filtered_manifest.csv")
    tau_df = filter_manifest(
        manifest_path=TAU_MANIFEST,
        tracer=TAU_TRACER,
        output_path=tau_output,
        modality_label="TAU",
    )

    print(f"Manifest filtering completed.")

if __name__ == "__main__":
    main()