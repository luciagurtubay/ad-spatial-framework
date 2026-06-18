# ─── IMPORTS ───

import os
import nibabel as nib
import pandas as pd
import numpy as np  
from nilearn.image import resample_to_img
import glob


# ─── CONFIGURATION ───

ATLAS_PATH = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\atlas\DKT_MNI152_atlas_filtered.nii"
ATLAS_LABELS = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\atlas\DKT_MNI152_atlas_label.csv"

AMYLOID_INPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_mni\amyloid_mni"
TAU_INPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_mni\tau_mni"

OUTPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\outputs\amyloid_tau_parcellation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CEREBELLUM_LABELS = {8, 47}

COMPUTE_AMYLOID_SUVR = True
COMPUTE_TAU_SUVR = True


# ─── DKT ATLAS LOADING ───

print("Loading DKT atlas...")

atlas_img = nib.load(ATLAS_PATH)
atlas_data = atlas_img.get_fdata().astype(int)

atlas_lab = pd.read_csv(ATLAS_LABELS)
atlas_lab["id"] = atlas_lab["id"].astype(int)
id_to_region = dict(zip(atlas_lab["id"], atlas_lab["structure"]))

all_labels = sorted([
    lab for lab in np.unique(atlas_data)
    if lab != 0
])
all_labels_names = [
    id_to_region.get(lab, f"label_{lab}") for lab in all_labels
]

print(f"Number of atlas regions: {len(all_labels)}")
print(f"Reference regions (cerebellum labels): {CEREBELLUM_LABELS}")


# ─── PET IMAGE REGIONAL SUVR EXTRACT ───

def extract_regional_values(pet_path, atlas_img, atlas_data, all_labels, id_to_region, compute_suvr_ratio):

    # STEP 1: LOAD AND RESAMPLE PET IMAGE TO ATLAS SPACE (LINEAR INTERPOLATION)

    try:
        pet_img = nib.load(pet_path)
        pet_resampled = resample_to_img(
            source_img=pet_img,
            target_img=atlas_img,
            interpolation="linear",  
        )
        pet_data = pet_resampled.get_fdata()
    
    except Exception as e:
        print(f"ERROR loading/resampling {os.path.basename(pet_path)}: {e}")
        return None

    # STEP 2: COMPUTE CEREBELLAR REFERENCE VALUE

    agg_fn = np.median
    reference_value = 1.0 

    if compute_suvr_ratio:
        cerebellum_voxels = []

        for cereb_label in CEREBELLUM_LABELS:
            mask = atlas_data == cereb_label
            vals = pet_data[mask]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            cerebellum_voxels.extend(vals.tolist())

        if len(cerebellum_voxels) > 0:
            reference_value = agg_fn(cerebellum_voxels)
        else:
            print(f"WARNING: No cerebellar voxels found.")
            reference_value = 1.0

    # STEP 3: EXTRACT REGIONAL VALUES

    regional_values = {}

    for label in all_labels:
        region_name = id_to_region.get(label, f"label_{label}")

        mask = atlas_data == label
        vals = pet_data[mask]
        vals = vals[np.isfinite(vals) & (vals > 0)]

        if len(vals) == 0:
            regional_values[region_name] = np.nan
        else:
            regional_value = agg_fn(vals)
            regional_values[region_name] = regional_value / reference_value

    return regional_values


# ─── SUBJECT ID EXTRACTION ───

def parse_subject_id(filename):

    base = os.path.basename(filename)
    base = base.replace(".nii.gz", "").replace(".nii", "")

    if base.startswith("MNI_"):
        base = base[4:]

    return base 


# ─── MAIN PARCELLATION FUNCTION ───

def parcellate_modality(input_dir, modality_label, atlas_img, atlas_data, all_labels, all_labels_names, id_to_region, output_dir, compute_suvr_ratio):

    print(f"Processing: {modality_label}")

    pet_files = sorted(
        glob.glob(os.path.join(input_dir, "*.nii.gz")) +
        glob.glob(os.path.join(input_dir, "*.nii"))
    )

    pet_files = sorted(set(pet_files))

    if len(pet_files) == 0:
        print(f"No PET files found. Check that {input_dir} contains .nii or .nii.gz files.")
        return None, None

    print(f"Found {len(pet_files)} PET files in {input_dir}")

    # STEP 1: INITIALIZE TRACKING STRUCTURES

    subject_results = {}
    failed = []

    for i, pet_path in enumerate(pet_files):

        subject_id = parse_subject_id(pet_path)
        print(f"[{i+1}/{len(pet_files)}] {subject_id}")

        # STEP 2: EXTRACT REGIONAL VALUES

        regional_values = extract_regional_values(
            pet_path=pet_path,
            atlas_img=atlas_img,
            atlas_data=atlas_data,
            all_labels=all_labels,
            id_to_region=id_to_region,
            compute_suvr_ratio=compute_suvr_ratio,
        )

        if regional_values is None:
            print(f"FAILED: {subject_id}")
            failed.append(subject_id)
            continue

        subject_results[subject_id] = regional_values

    print(f"Completed: {len(subject_results)} subjects.")
    if failed:
        print(f"Failed:{len(failed)} subjects: {failed}.")

    if len(subject_results) == 0:
        print("No valid results.")
        return None, None

    # STEP 3: BUILD REGIONS X SUBJECTS DATAFRAME 

    subjects_df = pd.DataFrame(
        subject_results,         
        index=all_labels_names  
    )
    subjects_df.index.name = "region"

    # STEP 4: BUILD COVERAGE (SUBJECTS WITH DATA PER REGION) DATAFRAME

    coverage = subjects_df.notna().sum(axis=1).rename("subjects_with_data")
    coverage_df = pd.DataFrame({
        "region": subjects_df.index,
        "subjects_with_data": coverage.values,
        "total_subjects": len(subject_results),
        "coverage_pct": (coverage.values / len(subject_results) * 100).round(1),
    })

    # STEP 5: BUILD REGIONS X GROUP DATAFRAME 

    group_df = pd.DataFrame({
        "region": subjects_df.index,
        "median_suvr": subjects_df.median(axis=1).values,
        "mean_suvr": subjects_df.mean(axis=1).values,
        "std_suvr": subjects_df.std(axis=1).values,
        "n_subjects": coverage.values,
    })
    group_df = group_df.set_index("region")

    subject_path = os.path.join(output_dir, f"individual_region_{modality_label}.csv")
    coverage_path = os.path.join(output_dir, f"coverage_{modality_label}.csv")
    group_path = os.path.join(output_dir, f"group_region_{modality_label}.csv")

    subjects_df.to_csv(subject_path)
    coverage_df.to_csv(coverage_path, index=False)
    group_df.to_csv(group_path)

    print(f"Saved:")
    print(f"  Subject-level: {subject_path}")
    print(f"  Coverage: {coverage_path}")
    print(f"  Group map: {group_path}")

    return subjects_df, group_df


# ─── MAIN ORCHESTRATOR ───

def main():

    print("Starting PET parcellation...")

    amyloid_subjects, amyloid_group = parcellate_modality(
        input_dir=AMYLOID_INPUT_DIR,
        modality_label="amyloid",
        atlas_img=atlas_img,
        atlas_data=atlas_data,
        all_labels=all_labels,
        all_labels_names=all_labels_names,
        id_to_region=id_to_region,
        output_dir=OUTPUT_DIR,
        compute_suvr_ratio=COMPUTE_AMYLOID_SUVR
    )

    tau_subjects, tau_group = parcellate_modality(
        input_dir=TAU_INPUT_DIR,
        modality_label="tau",
        atlas_img=atlas_img,
        atlas_data=atlas_data,
        all_labels=all_labels,
        all_labels_names=all_labels_names,
        id_to_region=id_to_region,
        output_dir=OUTPUT_DIR,
        compute_suvr_ratio=COMPUTE_TAU_SUVR
    )

    print("Files created:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {fname}")

if __name__ == "__main__":
    main()