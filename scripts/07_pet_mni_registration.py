# ─── IMPORTS ───

import os
import sys
import ants
import numpy as np  
import glob
import pandas as pd
import traceback
import argparse


# ─── CONFIGURATION ───

AMYLOID_INPUT = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_nifti\amyloid_pet_nifti"
TAU_INPUT = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_nifti\tau_pet_nifti"

AMYLOID_MNI = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_mni\amyloid_pet_mni"
TAU_MNI = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_mni\tau_pet_mni"

MNI_TEMPLATE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\atlas\mni_icbm152_t1_tal_nlin_asym_09c.nii"

TRANSFORM_TYPE = "Affine"


# ─── PET IMAGE TO MNI SPACE REGISTRATION ───

# STEP 1: VERIFY INPUTS

def verify_setup(input_dir, output_dir, mni_path):

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    if not os.path.exists(mni_path):
        print(f"ERROR: MNI template not found: {mni_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)


# STEP 2: CHECK IF PET IMAGES ALREADY IN MNI SPACE 

def check_if_already_mni(pet_img, tolerance_mm=10.0):

    origin  = pet_img.origin
    shape   = pet_img.shape
    spacing = pet_img.spacing

    mni_shapes = {
        "1mm": (182, 218, 182),
        "2mm": (91, 109, 91),
    }

    origin_dist = (origin[0]**2 + origin[1]**2 + origin[2]**2) ** 0.5

    shape_match = any(
        abs(shape[0] - s[0]) < 5 and
        abs(shape[1] - s[1]) < 5 and
        abs(shape[2] - s[2]) < 5
        for s in mni_shapes.values()
    )

    if origin_dist and shape_match < tolerance_mm:
        return True, "Image origin and dimensions suggest MNI space."
    return False, f"Origin dist {origin_dist:.1f}mm from (0,0,0) and shape {shape}."


# STEP 3: REGISTER PET IMAGE 

def register_pet_to_mni(pet_path, mni_img, output_path, transform_type):

    pet = ants.image_read(pet_path)

    # STEP 3.1: CHECK MNI SPACE 

    already_mni, reason = check_if_already_mni(pet)
    if already_mni:
        print(f"Note: {reason} — affine registration may be minimal.")

    # STEP 3.2: RUN REGISTRATION 

    reg = ants.registration(
        fixed=mni_img,
        moving=pet,
        type_of_transform=transform_type,
        verbose=False,
    )

    pet_mni = reg["warpedmovout"]
    ants.image_write(pet_mni, output_path)

    if not os.path.exists(output_path):
        return False

    # STEP 3.3: VALIDATE REGISTRATION

    reg = ants.image_read(output_path)
    shape_ok = reg.shape == mni_img.shape
    spacing_ok = all(abs(r - m) < 0.01 for r, m in zip(reg.spacing, mni_img.spacing))
    det = np.linalg.det(reg.direction)
    orient_ok = abs(abs(det) - 1.0) < 0.01
    origin_dev = np.linalg.norm(np.array(reg.origin) - np.array(mni_img.origin))
    origin_ok = origin_dev < 15.0

    print(f"Shape: {reg.shape} vs template {mni_img.shape} — {'✓' if shape_ok else '✗'}")
    print(f"Spacing: {tuple(round(s,2) for s in reg.spacing)} — {'✓' if spacing_ok else '✗'}")
    print(f"Orientation: det={det:.3f} — {'✓' if orient_ok else '✗'}")
    print(f"Origin: {tuple(round(o,1) for o in reg.origin)}, deviation {origin_dev:.1f}mm — {'✓' if origin_ok else '✗'}")
    passed = all([shape_ok, spacing_ok, orient_ok, origin_ok])
    print(f"{'PASSED ✓' if passed else 'FAILED ✗'}")

    print(f"Saved MNI registered pet image: {os.path.basename(output_path)}")
    return True

# STEP 4: PROCESS ALL IMAGES IN DIRECTORY

def process_directory(input_dir, output_dir, mni_img, transform_type, modality_label):

    img_nifti = (
        glob.glob(os.path.join(input_dir, "*.nii.gz")) +
        glob.glob(os.path.join(input_dir, "*.nii"))
    )
    
    img_nifti = sorted(set(img_nifti))

    if len(img_nifti) == 0:
        print(f"No NIfTI files found. Check that {input_dir} contains .nii or .nii.gz files.")
        return pd.DataFrame()

    print(f"Found {len(img_nifti)} NIfTI files in {input_dir}")
    
    results = []

    for i, pet_path in enumerate(img_nifti):
        filename = os.path.basename(pet_path)
        base_name  = filename.replace(".nii.gz", "").replace(".nii", "")
        out_name   = f"MNI_{base_name}.nii.gz"
        out_path   = os.path.join(output_dir, out_name)
        print(f"[{i+1}/{len(img_nifti)}] {filename}")

        if os.path.exists(out_path):
            print(f"Already MNI registered pet image. Skip.")
            results.append({
                "modality": modality_label,
                "input_file": filename,
                "output_file": out_name,
                "status": "skipped_already_done",
            })
            continue

        try:
            success = register_pet_to_mni(
                pet_path, mni_img, out_path, transform_type
            )
            status = "registered" if success else "failed_no_output"

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            status = f"failed_exception"
            success = False

        results.append({
            "modality": modality_label,
            "input_file": filename,
            "output_file": out_name if success else "",
            "status": status,
        })

    # SUMMARY

    df = pd.DataFrame(results)
    n_done = (df["status"] == "registered").sum()
    n_skipped = (df["status"] == "skipped_already_done").sum()
    n_failed = df["status"].str.startswith("failed").sum()

    print(f"{modality_label.upper()} pet images MNI registration:")
    print(f"  Registered: {n_done}")
    print(f"  Already done: {n_skipped}")
    print(f"  Failed: {n_failed}")

    return df


# ─── MAIN ORCHESTRATOR ───

def main():

    # COMMAND LINE MODALITY SELECTION

    parser = argparse.ArgumentParser(
        description="Registration of PET images to MNI152 space"
    )
    parser.add_argument(
        "--modality",
        choices=["amyloid", "tau", "both"],
        default="both",
        help="which modality to process (default: both)"
    )
    args = parser.parse_args()

    # MNI152 TEMPLATE LOADING
    print("Loading MNI152 template...")
    if not os.path.exists(MNI_TEMPLATE):
        print(f"ERROR: MNI template not found: {MNI_TEMPLATE}")
        sys.exit(1)
    mni_img = ants.image_read(MNI_TEMPLATE)
    print(f"  Template shape:   {mni_img.shape}")
    print(f"  Template spacing: {mni_img.spacing}")

    all_results = []

    # AMYLOID PROCESSING 

    if args.modality in ("amyloid", "both"):
        verify_setup(AMYLOID_INPUT, AMYLOID_MNI, MNI_TEMPLATE)
        df = process_directory(
            AMYLOID_INPUT, AMYLOID_MNI, mni_img,
            TRANSFORM_TYPE, "amyloid"
        )
        all_results.append(df)

    # TAU PROCESSING 

    if args.modality in ("tau", "both"):
        verify_setup(TAU_INPUT, TAU_MNI, MNI_TEMPLATE)
        df = process_directory(
            TAU_INPUT, TAU_MNI, mni_img,
            TRANSFORM_TYPE, "tau"
        )
        all_results.append(df)

    # LOG SAVING 

    if all_results:
        log_df = pd.concat(all_results, ignore_index=True)
        log_path = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_mni\registration_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"MNI registration log saved: {log_path}")

if __name__ == "__main__":
    main()