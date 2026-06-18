# ─── IMPORTS ───

import os
import subprocess
import zipfile
from pathlib import Path
import shutil
import sys
import pandas as pd


# ─── CONFIGURATION ───

DCM2NIIX_PATH = "dcm2niix"

BASE_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_dicom"

AMYLOID_ZIPS = [
    "Fully_Processed_Amyloid_PET_PET_1.zip",
    "Fully_Processed_Amyloid_PET_PET_2.zip",
]
TAU_ZIPS = [
    "Fully_Processed_Tau_PET_PET_1.zip",
    "Fully_Processed_Tau_PET_PET_2.zip",
]

AMYLOID_MANIFEST = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\filtered_manifests\amyloid_filtered_manifest.csv"
TAU_MANIFEST = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\filtered_manifests\tau_filtered_manifest.csv"

OUTPUT_DIR = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\amyloid_tau_pet_nifti"

AMYLOID_NIFTI = os.path.join(OUTPUT_DIR, "amyloid_pet_nifti")
TAU_NIFTI = os.path.join(OUTPUT_DIR, "tau_pet_nifti")

TEMP_DICOM_DIR = os.path.join(OUTPUT_DIR, "temp_pet_dicom")

for folder in [AMYLOID_NIFTI, TAU_NIFTI, TEMP_DICOM_DIR]:
    os.makedirs(folder, exist_ok=True)


# ─── VERIFY dcm2niix ACCESSIBILITY ───

def check_dcm2niix():
 
    try:
        result = subprocess.run(
            [DCM2NIIX_PATH, "--version"],
            capture_output=True, text=True, timeout=10
        )
        print(f"dcm2niix found: {result.stdout.strip()}")
        return True
    
    except (subprocess.SubprocessError, FileNotFoundError):
        print("ERROR: dcm2niix not found.")
        print("Install it with: conda install -c conda-forge dcm2niix")
        return False


# ─── DICOM TO NIFTI CONVERSION ───

def convert_dicom_to_nifti(dicom_dir, output_dir, output_filename_base):

    # STEP 1: DEFINE dcm2niix COMMAND

    cmd = [
        DCM2NIIX_PATH,
        "-z", "y",                      
        "-f", output_filename_base,     
        "-o", output_dir,             
        "-s", "y",                     
        "-m", "y",                      
        "-b", "n",                     
        dicom_dir,                   
    ]

    # STEP 2: RUN dcm2niix AND CHECK ITS PERFORMANCE

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120 
        )

        if result.returncode != 0:
            print(f"dcm2niix error: {result.stderr.strip()}")
            return None

        expected_path = os.path.join(output_dir, f"{output_filename_base}.nii.gz")
        if os.path.exists(expected_path):
            return expected_path

        created_files = [
            f for f in os.listdir(output_dir)
            if output_filename_base in f and
            (f.endswith(".nii.gz") or f.endswith(".nii"))
        ]
        if created_files:
            return os.path.join(output_dir, created_files[0])

        print(f"Conversion completed but no output file found")
        return None

    except subprocess.TimeoutExpired:
        print(f"Conversion timed out for {output_filename_base}")
        return None
    
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None


# ─── MAIN EXTRACTION AND CONVERSION FUNCTION ───

def extract_and_convert(zip_paths, target_ids, id_to_subject, nifti_output_dir, modality_label):

    # STEP 1: INITIALIZE TRACKING STRUCTURES

    found = {}
    not_found = set(target_ids)
    failed_conversion = set()

    # STEP 2: LOOP OVER ALL ZIP FILES

    for zip_path in zip_paths:
        if not os.path.exists(zip_path):
            print(f"WARNING: Zip not found: {zip_path}")
            continue

        print(f"Searching: {os.path.basename(zip_path)}")

        with zipfile.ZipFile(zip_path, 'r') as zf:

            all_entries = zf.namelist()

            # STEP 2.1: BUILD INDEX

            id_to_entries = {}
            for entry in all_entries:
                parts = Path(entry).parts
                for part in parts:
                    if part.startswith('I') and part[1:].isdigit():
                        img_id = part[1:] 
                        if img_id not in id_to_entries:
                            id_to_entries[img_id] = []
                        id_to_entries[img_id].append(entry)
                        break
            print(f"Image IDs indexed in this zip: {len(id_to_entries)}")

            ids_in_zip = set(id_to_entries.keys()) & not_found
            print(f"Target IDs found in this zip: {len(ids_in_zip)}")

            # STEP 2.2: PROCESS EACH FOUND IMAGE ID

            for img_id in sorted(ids_in_zip):
                subject_id = id_to_subject.get(img_id, f"unknown_{img_id}")
                entries = id_to_entries[img_id]

                dicom_entries = [
                    e for e in entries
                    if not e.endswith('/')   
                    and not os.path.basename(e).startswith('.')   
                ]

                if len(dicom_entries) == 0:
                    print(f"No files found for I{img_id}. Skip.")
                    not_found.discard(img_id)
                    failed_conversion.add(img_id)
                    continue

                print(f"Processing I{img_id} ({subject_id}): {len(dicom_entries)} files")

                # STEP 2.2.1: CHECK EACH IMAGE STATE

                img_state = f"{subject_id}_I{img_id}"
                nifti_path = os.path.join(nifti_output_dir, f"{img_state}.nii.gz")
                if os.path.exists(nifti_path):
                    print(f"Image already converted: {img_state}.nii.gz")
                    found[img_id] = nifti_path
                    not_found.discard(img_id)
                    continue
                
                # STEP 2.2.2: EXTRACT DICOM IMAGES TEMPORARILY

                temp_dir = os.path.join(TEMP_DICOM_DIR, f"I{img_id}")
                os.makedirs(temp_dir, exist_ok=True)

                print(f"Extracting {len(dicom_entries)} DICOM files...")
                for entry in dicom_entries:
                    filename = os.path.basename(entry)
                    temp_path = os.path.join(temp_dir, filename)

                    with zf.open(entry) as source, open(temp_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

                # STEP 2.2.3: CONVERT DICOM IMAGES TO NIfTI
                print(f"Converting to NIfTI...")
                img_result = convert_dicom_to_nifti(
                    temp_dir,
                    nifti_output_dir,
                    img_state
                )

                # STEP 2.2.4: CLEAN UP TEMPORARY DICOM IMAGES

                shutil.rmtree(temp_dir, ignore_errors=True)

                if img_result and os.path.exists(img_result):
                    print(f"NIfTI image saved: {os.path.basename(img_result)}")
                    found[img_id] = img_result
                    not_found.discard(img_id)
                else:
                    print(f"NIfTI conversion failed for image I{img_id}")
                    failed_conversion.add(img_id)
                    not_found.discard(img_id)

        if len(not_found) == 0:
            print(f"All {modality_label} DICOM images converted to NIfTI.")
            break

    return found, not_found, failed_conversion


# ─── MAIN ORCHESTRATOR ───

def main():

    if not check_dcm2niix():
        sys.exit(1)

    print("Loading manifests...")
    amyloid_df = pd.read_csv(AMYLOID_MANIFEST)
    tau_df = pd.read_csv(TAU_MANIFEST)

    amyloid_ids = set(amyloid_df["image_id"].astype(str).tolist())
    tau_ids = set(tau_df["image_id"].astype(str).tolist())

    amyloid_id_to_subject = dict(zip(
        amyloid_df["image_id"].astype(str),
        amyloid_df["subject_id"].astype(str)
    ))
    tau_id_to_subject = dict(zip(
        tau_df["image_id"].astype(str),
        tau_df["subject_id"].astype(str)
    ))

    print(f"Amyloid ID images to process: {len(amyloid_ids)}")
    print(f"Tau ID images to process: {len(tau_ids)}")

    amyloid_zip_paths = [os.path.join(BASE_DIR, z) for z in AMYLOID_ZIPS]
    tau_zip_paths = [os.path.join(BASE_DIR, z) for z in TAU_ZIPS]

    amyloid_found, amyloid_missing, amyloid_failed = extract_and_convert(
        amyloid_zip_paths, amyloid_ids, amyloid_id_to_subject,
        AMYLOID_NIFTI, "amyloid"
    )
    tau_found, tau_missing, tau_failed = extract_and_convert(
        tau_zip_paths, tau_ids, tau_id_to_subject,
        TAU_NIFTI, "tau"
    )

    # SUMMARY 

    print(f"Amyloid ({len(amyloid_ids)} images conversion requested):")
    print(f"  Successfully converted: {len(amyloid_found)}")
    print(f"  Conversion failed:      {len(amyloid_failed)}")
    print(f"  Not found in zips:      {len(amyloid_missing)}")

    print(f"Tau ({len(tau_ids)} images conversion requested):")
    print(f"  Successfully converted: {len(tau_found)}")
    print(f"  Conversion failed:      {len(tau_failed)}")
    print(f"  Not found in zips:      {len(tau_missing)}")


    # LOG SAVING 

    log_rows = []

    for img_id, path in amyloid_found.items():
        log_rows.append({
            "modality": "amyloid",
            "image_id": img_id,
            "subject_id": amyloid_id_to_subject.get(img_id, ""),
            "status": "converted",
            "nifti_path": path,
        })
    for img_id in amyloid_failed:
        log_rows.append({
            "modality": "amyloid",
            "image_id": img_id,
            "subject_id": amyloid_id_to_subject.get(img_id, ""),
            "status": "conversion_failed",
            "nifti_path": "",
        })
    for img_id in amyloid_missing:
        log_rows.append({
            "modality": "amyloid",
            "image_id": img_id,
            "subject_id": amyloid_id_to_subject.get(img_id, ""),
            "status": "not_found_in_zip",
            "nifti_path": "",
        })

    for img_id, path in tau_found.items():
        log_rows.append({
            "modality": "tau",
            "image_id": img_id,
            "subject_id": tau_id_to_subject.get(img_id, ""),
            "status": "converted",
            "nifti_path": path,
        })
    for img_id in tau_failed:
        log_rows.append({
            "modality": "tau",
            "image_id": img_id,
            "subject_id": tau_id_to_subject.get(img_id, ""),
            "status": "conversion_failed",
            "nifti_path": "",
        })
    for img_id in tau_missing:
        log_rows.append({
            "modality": "tau",
            "image_id": img_id,
            "subject_id": tau_id_to_subject.get(img_id, ""),
            "status": "not_found_in_zip",
            "nifti_path": "",
        })

    log_df = pd.DataFrame(log_rows)
    log_path = os.path.join(OUTPUT_DIR, "pet_extraction_conversion_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"Log saved: {log_path}")

    # CLEAN UP TEMPORARY DIRECTORY
    if os.path.exists(TEMP_DICOM_DIR):
        shutil.rmtree(TEMP_DICOM_DIR, ignore_errors=True)
        print("Temporary DICOM directory cleaned up.")

    print("Pet extraction and conversion done.")

if __name__ == "__main__":
    main()