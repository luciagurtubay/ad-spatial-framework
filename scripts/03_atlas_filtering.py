# ─── IMPORTS ───

import nibabel as nib
import numpy as np


# ─── CONFIGURATION ───

DKT_ATLAS_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\atlas\DKT_MNI152_atlas.nii"
FILTERED_DKT_ATLAS_FILE = r"C:\Users\lucia\AppData\Roaming\SPB_Data\genetics\inputs\atlas\DKT_MNI152_atlas_filtered.nii"

TARGET_LABELS = [
    8, 10, 11, 12, 13, 16, 17, 18, 26, 28,
    47, 49, 50, 51, 52, 53, 54, 58, 60,

    1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018,
    1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026,
    1027, 1028, 1029, 1030, 1031, 1034, 1035,

    2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010,
    2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026,
    2027, 2028, 2029, 2030, 2031, 2034, 2035
]


# ─── ATLAS LOADING ───

atlas_image = nib.load(DKT_ATLAS_FILE)
atlas_data = atlas_image.get_fdata().astype(int)


# ─── ATLAS FILTERING ─── 

filtered_atlas = np.where(np.isin(atlas_data, TARGET_LABELS), atlas_data, 0) 
nib.save(nib.Nifti1Image(filtered_atlas, atlas_image.affine, atlas_image.header), FILTERED_DKT_ATLAS_FILE)

# ─── VALIDATION ───

filtered_atlas_image = nib.load(FILTERED_DKT_ATLAS_FILE)
filtered_atlas_data = filtered_atlas_image.get_fdata().astype(int)

unique_labels = set(np.unique(filtered_atlas_data)) - {0}

expected = set(TARGET_LABELS)
unexpected = unique_labels - expected    
missing = expected - unique_labels    

print(f"Expected labels in the DKT atlas ({len(expected)}): {sorted(expected)}")
print(f"Found labels in the DKT atlas ({len(unique_labels)}): {sorted(int(x) for x in unique_labels)}")

if not missing and not unexpected:
    print("Atlas filtering validated: all and only the target labels are present in the DKT atlas.")
else:
    if missing:
        print(f"Missing labels ({len(missing)}): {sorted(missing)}")
    if unexpected:
        print(f"Unexpected labels ({len(unexpected)}): {sorted(unexpected)}")