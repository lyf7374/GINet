import os
def load_H_extra_re(base_path ):
#     base_path = 'Health_extra' 
    skull_paths_LBPA40 = []
    brain_mask_paths_LBPA40 = []

    folder  = "LBPA40"
    # , ""]:
    folder_path = os.path.join(base_path, folder)

    # Iterate over each patient folder in the subfolder
    for patient_folder in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient_folder)

        # Check if it's a directory
        if os.path.isdir(patient_path):
            skull_file = os.path.join(patient_path, "T1_re.nii.gz")
            brain_mask_file = os.path.join(patient_path, "mask_re.nii.gz")

            # Check if the skull and brain mask files exist
            if os.path.isfile(skull_file):
                skull_paths_LBPA40.append(skull_file)
            if os.path.isfile(brain_mask_file):
                brain_mask_paths_LBPA40.append(brain_mask_file)

    skull_paths_NFBS= []
    brain_mask_paths_NFBS = []
    folder  = "NFBS"

    folder_path = os.path.join(base_path, folder)

    # Iterate over each patient folder in the subfolder
    for patient_folder in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient_folder)

        # Check if it's a directory
        if os.path.isdir(patient_path):
            skull_file = os.path.join(patient_path, "T1_re.nii.gz")
            brain_mask_file = os.path.join(patient_path, "mask_re.nii.gz")

            # Check if the skull and brain mask files exist
            if os.path.isfile(skull_file):
                skull_paths_NFBS.append(skull_file)
            if os.path.isfile(brain_mask_file):
                brain_mask_paths_NFBS.append(brain_mask_file)

    print('Extra Dataset LBPA40--patients :{}   NFBS--patients {}'.format(len(skull_paths_LBPA40),len(skull_paths_NFBS)))

    return skull_paths_LBPA40, brain_mask_paths_LBPA40, skull_paths_NFBS, brain_mask_paths_NFBS

def load_H_extra(base_path ):
#     base_path = 'Health_extra' 
    skull_paths_LBPA40 = []
    brain_mask_paths_LBPA40 = []

    folder  = "LBPA40"
    # , ""]:
    folder_path = os.path.join(base_path, folder)

    # Iterate over each patient folder in the subfolder
    for patient_folder in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient_folder)

        # Check if it's a directory
        if os.path.isdir(patient_path):
            skull_file = os.path.join(patient_path, "T1.nii.gz")
            brain_mask_file = os.path.join(patient_path, "mask.nii.gz")

            # Check if the skull and brain mask files exist
            if os.path.isfile(skull_file):
                skull_paths_LBPA40.append(skull_file)
            if os.path.isfile(brain_mask_file):
                brain_mask_paths_LBPA40.append(brain_mask_file)

    skull_paths_NFBS= []
    brain_mask_paths_NFBS = []
    folder  = "NFBS"

    folder_path = os.path.join(base_path, folder)

    # Iterate over each patient folder in the subfolder
    for patient_folder in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient_folder)

        # Check if it's a directory
        if os.path.isdir(patient_path):
            skull_file = os.path.join(patient_path, "T1.nii.gz")
            brain_mask_file = os.path.join(patient_path, "mask.nii.gz")

            # Check if the skull and brain mask files exist
            if os.path.isfile(skull_file):
                skull_paths_NFBS.append(skull_file)
            if os.path.isfile(brain_mask_file):
                brain_mask_paths_NFBS.append(brain_mask_file)

    print('Extra Dataset LBPA40--patients :{}   NFBS--patients {}'.format(len(skull_paths_LBPA40),len(skull_paths_NFBS)))

    return skull_paths_LBPA40, brain_mask_paths_LBPA40, skull_paths_NFBS, brain_mask_paths_NFBS
