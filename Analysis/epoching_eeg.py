import numpy as np
import pandas as pd
import os

def get_eeg_data(root_path):
    # Initialize an empty DataFrame to store concatenated data
    concatenated_data = []
    # Walk through all subdirectories in the root path
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file_name in filenames:
            # Check if the file ends with _ultracortex.csv
            if file_name.endswith("_ultracortex.csv"):
                file_path = os.path.join(dirpath, file_name)
                # Read the CSV file into a DataFrame
                data = pd.read_csv(file_path, index_col=0)
                data = data.values.T
                # Concatenate the data to the existing DataFrame
                concatenated_data.append(data)
    return concatenated_data

def get_emotibit_data(root_path):
    # Initialize an empty DataFrame to store concatenated data
    concatenated_data = []
    # Walk through all subdirectories in the root path
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file_name in filenames:
            # Check if the file ends with _ultracortex.csv
            if file_name.endswith("_emotibit.csv"):
                file_path = os.path.join(dirpath, file_name)
                # Read the CSV file into a DataFrame
                data = pd.read_csv(file_path, index_col=0)
                data = data.values.T
                # Concatenate the data to the existing DataFrame
                concatenated_data.append(data)
    return concatenated_data

def get_aux_data(root_path):
    # Initialize an empty DataFrame to store concatenated data
    concatenated_data = []
    # Walk through all subdirectories in the root path
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file_name in filenames:
            # Check if the file ends with _aux.csv
            if file_name.endswith("_aux.csv"):
                file_path = os.path.join(dirpath, file_name)
                # Read the CSV file into a DataFrame
                data = pd.read_csv(file_path, index_col=0)
                data = data.values.T
                # Concatenate the data to the existing DataFrame
                concatenated_data.append(data)
    return concatenated_data

def get_eeg_boundaries(aux_data):
    epoch_boundaries = []
    for subj_idx in range(len(aux_data)):
        subject_boundaries = []
        for sample_idx in range(len(aux_data[subj_idx][-1])):
            if aux_data[subj_idx][-1, sample_idx] == -1:
                if aux_data[subj_idx][-2, sample_idx] == 1:
                    start_sample = aux_data[subj_idx][1,sample_idx] - 3125
                    subject_boundaries.append([int(start_sample), int(start_sample + 3125)])
                #epoch_start = aux_data[subj_idx][1, sample_idx]     
                elif aux_data[subj_idx][-2, sample_idx] == 9:
                    start_sample = aux_data[subj_idx][1,sample_idx]
                    subject_boundaries.append([int(start_sample), int(start_sample + 3125)])
                    subject_boundaries.append([int(start_sample + 3126), int(start_sample + 3126 + 3125)])
                else:
                    start_sample = aux_data[subj_idx][1,sample_idx]
                    subject_boundaries.append([int(start_sample), int(start_sample + 3125)])
        subject_boundaries.insert(0, [0, 3125])
        subject_boundaries.insert(1, [3126, 3126 + 3125])
        epoch_boundaries.append(subject_boundaries)
    return np.array(epoch_boundaries)

def get_emotibit_boundaries(aux_data):
    epoch_boundaries = []
    for subj_idx in range(len(aux_data)):
        subject_boundaries = []
        for sample_idx in range(len(aux_data[subj_idx][-1])):
            if aux_data[subj_idx][-1, sample_idx] == -1:
                if not np.isnan(aux_data[subj_idx][2, sample_idx]):
                    if aux_data[subj_idx][-2, sample_idx] == 1:
                        start_sample = aux_data[subj_idx][2,sample_idx] - 625
                        subject_boundaries.append([int(start_sample), int(start_sample + 625)])
                    #epoch_start = aux_data[subj_idx][1, sample_idx]     
                    elif aux_data[subj_idx][-2, sample_idx] == 9:
                        start_sample = aux_data[subj_idx][2,sample_idx]
                        subject_boundaries.append([int(start_sample), int(start_sample + 625)])
                        subject_boundaries.append([int(start_sample + 626), int(start_sample + 626 + 625)])
                    else:
                        start_sample = aux_data[subj_idx][2,sample_idx]
                        subject_boundaries.append([int(start_sample), int(start_sample + 625)])
        subject_boundaries.insert(0, [0, 625])
        subject_boundaries.insert(1, [626, 626 + 625])
        epoch_boundaries.append(subject_boundaries)
    return np.array(epoch_boundaries)

def segment_eeg_data(eeg_data, epoch_boundaries):
    segmented_epochs = []
    for subj_idx in range(len(eeg_data)):
        subject_epochs = []
        for epoch_idx in range(epoch_boundaries.shape[1]):
            start_sample = epoch_boundaries[subj_idx, epoch_idx, 0]
            end_sample = epoch_boundaries[subj_idx, epoch_idx, 1]
            epoch_data = eeg_data[subj_idx][:, start_sample:end_sample]
            subject_epochs.append(epoch_data)
        segmented_epochs.append(subject_epochs)
    return np.array(segmented_epochs)

def segment_emotibit_data(emotibit_data, epoch_boundaries):
    segmented_epochs = []
    for subj_idx in range(len(emotibit_data)):
        subject_epochs = []
        for epoch_idx in range(len(epoch_boundaries[subj_idx])):
            start_sample = epoch_boundaries[subj_idx][epoch_idx][0]
            end_sample = epoch_boundaries[subj_idx][epoch_idx][1]
            epoch_data = emotibit_data[subj_idx][:, start_sample:end_sample]
            subject_epochs.append(epoch_data)
        segmented_epochs.append(subject_epochs)
    return np.array(segmented_epochs)

"""
eeg = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg.npz')
eeg = [eeg[subject] for subject in eeg]
eeg_filt = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg_filt.npz')
eeg_filt = [eeg_filt[subject] for subject in eeg_filt]
eeg_clean = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg_clean.npz')
eeg_clean = [eeg_clean[subject] for subject in eeg_clean]

path = 'D:/Downloads 2/Andre_Thesis/Data/WOLF_dataset/'
#eeg = get_eeg_data(path)
aux = get_aux_data(path)
boundaries = get_eeg_boundaries(aux)
epochs = segment_eeg_data(eeg, boundaries)
epochs_filt = segment_eeg_data(eeg_filt, boundaries)
epochs_clean = segment_eeg_data(eeg_clean, boundaries)

np.save('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs.npy', epochs)
np.save('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs_filt.npy', epochs_filt)
np.save('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs_clean.npy', epochs_clean)


path = 'D:/Downloads 2/Andre_Thesis/Data/WOLF_dataset/'
emotibit = get_emotibit_data(path)
aux = get_aux_data(path)
boundaries = get_emotibit_boundaries(aux)
epochs = segment_emotibit_data(emotibit, boundaries)

np.savez('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/emotibit.npz', *emotibit)
np.save('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/emotibit_epochs.npy', epochs)
"""