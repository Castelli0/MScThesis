import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

def dreamer_to_np(DREAMER_dir):
    dreamer_mat = scipy.io.loadmat(DREAMER_dir)
    dreamer_data = np.squeeze(dreamer_mat['DREAMER']['Data'][0, 0])
    
    subject_count = len(dreamer_data)
    trial_count = len(dreamer_data[0]['ECG'][0, 0]['stimuli'][0, 0])
    
    data = []
    # Loop over subjects
    for subject in range(subject_count):
        subject_data = []
        # Loop over trials/videos
        for trial in range(trial_count):
            subject_trial_data = []
            subject_trial_ECG = dreamer_data[subject]['ECG'][0, 0]['stimuli'][0, 0][trial, 0]
            subject_trial_EEG = dreamer_data[subject]['EEG'][0, 0]['stimuli'][0, 0][trial, 0]
            subject_trial_data.append(subject_trial_ECG)
            subject_trial_data.append(subject_trial_EEG)
            subject_data.append(subject_trial_data)

        data.append(subject_data)
        
    data = np.array(data, dtype=object)

    return data, subject_count, trial_count

def get_trial_eeg(data, trial_index=0):
    subjects, trials, sensors = data.shape

    # Extract the desired data for the given trial index
    selected_data = data[:, trial_index, 1]

    # Create an empty ndarray to hold the eeg_data
    eeg_data = np.empty((subjects, selected_data[0].shape[1], selected_data[0].shape[0]))

    # Iterate over subjects and assign the selected data
    for i in range(subjects):
        eeg_data[i] = selected_data[i].T

    return eeg_data

data, subject_count, trial_count = dreamer_to_np('D:/Downloads 2/Andre_Thesis/Archive/DREAMER.mat')

eeg = np.array(data[:,:,1], dtype=object)

#
#To access ECG or EEG from a specific subject, and specific video
#
"""
subject_nbr = 22    #range 0-22
video_nbr = 17      #range 0-17
ECG = dreamer_data[subject_nbr]['ECG'][0,0]['stimuli'][0,0][video_nbr,0]
EEG = dreamer_data[subject_nbr]['EEG'][0,0]['stimuli'][0,0][video_nbr,0]

plt.figure()
plt.plot(range(1000), ECG[:1000,0])
plt.plot(range(1000), ECG[:1000,1])

plt.figure()
for i in range(14):
    plt.plot(range(1000), EEG[:1000,i])
plt.show
"""


#
# Extracting data from DREAMER as dict
#
"""
data_dict = {}

# Loop over subjects
for subject in range(subject_count):
    subject_id = 'subject_{}'.format(subject)
    data_dict[subject_id] = {}
    
    # Loop over trials/videos
    for trial in range(trial_count):
        trial_id = 'trial_{}'.format(trial)
        data_dict[subject_id][trial_id] = {'ECG':dreamer_data[subject]['ECG'][0,0]['stimuli'][0,0][trial,0],
                                      'EEG':dreamer_data[subject]['EEG'][0,0]['stimuli'][0,0][trial,0]}
        
        
plt.figure()
plt.plot(range(1000), data_dict['subject_0']['trial_0']['ECG'][:1000,0])

plt.figure()
for i in range(14):
    plt.plot(range(1000), data_dict['subject_0']['trial_0']['EEG'][:1000,i])
plt.show   
"""


"""
# Plot first 1000 samples ECG - subject 0, video 0
plt.figure()
plt.plot(range(1000), data[0][0][0][:1000,0])
plt.plot(range(1000), data[0][0][0][:1000,1])
plt.show()

# Plot EEG - subject 0, video 0
plt.figure()
for i in range(14):
    plt.plot(range(25472), data[0][0][1][:,i])
plt.show()       
""" 