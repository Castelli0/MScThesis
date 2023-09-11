import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
import os
#from dreamer_read import eeg #, subject_count, trial_count
from epoching_eeg import get_aux_data, get_eeg_data, get_eeg_boundaries, segment_eeg_data
# Set plotting style for plotting with MNE functions
#plt.style.use('default')    #default
matplotlib.use('Qt5Agg')    #Interactive pop-up

# Set MNE info for WOLF dataset
sampling_frequency = 125
ch_names = ['Fp1','Fp2','C3','C4','P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3',
      'F4', 'T7', 'T8','P3','P4']
mont = mne.channels.make_standard_montage("standard_1020")
#mont.plot()

# Create an MNE Info object
info = mne.create_info(ch_names=ch_names, ch_types= 'eeg',
                       sfreq=sampling_frequency)
info.set_montage(mont, on_missing='ignore')


# Applying filters to the full length recordings
path = 'D:/Downloads 2/Andre_Thesis/Data/WOLF_dataset/'

eeg = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg.npz')
eeg = [eeg[subject] for subject in eeg]
eeg.pop(1) #elimate subject 1

for i, data in enumerate(eeg):
    # Filter EEG
    raw = mne.io.RawArray(data, info)
    raw.load_data()
    raw_filt = raw.copy().filter(l_freq=0.5, h_freq=40).notch_filter(25).notch_filter(50)
    eeg[i] = raw_filt.get_data()

eeg = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg_filt05_40_notch25_50.npz')
eeg = [eeg[subject] for subject in eeg]



# Epoching after filters, Selecting bad channels
aux = get_aux_data(path)
boundaries = get_eeg_boundaries(aux)
epochs = segment_eeg_data(eeg, boundaries)
epochs = epochs[:,1:,:,:]

#eeg_concat = epochs.transpose(0, 2, 1, 3).reshape(epochs.shape[0], epochs.shape[2], -1)

#eeg_filt = []
#eeg_clean = []

epochs_nobads = np.empty((11, 8), dtype=object)

for subject in range(epochs.shape[0]):
    for trial in range(epochs.shape[1]):
        # Create an MNE RawArray object
        raw = mne.io.RawArray(epochs[subject,trial,:,:], info)#epochs[subject, trial,:,:], info)
        # select bad channels
        raw.plot_psd()
        raw.plot(block= True, scalings='auto')
        epochs_nobads[subject,trial] = raw
#eeg_filt.append(raw_filt.get_data())
#eeg_clean.append(raw_clean.get_data())




# ICA 
epochs_nobads = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs_nobads.npz', allow_pickle=True)
epochs_nobads = [epochs_nobads[subject] for subject in epochs_nobads]
epochs_nobads.pop(9)
epochs_clean = np.empty((10, 8), dtype=object)
#for subject in range(len(epochs_nobads)):
#    for trial in range(len(epochs_nobads[subject])):
    
subject = 1
trial = 0


print(f'subject:{subject}, trial:{trial}')
raw = epochs_nobads[subject][trial]
ica = mne.preprocessing.ICA(n_components=16-len(raw.info['bads']), max_iter="auto", random_state=42)
ica.fit(raw)
ica.plot_components()
ica.plot_sources(raw, block=False, show_scrollbars=True)



ica.plot_properties(raw, picks=(2,3))






raw_clean = ica.apply(raw.copy())
raw_clean.interpolate_bads(reset_bads=True)
raw_clean.plot(scalings='auto')




epochs_clean[subject, trial] = raw_clean
trial +=1





subject +=1        


np.save('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs_ica.npy', epochs_clean)
"""
def extract_ultracortex_data(root_path):
    subject_folders = [folder for folder in os.listdir(root_path) if folder.startswith("Subject_")]
    subject_folders.sort()
    ultracortex_data = []
    
    for subject_folder in subject_folders:
        subject_path = os.path.join(root_path, subject_folder)
        ultracortex_files = [file for file in os.listdir(subject_path) if file.endswith("_ultracortex.npy")]
        
        subject_data = []
        for file in ultracortex_files:
            file_path = os.path.join(subject_path, file)
            data = np.load(file_path)
            subject_data.append(data)
        
        if subject_data:
            combined_data = np.stack(subject_data)  # Combine data from different files into one array
            ultracortex_data.append(combined_data)
    
    if ultracortex_data:
        final_array = np.stack(ultracortex_data)  # Combine data from different subjects into one array
        return final_array

np.savez('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg.npz', *eeg)
np.savez('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg_filt.npz', *eeg_filt)
np.savez('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg_clean.npz', *eeg_clean)
loaded_data = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/eeg_filt.npz')
"""