import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from dreamer_read import eeg #, subject_count, trial_count

# Set plotting style for plotting with MNE functions
plt.style.use('default')    #default
#matplotlib.use('Qt5Agg')    #Interactive pop-up

# Set MNE info for DREAMER dataset
sampling_frequency = 128
ch_names = ['AF3', 'AF4', 'F7', 'F3', 'F4',
          'F8', 'FC5', 'FC6', 'T7', 'T8',
          'P7', 'P8', 'O1', 'O2']
mont = mne.channels.make_standard_montage("standard_1020")
#mont.plot()

# Create an MNE Info object
info = mne.create_info(ch_names=ch_names, ch_types= 'eeg',
                       sfreq=sampling_frequency)
info.set_montage(mont, on_missing='ignore')


post_processed_eeg = np.empty_like(eeg, dtype=object)

for subject in range(eeg.shape[0]):
    for trial in range(eeg.shape[1]): 
        # Create an MNE RawArray object
        raw = mne.io.RawArray(eeg[subject,trial].T, info)
        raw.load_data()
        
        # filters
        raw_filt = raw.copy().filter(l_freq=0.5, h_freq=None).notch_filter(50)
        
        """
        # PSD plot
        fig, ax = plt.subplots(2)
        raw.plot_psd(ax=ax[0], show=False)
        raw_filt.plot_psd(ax=ax[1], show=False)
        ax[0].set_title('PSD before filtering')
        ax[1].set_title('PSD after filtering')
        ax[1].set_xlabel('Frequency (Hz)')
        fig.set_tight_layout(True)
        plt.show(block=False)
        """
        
        # select bad channels
        """
        # select by code
        raw.info['bads'] += ['P8']
        """
        # select visually
        raw_filt.plot(block= False, scalings='auto')
        #raw_filt.plot_sensors(block=False, show_names=True)
        #raw_filt.plot_sensors(kind='3d', block=False, show_names=True)
        
        
        # ICA
        ica = mne.preprocessing.ICA(n_components=len(eeg[subject,trial].T),
                                    max_iter="auto", random_state=42)
        ica.fit(raw_filt)
        ica.exclude = [1]
        raw_clean = ica.apply(raw_filt.copy())
        
        ica.plot_components()
        ica.plot_properties(raw_filt, picks=1)
        ica.plot_sources(raw_filt, show_scrollbars=False)
        
        post_processed_eeg[subject, trial] = raw.get_data()