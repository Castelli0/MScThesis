import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat

from CorrCA import CorrCA, stats, get_rolling_ISC

# Set plotting style for plotting with MNE functions
plt.style.use('default')    #default
#matplotlib.use('Qt5Agg')    #Interactive pop-up

cohen_data = loadmat('C:/Users/andre/Documents/DTU/Thesis/ISC_Parra_CorrCA/code-and-data/EEG-Cohen-2017/sunday_at_roccos.mat')['X'].T
superbowl = loadmat('C:/Users/andre/Documents/DTU/Thesis/ISC_Parra_CorrCA/code-and-data/EEG-Dmochowski-2014/superbowlData.mat')['dataTrain'].T

# Set the sampling frequency
fs = 512

X = superbowl
W, ISC, Y, A = CorrCA(X, k=2)

# Biosemi Info
# Set the channel names using the BioSemi 64 channel montage
montage = 'biosemi64'
info = mne.create_info(ch_names=mne.channels.make_standard_montage(montage).ch_names,
                   sfreq=fs, ch_types='eeg')

# Set the channel locations using the BioSemi 64 channel montage
montage = mne.channels.make_standard_montage(montage)
info.set_montage(montage)

raw_W = mne.io.RawArray(W, info)
raw_W.load_data()

# MNE topomap based on data from a pandas dataframe emulating components
# Set plotting style to default for plotting with MNE functions
#plt.style.use('default')
map_info = raw_W.info    

# Make plot window
fig, axes = plt.subplots(figsize=(4*W.shape[1],4), ncols=W.shape[1])
for ax, c in zip(axes, range(W.shape[1])):
    ax.set_title("C{}, ISC: {:.6f}".format(c+1, ISC[c]))
    im, cn = mne.viz.plot_topomap(W[:,c],pos=map_info,
                                   axes=ax, show=False, cmap = "RdBu_r",
                                   outlines = "head")
    
# Add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
plt.suptitle('First three correlated components - Cohen Data')
plt.show()


#Plot EEG
raw = mne.io.RawArray(X[0], info)

# filters
raw_filt = raw.copy().filter(l_freq=0.5, h_freq=60).notch_filter(50)

# select bad channels
raw_filt.info['bads'] += ['Fp1', 'AF7', 'AF3', 'F1', 'F5', 'Fp2', 'Fpz', 'AF8', 'AF4', 'Fz', 'AFz'] #['POz'] 
# select visually
#raw.plot(block= False, scalings='auto')
#raw.plot_sensors(block=False, show_names=True)
#raw.plot_sensors(kind='3d', block=False, show_names=True)

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
# ICA
num_good_channels = len(raw_filt.info['ch_names']) - len(raw_filt.info['bads'])
ica = mne.preprocessing.ICA(n_components=num_good_channels, max_iter="auto", random_state=42)
ica.fit(raw_filt)
#ica.exclude = [1]
raw_clean = ica.apply(raw_filt.copy())

ica.plot_components()
ica.plot_properties(raw_filt, picks=1)
ica.plot_sources(raw_filt, show_scrollbars=False)
"""


k=10
# Calculate ISC over time a given trial
thr, ISC_null = stats(X, k=k, n_surrogates=10)
ISC_values, ISC_total, W, time_values, num_windows, step_size = get_rolling_ISC(X, k=k)

# Plotting
fig, axes = plt.subplots(nrows=2 ,ncols=k, figsize=(4*k,7))
# Plot components topomaps
for ax, c in zip(axes[0,:], range(k)):
    ax.set_title("CorrC{}, ISC: {:.6f}".format(c+1, ISC_total[c]))
    im, cn = mne.viz.plot_topomap(W[:,c],pos=map_info,
                                   axes=ax, show=False, cmap = "RdBu_r",
                                   outlines = "head")
# Add colorbar
fig.colorbar(im, ax=axes[0,:].ravel().tolist(), shrink=0.9)

# Plot ISC over time
plt.subplot(2,1,2)
plt.title('ISC values over time')
# Plot the ISC values for each component as line plots with respect to time
for component in range(k):
    plt.plot(time_values, ISC_values[:, component], label=f'CorrC{component + 1}')

# Add threshold
plt.hlines(thr, xmin=0, xmax=time_values[-1], linestyles='dashed', label=f'Threshold: {thr:.6f}')
# Set plot title, labels, and legend
plt.xlabel('Time (s)')
plt.ylabel('ISC')
plt.legend()
plt.suptitle('Correlated Component Analysis', y=0.99)
plt.show()