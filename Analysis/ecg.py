import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import detrend, welch, windows
from obspy.signal.util import next_pow_2
import os
import biosppy.signals.ecg as ecg

from dreamer_read import data, subject_count, trial_count

#
# ECG processing
#

def get_rr_interpol(data, ecg_fs=256, interpol_fs=10):
    subject_count, trial_count = len(data), len(data[0])
    steps = 1 / interpol_fs
    rr_interpol = []

    for subject in range(subject_count):
        subject_rr = []
        for trial in range(trial_count):
            subject_trial_rr = []
            
            # Process the ECG data to detect R-peaks
            ecg_pre = ecg.ecg(signal=data[subject][trial][0][:,0], sampling_rate=ecg_fs, show=False) # , show=True, interactive=True)
    
            # Retrieve the R-peak indices
            rpeaks = ecg_pre['rpeaks']
    
            # Calculate R-R intervals
            rr_intervals = []
            for i in range(1, len(rpeaks)):
                rr_interval = rpeaks[i] - rpeaks[i - 1]
                rr_intervals.append(rr_interval)
                
            for item in rr_intervals:
                subject_trial_rr.append(item/ecg_fs)
            
            # Interpolate
            timestamp = np.cumsum(subject_trial_rr)
            f = interp1d(timestamp, subject_trial_rr , 'linear')
            # Now we can sample from interpolation function
            timeindex_inter = np.arange(np.min(timestamp), np.max(timestamp), steps)
            subject_trial_rr = f(timeindex_inter)
                
            
            subject_rr.append(subject_trial_rr)
        rr_interpol.append(subject_rr)
    return rr_interpol

ecg_fs=256
interpol_fs=10
rr_interpol = get_rr_interpol(data, ecg_fs=ecg_fs, interpol_fs=interpol_fs )

# Detrend time-series (to remove slow drifts)
def plot_hrv_psd(rr_interpol, subject=0, trial=0, interpol_fs=10):
    rr_interpol_detrend = detrend(rr_interpol[subject][trial])
    
    # Plotting the power spectrum
    nfft = next_pow_2(len(rr_interpol_detrend))
    window = windows.hamming(len(rr_interpol_detrend)//4)
    freqs, PSD = welch(rr_interpol_detrend, fs=interpol_fs, window=window, nfft=nfft, scaling='density', return_onesided=True, detrend=False)

    # Calculating the power spectral density for high frequencies
    cond_hf = (freqs > 0.15) & (freqs < 0.4)
    hf = np.trapz(PSD[cond_hf],freqs[cond_hf])
    # Calculating the power spectral density for low frequencies
    cond_lf = (freqs > 0.04) & (freqs < 0.15)
    lf = np.trapz(PSD[cond_lf], freqs[cond_lf])
    lf_hf_ratio = lf/hf
    
    plt.figure(figsize=(15, 7))
    plt.plot(freqs, PSD)
    plt.xlim(0.04,0.4)
    #plt.ylim(0, 0.15)
    plt.vlines(0.15, ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashed', colors='green')
    plt.text(0.11, plt.ylim()[1]*0.96, 'lf: {:.3e}'.format(lf))
    plt.text(0.18, plt.ylim()[1]*0.96, 'hf: {:.3e}'.format(hf))
    plt.text(0.155, plt.ylim()[1]*0.91, 'lf/hf: {:.3f}'.format(lf_hf_ratio))
    plt.xlabel('Frequency')
    plt.ylabel('Power spectrum')
    plt.title("FFT Spectrum (Welch's periodogram) - HR - subject {} trial {}".format(subject+1, trial+1))
    plt.show()
    
    return PSD, freqs
plot_hrv_psd(rr_interpol)
plot_hrv_psd(rr_interpol, subject=1)

#
# Calculate correlation matrix
#
for trial in range(trial_count): 
    ecg_corr_matrix = np.zeros((subject_count, subject_count))
    for i in range(subject_count):
        ecg_i = rr_interpol[i][trial]
        for j in range(i, subject_count):
            ecg_j = rr_interpol[j][trial]
            
            # Determine the length of the smallest signal
            min_length = min(len(ecg_i), len(ecg_j))
            
            ecg_corr_matrix[i, j] = pearsonr(ecg_i[:min_length], ecg_j[:min_length])[0]
            ecg_corr_matrix[j, i] = ecg_corr_matrix[i, j]
            
    ecg_isc = []
    
    # Create a list of the correlations for each subject
    for i in range(subject_count):
        # Exclude the diagonal values
        ecg_isc.append(np.delete(ecg_corr_matrix[i], i))
    
    ecg_isc_mean = np.mean(ecg_isc)
    # Plot ECG correlation
    plt.figure(figsize=(18,7))
    plt.subplot(1,2,1)
    plt.suptitle('Inter-Subject Correlation - HR - DREAMER trial {}'.format(trial+1))
    plt.imshow(ecg_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlabel('Subject')
    plt.ylabel('Subject')
    plt.colorbar()
    plt.title('Correlation Matrix')
    
    plt.subplot(1,2,2)
    plt.title('Average Pearsons correlation')
    plt.boxplot(ecg_isc)
    #plt.text(-0.5, ecg_isc_mean+ 0.01, f'Average = {ecg_isc_mean:.2f}', color='green', fontsize = 15)
    #plt.hlines(ecg_isc_mean, 0, subject_count, linestyles='dashed', color='green')
    plt.xticks(range(1,1+subject_count),['Subject {}'.format(i+1) for i in range(subject_count)], rotation=60)
    plt.ylabel('Pearsons correlation coefficient')
    plt.show()
    
    