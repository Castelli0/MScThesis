import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import biosppy.signals.ppg as ppg
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values


file = 'D:/Downloads 2/Andre_Thesis/Data/ppg-ecg_test/test1/test1_emotibit.csv'

def extract_rr_from_files(folder_path):
    result_data = []
    ppg_data = []
    def process_folder(directory):
        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)
            if os.path.isdir(entry_path):
                process_folder(entry_path)
            elif entry.endswith('_IBI.csv'):
                df = pd.read_csv(entry_path, delimiter=';', skiprows=2)
                if 'Artifact corrected RR' in df.columns:
                    
                    rr_column = df['Artifact corrected RR'].tolist()
                    result_data.append(rr_column)
            elif entry.endswith('_emotibit.csv'):
                df = pd.read_csv(entry_path)
                ppg_columns = df[['ppg_1','ppg_2','ppg_3', 'unix', 'marker']].values
                ppg_data.append(ppg_columns)
    if os.path.isdir(folder_path):
        process_folder(folder_path)
    else:
        print("Invalid folder path. Please provide a valid folder path.")

    return result_data, ppg_data

def plot_ppg_rr(ppg_trial):
    rr = []#np.empty((len(onsets)-1, 3))
    rr_timestamps = []#np.empty_like(rr)
    
    for channel in range(len(ppg_trial[0])):
        ts, filtered, onsets, HR_ts, HR = ppg.ppg(ppg_trial[:,channel], sampling_rate=25,
                                                  show=False)
        rr_ = np.diff(ts[onsets])
        
        rr_timestamps.append(np.cumsum(rr_))
        rr_ = rr_*1000
        rr.append(rr_)
        
        # This remove outliers from signal
        rr_no_oulier = remove_outliers(rr_intervals=rr_,  
                                                        low_rri=300, high_rri=1500)
        # This replace outliers nan values with linear interpolation
        rr_interpol_oulier = interpolate_nan_values(rr_intervals=rr_no_oulier,
                                                           interpolation_method="linear")
        
    
        plt.figure(figsize=(16, 14))
        plt.suptitle('ppg_{} Preprocess'.format(channel+1), y=0.91)
        
        plt.subplot(2,1,1)
        plt.title('Onset detection')
        plt.plot(ts, filtered, label='filtered')
        plt.vlines(ts[onsets], ymin=min(filtered), ymax=max(filtered),
                   colors='r', alpha= 0.4, label='Onsets')
        plt.legend()
        
        #plt.subplot(4,1,2)
        #plt.title('Heart Rate')
        #plt.plot(HR_ts, HR, label='HR')
        #plt.ylabel('HR (bpm)')
        
        plt.subplot(2,1,2)
        plt.title('R-R Intervals')
        plt.plot(np.cumsum(rr_)/1000, rr_)
        plt.ylabel('r-r (ms)')
        plt.xlabel('Time (s)')
        plt.fill_between(np.cumsum(rr_)/1000, 1500, max(rr_), color='gray', alpha=0.5)
        plt.fill_between(np.cumsum(rr_)/1000, 0, 300, color='gray', alpha=0.5)
        """
        plt.subplot(4,1,3)
        plt.title('R-R without outliers')
        plt.ylabel('r-r (ms)')
        plt.xlabel('Time (s)')
        plt.plot(np.cumsum(rr_)/1000, rr_no_oulier)
     
        plt.subplot(4,1,4)
        plt.title('R-R interpolated')
        plt.ylabel('r-r (ms)')
        plt.xlabel('Time (s)')
        plt.plot(np.cumsum(rr_)/1000, rr_interpol_oulier)
        """
        plt.ylim(0)
        
    return rr, rr_timestamps 


def find_delay(signal1, signal2):
    # Calculate the cross-correlation between the two signals
    cross_corr = np.correlate(signal1, signal2, mode='full')
    
    # Determine the delay (shift) that gives the maximum correlation
    delay = np.argmax(cross_corr) - (len(signal2) - 1)

    return delay, cross_corr


if __name__ == "__main__":
    folder_path = 'D:/Downloads 2/Andre_Thesis/Data/ppg-ecg_test/'
    ecg_rr, ppg_data = extract_rr_from_files(folder_path)

    ppg_test1 = ppg_data[-1][:,:3]
    rr, rr_timestamps = plot_ppg_rr(ppg_test1)

    # This remove outliers from signal
    rr_no_oulier = remove_outliers(rr_intervals=rr[2],  
                                                    low_rri=300, high_rri=1500)
    # This replace outliers nan values with linear interpolation
    rr_interpol_oulier = interpolate_nan_values(rr_intervals=rr_no_oulier,
                                                       interpolation_method="linear")
    
    outlier_percentage = np.isnan(rr_no_oulier).sum()*100/len(rr_no_oulier)
    
    ecg_rr_timestamps = (np.cumsum(ecg_rr[-1])/1000)
    ecg_rr_timestamps = ecg_rr_timestamps - ecg_rr_timestamps[0]
    
    closest_index = np.argmin(np.abs(np.array(ecg_rr_timestamps) - (ecg_rr_timestamps[-1] - rr_timestamps[2][-1])))
    ecg_rr_timestamps = ecg_rr_timestamps[closest_index:] - ecg_rr_timestamps[closest_index]
    
    
    plt.figure(figsize=(15, 22))
    plt.suptitle('Emotibit ppg_3 - Firstbeaat Comparison', y=0.90)
    plt.subplot(411)
    plt.ylabel('r-r (ms)')
    plt.xlabel('Time (s)')
    plt.title('R-R from ppg_3')
    plt.plot(rr_timestamps[2], rr[2])
    plt.fill_between(rr_timestamps[2], 1500, 5000, color='gray', alpha=0.5)
    plt.fill_between(rr_timestamps[2], 0, 300, color='gray', alpha=0.5)
    #plt.hlines(1500, 0,600, 'r', linestyles= 'dashed') #sup limit for outlier detection
    #plt.hlines(300, 0,600, 'r', linestyles= 'dashed')
    plt.subplot(412)
    plt.ylabel('r-r (ms)')
    plt.xlabel('Time (s)')
    plt.title(f'R-R without outliers from ppg_3 - {outlier_percentage:.2f}% of samples dropped')
    plt.plot(rr_timestamps[2], rr_no_oulier)
    plt.subplot(413)
    plt.ylabel('r-r (ms)')
    plt.xlabel('Time (s)')
    plt.title('R-R interpolated from ppg_3')
    plt.plot(rr_timestamps[2], rr_interpol_oulier)
    plt.subplot(414)
    plt.ylabel('r-r (ms)')
    plt.xlabel('Time (s)')
    plt.title('Artifact corrected R-R from Firstbeat')
    plt.plot(ecg_rr_timestamps, ecg_rr[-1][closest_index:])
    plt.show()
    
    delay, corr = find_delay(rr[2], ecg_rr[-1])
    
    