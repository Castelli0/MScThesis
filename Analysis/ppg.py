import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import biosppy.signals.ppg as ppg

data_path = "C:/Users/andre/Documents/DTU/Thesis/Data/TestData/OBS/"
df = pd.read_csv(data_path + "test_obs_emotibit.csv", index_col=0)

ppg_data = df[['ppg_1','ppg_2','ppg_3']].values
rr = []#np.empty((len(onsets)-1, 3))
rr_timestamps = []#np.empty_like(rr)

for channel in range(ppg_data.shape[1]):
    ts, filtered, onsets, HR_ts, HR = ppg.ppg(ppg_data[:,channel], sampling_rate=25,
                                              show=False)
    rr_ = np.diff(ts[onsets])
    rr.append(rr_)
    rr_timestamps.append(np.cumsum(rr_))

    plt.figure(figsize=(16, 10))
    plt.suptitle('ppg_{}'.format(channel+1))
    plt.subplot(3,1,1)
    plt.title('Onset detection')
    plt.plot(ts, filtered, label='filtered')
    plt.vlines(ts[onsets], ymin=min(filtered), ymax=max(filtered),
               colors='r', alpha= 0.5, label='Onsets')
    plt.legend()
    plt.subplot(3,1,2)
    plt.title('Heart Rate')
    plt.plot(HR_ts, HR, label='HR')
    plt.ylabel('HR (bpm)')
    plt.subplot(3,1,3)
    plt.title('R-R Intervals')
    plt.plot(np.cumsum(rr_), rr_)
    plt.ylabel('r-r (s)')
    plt.xlabel('time (s)')
    plt.ylim(0)


"""
# ppg.ppg already filters 1Hz-8Hz before finding onsets
# filter 1Hz - 10Hz bandpass
Wn=[1/25/2, 10/25/2]
ppg_data_filt = np.zeros_like(ppg_data)
b, a = signal.butter(N=4, Wn=Wn, btype='band')

for i in range(3):
    ppg.ppg(ppg_data[:,i], sampling_rate=25, show= True)
    
    ppg_data_filt[:,i] = signal.filtfilt(b, a, ppg_data[:,i])
    ppg.ppg(ppg_data_filt[:,i], sampling_rate=25, show= True)

"""