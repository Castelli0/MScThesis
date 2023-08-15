import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowPresets
from scipy.signal import resample
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

data_path = "D:/Downloads 2/Andre_Thesis/Data/WOLF_dataset/Subject_0/Subject_0_"

columns = ['package_num_channel', 'accel_1', 'accel_2', 'accel_3', 'gyro_1', 'gyro_2', 'gyro_3', 'mag_1', 'mag_2', 'mag_3', 'timestamp', 'marker']
df = pd.read_csv(data_path + "emotibit.csv", sep='\t', header=None, names=columns)

columns_aux = ['package_num_channel', 'ppg_1', 'ppg_2', 'ppg_3', 'timestamp', 'marker']
df_aux = pd.read_csv(data_path + "emotibit_aux.csv", sep='\t', header=None, names=columns_aux)

columns_anc = ['package_num_channel', 'eda_1', 'temperature', 'other', 'timestamp', 'marker']
df_anc = pd.read_csv(data_path + "emotibit_anc.csv", sep='\t', header=None, names=columns_anc)

df = pd.read_cs

emotibit_data = df.values
emotibit_data_aux = df_aux.values
emotibit_data_anc = df_anc.values

for column in range(emotibit_data.shape[1]):
    plt.figure()
    plt.title('default {}'.format(columns[column]))
    plt.plot(emotibit_data[:,column])
    plt.show()
    
for column in range(emotibit_data_aux.shape[1]):
    plt.figure()
    plt.title('aux {}'.format(columns_aux[column]))
    plt.plot(emotibit_data_aux[:,column])
    plt.show()
    
for column in range(emotibit_data_anc.shape[1]):
    plt.figure()
    plt.title('anc {}'.format(columns_anc[column]))
    plt.plot(emotibit_data_anc[:,column])
    plt.show()
    
    
    
# Upsample EDA and temp
max_len = max(len(emotibit_data), len(emotibit_data_aux), len(emotibit_data_anc))

emotibit_data_aux_up = resample(emotibit_data_aux, max_len)
emotibit_data_anc_up = resample(emotibit_data_anc, max_len)

# plot
for column in range(1,3):
    plt.figure()
    plt.title('RESAMPLE anc {}'.format(columns_anc[column]))
    plt.plot(emotibit_data_anc_up[:,column])
    plt.show()


# Upsample to match max length
max_len = max(len(emotibit_data), len(emotibit_data_aux), len(emotibit_data_anc))

# select accel, gyro, and mag
data = emotibit_data[:, 1:10]

# interpolate PPG
for i in range(1,4):
    interpolator = interp1d(np.arange(len(emotibit_data_aux)), emotibit_data_aux[:,i])
    interpolated = interpolator(np.linspace(0, len(emotibit_data_aux) - 1, max_len)).reshape(max_len,1)
    data = np.concatenate((data, interpolated), axis=1)

# interpolate EDA, temp   
for i in range(1,3):
    interpolator = interp1d(np.arange(len(emotibit_data_anc)), emotibit_data_anc[:,i])
    interpolated = interpolator(np.linspace(0, len(emotibit_data_anc) - 1, max_len)).reshape(max_len,1)
    data = np.concatenate((data, interpolated), axis=1)


plt.figure(figsize=(20,8))
plt.subplot(2,1,1)
plt.plot(emotibit_data_anc[:,1])
plt.subplot(2,1,2)
plt.plot(data[:,12])

# save data to DataFrame
df_up = pd.DataFrame(data.T, columns = ['accel_1', 'accel_2', 'accel_3', 
                                     'gyro_1', 'gyro_2', 'gyro_3', 
                                     'mag_1', 'mag_2', 'mag_3',
                                     'ppg_1', 'ppg_2', 'ppg_3',
                                     'eda1', 'temperature'])
    
# plot
for column in range(14):
    plt.figure()
    plt.title('UP')
    plt.plot(data[:, column])
    plt.show()


print(BoardShim.get_board_descr(BoardIds.EMOTIBIT_BOARD.value))  
print(BoardShim.get_board_descr(BoardIds.EMOTIBIT_BOARD.value, preset=BrainFlowPresets.AUXILIARY_PRESET))
print(BoardShim.get_board_descr(BoardIds.EMOTIBIT_BOARD.value, preset=BrainFlowPresets.ANCILLARY_PRESET))