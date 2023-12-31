import numpy as np
from scipy import linalg as sp_linalg
from scipy import diag as sp_diag
import matplotlib.pyplot as plt
import mne
import pandas as pd
from dreamer_read import get_trial_eeg, dreamer_to_np
#from epoching_eeg import get_data
#from epoching_eeg import epochs
#from eeg_preprocess import eeg, eeg_filt, eeg_clean

# Correlated Component Analysis (CorrCA) method based on
# original matlab code from Parra's lab (https://www.parralab.org/corrca/).
# original python code from Renzo Comolatti (https://github.com/renzocom/CorrCA/blob/master/CorrCA.py)

def fit(X, version=2, gamma=0, k=None):
    '''
    Correlated Component Analysis (CorrCA).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal to calculate CorrCA.
    k : int,
        Truncates eigenvalues on the Kth component.
    gamma : float,
        Truncates eigenvalues using SVD.

    Returns
    -------
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).
    ISC : list of floats
        Inter-subject Correlation values.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).
    '''

    # TODO: implement case 3, tsvd truncation

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    if k is not None: # truncate eigenvalues using SVD
        gamma = 0
    else:
        k = D

    # Compute within- (Rw) and between-subject (Rb) covariances
    if False: # Intuitive but innefficient way to calculate Rb and Rw
        Xcat = X.reshape((N * D, T)) # T x (D + N) note: dimensions vary first, then subjects
        Rkl = np.cov(Xcat).reshape((N, D, N, D)).swapaxes(1, 2)
        Rw = Rkl[range(N), range(N), ...].sum(axis=0) # Sum within subject covariances
        Rt = Rkl.reshape(N*N, D, D).sum(axis=0)
        Rb = (Rt - Rw) / (N-1)

    # Rw = sum(np.cov(X[n,...]) for n in range(N))
    # Rt = N**2 * np.cov(X.mean(axis=0))
    # Rb = (Rt - Rw) / (N-1)

    # fix for channel specific bad trial
    temp = [np.cov(X[n,...]) for n in range(N)]
    Rw = np.nansum(temp, axis=0)
    Rt = N**2 * np.cov(np.nanmean(X, axis=0))
    Rb = (Rt - Rw) / (N-1)

    rank = np.linalg.matrix_rank(Rw)
    if rank < D and gamma != 0:
        print('Warning: data is rank deficient (gamma not used).')

    k = min(k, rank) # handle rank deficient data.
    if k < D:
        def regInv(R, k):
            '''PCA regularized inverse of square symmetric positive definite matrix R.'''

            U, S, Vh = np.linalg.svd(R)
            invR = U[:, :k].dot(np.diag(1 / S[:k])).dot(Vh[:k, :])
            return invR

        invR = regInv(Rw, k)
        ISC, W = sp_linalg.eig(invR.dot(Rb))
        ISC, W = ISC[:k], W[:, :k]

    else:
        Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
        ISC, W = sp_linalg.eig(Rb, Rw_reg) # W is already sorted by eigenvalue and normalized

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))

    ISC, W = np.real(ISC), np.real(W)

    if k==D:
        A = Rw.dot(W).dot(sp_linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))

    return W, ISC, A

def transform(X, W):
    '''
    Get CorrCA components from signal(X), e.g. epochs or evoked, using backward model (W).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times) or (n_dim, n_times)
        Signal  to transform.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    '''

    flag = False
    if X.ndim == 2:
        flag = True
        X = X[np.newaxis, ...]
    N, _, T = X.shape
    K = W.shape[1]
    Y = np.zeros((N, K, T))
    for n in range(N):
        Y[n, ...] = W.T.dot(X[n, ...])
    if flag:
        Y = np.squeeze(Y, axis=0)
    return Y

def get_ISC(X, W):
    '''
    Get ISC values from signal (X) and backward model (W)

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal to calculate CorrCA.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    ISC : list of floats
        Inter-subject Correlation values.
    '''
    N, D, T = X.shape

    Rw = sum(np.cov(X[n,...]) for n in range(N))
    Rt = N**2 * np.cov(X.mean(axis=0))
    Rb = (Rt - Rw) / (N-1)

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))
    return np.real(ISC)

def get_forwardmodel(X, W):
    '''
    Get forward model from signal(X) and backward model (W).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal  to transform.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).
    '''

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    Rw = sum(np.cov(X[n,...]) for n in range(N))
    Rt = N**2 * np.cov(X.mean(axis=0))
    Rb = (Rt - Rw) / (N-1)

    k = np.linalg.matrix_rank(Rw)
    if k==D:
        A = Rw.dot(W).dot(sp_linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))
    return A

def reconstruct(Y, A):
    '''
    Reconstruct signal(X) from components (Y) and forward model (A).

    Parameters
    ----------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).

    Returns
    -------
    X : ndarray of shape = (n_subj, n_dim, n_times) or (n_dim, n_times)
        Signal.
    '''

    flag = False
    if Y.ndim == 2:
        flag = True
        Y = Y[np.newaxis, ...]
    N, _, T = Y.shape
    D = A.shape[0]
    X = np.zeros((N, D, T))
    for n in range(N):
        X[n, ...] = A.dot(Y[n, ...])

    if flag:
        X = np.squeeze(X, axis=0)
    return X

def stats(X, gamma=0, k=None, n_surrogates=200, alpha=0.05):
    '''
    Compute ISC statistical threshold using circular shift surrogates.
    Parameters
    ----------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).

    Returns
    -------
    '''
    ISC_null = []
    for n in range(n_surrogates):
        if n%10==0:
            print('#', end='')
        surrogate = circular_shift(X)
        W, ISC, A = fit(surrogate, gamma=gamma, k=k)
        ISC_null.append(ISC[0]) # get max ISC
    ISC_null = np.array(ISC_null)
    thr = np.percentile(ISC_null, (1 - alpha) * 100)
    print('')
    return thr, ISC_null

def circular_shift(X):
    n_reps, n_dims, n_times = X.shape
    shifts = np.random.choice(range(n_times), n_reps, replace=True)
    surrogate = np.zeros_like(X)
    for i in range(n_reps):
        surrogate[i, ...] = np.roll(X[i, ...], shifts[i], axis=1)
    return surrogate

def calc_corrca(epochs, times, **par):
    ini_ix = time2ix(times, par['response_window'][0])
    end_ix = time2ix(times, par['response_window'][1])
    X = np.array(epochs)[..., ini_ix : end_ix]

    W, ISC, A = fit(X, gamma=par['gamma'], k=par['K'])

    n_components = W.shape[1]
    if stats:
        print('Calculating statistics...')
        ISC_thr, ISC_null = stats(X, par['gamma'], par['K'], par['n_surrogates'], par['alpha'])
        n_components = sum(ISC > ISC_thr)
        W, ISC, A = W[:, :n_components], ISC[:n_components], A[:, :n_components]
        
    Y = transform(X, W)
    Yfull = transform(np.array(epochs), W)
    return W, ISC, A, Y, Yfull, ISC_thr

def time2ix(times, t):
    return np.abs(times - t).argmin()

def CorrCA(X, W=None, version=2, gamma=0, k=None):
    '''
    Correlated Component Analysis.

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal to calculate CorrCA.
    k : int,
        Truncates eigenvalues on the Kth component.
    gamma : float,
        Truncates eigenvalues using SVD.
        
    Returns
    -------
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).
    ISC : list of floats
        Inter-subject Correlation values.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
         CorrCA components.
    '''

    # TODO: implement case 3, tsvd truncation

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    if k is not None: # truncate eigenvalues using SVD
        gamma = 0
    else:
        k = D

    # Compute within- and between-subject covariances
    if version == 1:
        Xcat = X.reshape((N * D, T)) # T x (D + N) note: dimensions vary first, then subjects
        Rkl = np.cov(Xcat).reshape((N, D, N, D)).swapaxes(1, 2)
        Rw = Rkl[range(N), range(N), ...].sum(axis=0) # Sum within subject covariances
        Rt = Rkl.reshape(N*N, D, D).sum(axis=0)
        Rb = (Rt - Rw) / (N-1)

    elif version == 2:
        Rw = sum(np.cov(X[n,...]) for n in range(N))
        Rt = N**2 * np.cov(X.mean(axis=0))
        Rb = (Rt - Rw) / (N-1)

    elif version == 3:
        pass

    if W is None:
        k = min(k, np.linalg.matrix_rank(Rw)) # handle rank deficient data.
        if k < D:
            def regInv(R, k):
                '''PCA regularized inverse of square symmetric positive definite matrix R.'''

                U, S, Vh = np.linalg.svd(R)
                invR = U[:, :k].dot(np.diag(1 / S[:k])).dot(Vh[:k, :])
                return invR

            invR = regInv(Rw, k)
            ISC, W = sp_linalg.eig(invR.dot(Rb))
            ISC, W = ISC[:k], W[:, :k]

        else:
            Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
            ISC, W = sp_linalg.eig(Rb, Rw_reg) # W is already sorted by eigenvalue and normalized

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))

    ISC, W = np.real(ISC), np.real(W)

    Y = np.zeros((N, k, T))
    for n in range(N):
        Y[n, ...] = W.T.dot(X[n, ...])

    if k==D:
        A = Rw.dot(W).dot(sp_linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))

    return W, ISC, Y, A

#

def get_rolling_ISC(X, fs=125, k=None, window_s=5, overlap=0.8):
    _, num_channels, num_samples = X.shape
    # Calculate the number of windows based on the window size and overlap
    window_size = window_s * fs
    step_size = int(window_size * (1 - overlap))
    num_windows = int(np.ceil((num_samples - window_size) / step_size)) + 1
    
    W, ISC_total ,_ = fit(X, k=k)
    
    # Apply the rolling window and compute ISC for each window
    ISC_values = []
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        if end > num_samples:
            end = num_samples
    
        window_data = X[:, :, start:end]
        ISC = get_ISC(window_data, W)
        ISC_values.append(ISC)
    ISC_values = np.array(ISC_values, dtype=float)
    
    time_values = np.arange(0, num_windows) * (step_size / fs)
    
    return ISC_values, ISC_total, W, time_values, num_windows, step_size

def count_signif_windows(ISC_values, thr):
    count = 0
    for item in ISC_values:
        if item > thr:
            count += 1
    percentage = (count / len(ISC_values)) * 100
    return count, percentage

def main():
    epochs = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs.npy')
    epochs_filt = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs_filt.npy')
    #epochs_clean = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs_clean.npy')
    
    epochs_nobads = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs_nobads.npz', allow_pickle=True)
    epochs_nobads = [epochs_nobads[subject] for subject in epochs_nobads]
    epochs_nobads.pop(9)
    
    epochs_clean = np.load('D:/Downloads 2/Andre_Thesis/Data/WOLF_eeg_preprocess/epochs_ica_clean.npy', allow_pickle=True)
    
    #dreamer,_,_ = dreamer_to_np('D:/Downloads 2/Andre_Thesis/Archive/DREAMER.mat')
    
    # Set EEG info
    fs = 125
    #ch_names = ['AF3','AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6',
    #                          'P7', 'P8', 'T7', 'T8', 'O1', 'O2']
    ch_names = ['Fp1','Fp2','C3','C4','P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3',
          'F4', 'T7', 'T8','P3','P4']
    mont = mne.channels.make_standard_montage("standard_1020")
    #mont.plot()

    # Create an MNE Info object
    info = mne.create_info(ch_names=ch_names, ch_types= 'eeg',
                           sfreq=fs)
    info.set_montage(mont, on_missing='ignore')

    # MNE topomap based on data from a pandas dataframe emulating components
    # Set plotting style to default for plotting with MNE functions
    plt.style.use('default')
    map_info = info
    
    a=[]
    b=np.array((4,16,3125*8))
    for subject in range(4):
        for trial in range(8):
            a=eeg_epochs[subject,trial].get_data()
        b[subject,:,trial*3125]=a
    
        eeg_epochs = epochs_clean
        # Rolling ISC
        k=3
        for trial in range(8):
            a = eeg_epochs[:,trial]
            a = [obj for obj in a if obj is not None]
            if a != []:
                a2 = np.empty((len(a),16,3125))
                for i in range(len(a)):
                    a2[i,:,:] = a[i].get_data()
                eeg=a2
                thr, ISC_null = stats(eeg, k=k, n_surrogates=200)
                ISC_values, ISC_total, W, time_values, num_windows, step_size = get_rolling_ISC(eeg, k=k, fs=fs, window_s=2, overlap=0.8)
            
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
                plt.suptitle('Correlated Component Analysis - Trial {} - {} subjects'.format(trial, len(a)), y=0.99)
                plt.show()

if __name__ == '__main__':
    main()


"""
a=[]
b=np.array((4,16,3125*8))
for subject in range(4):
    for trial in range(8):
        a=eeg_epochs[subject,trial].get_data()
    b[subject,:,trial*3125]=a

    eeg_epochs = epochs_clean
    # Rolling ISC
    k=3
    for trial in range(8):
        a = eeg_epochs[:,trial]
        a = [obj for obj in a if obj is not None]
        if a != []:
            a2 = np.empty((len(a),16,3125))
            for i in range(len(a)):
                a2[i,:,:] = a[i].get_data()
            eeg=a2
            thr, ISC_null = stats(eeg, k=k, n_surrogates=200)
            ISC_values, ISC_total, W, time_values, num_windows, step_size = get_rolling_ISC(eeg, k=k, fs=fs, window_s=2, overlap=0.8)
        
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
            plt.suptitle('Correlated Component Analysis - Trial {} - {} subjects'.format(trial, len(a)), y=0.99)
            plt.show()
            #plt.savefig(f'D:\Downloads 2\Andre_Thesis\Analysis\CorrCA/CorrCA_clean{trial}')
"""    
    
"""
    eeg_epochs = epochs_nobads
    # Rolling ISC
    k=3
    for trial in range(8):
        a=[]
        for subjectdata in eeg_epochs:
            a.append(subjectdata[trial].get_data())
        a = np.stack(a,axis=0)
        eeg=a
        thr, ISC_null = stats(eeg, k=k, n_surrogates=200)
        ISC_values, ISC_total, W, time_values, num_windows, step_size = get_rolling_ISC(eeg, k=k, fs=fs, window_s=2, overlap=0.8)
    
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
        plt.suptitle('Correlated Component Analysis - Trial {} - {} subjects'.format(trial, len(a)), y=0.99)
        #plt.show()
        plt.savefig(f'D:\Downloads 2\Andre_Thesis\Analysis\CorrCA/CorrCA_nobads{trial}')
"""   
    
    
    
    
    
    
    
    
"""
#CorrCA in one trial
X = get_trial_eeg(data, trial_index=0)
W, ISC, Y, A = CorrCA(X, k=5)

# Plot 3 first components
plt.figure(figsize=(18,11))
plt.suptitle('Projections onto ISC components')
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(Y[0, i,:128])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Component {}, ISC: {:.6f}'.format(i+1, ISC[i]))
plt.show()


#CorrCA for each trial
W_tot = []
ISC_tot = []
for trial in range(data.shape[1]):
    X = get_trial_eeg(data, trial_index=trial)
    W, ISC, Y, A = CorrCA(X, k=5)
    W_tot.append(W)
    ISC_tot.append(ISC)
    
#Calculate average W and ISC
W_tot = np.array(W_tot, dtype=float)
W_avg = np.mean(W_tot, axis=0, dtype=float)
ISC_tot = np.array(ISC_tot, dtype=float)
ISC_avg = np.mean(ISC_tot, axis=0, dtype=float)



# Plot Components
for trial in range(3):#W_tot.shape[0]):
    # Make plot window
    fig, axes = plt.subplots(figsize=(4*W.shape[1],4), ncols=W.shape[1])
    for ax, c in zip(axes, range(W.shape[1])):
        ax.set_title("C{}".format(c+1))
        im, cn = mne.viz.plot_topomap(W_tot[trial,:,c],pos=map_info,
                                       axes=ax, show=False, cmap = "RdBu_r",
                                       outlines = "head")
        
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    plt.suptitle('First three correlated components - Trial {}'.format(trial+1))
    plt.show()
    
# Make plot window
fig, axes = plt.subplots(figsize=(4*W.shape[1],4), ncols=W.shape[1])
for ax, c in zip(axes, range(W.shape[1])):
    ax.set_title("C{}, ISC: {:.6f}".format(c+1, ISC_avg[c]))
    im, cn = mne.viz.plot_topomap(W_avg[:,c],pos=map_info,
                                   axes=ax, show=False, cmap = "RdBu_r",
                                   outlines = "head")
    
# Add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    
"""