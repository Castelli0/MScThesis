# DreamER and auxiliary dataset pre-processing

## Installation instructions

To install the required libraries, you can use the following command in your terminal:

```
pip install -r requirements.txt
```

## Usage

### CorrCA analysis

To run CorrCA analysis, you will need to have the following data:

* A list of EEG epochs, each epoch being a 2D array of size (num_channels, num_samples).
* A list of corresponding labels for each epoch.

Once you have your data, you can run the CorrCA analysis using the following code:

```
import CorrCA

# Load EEG epochs and labels
eeg_epochs = np.load('EEG_epochs.npy')
labels = np.load('labels.npy')

# Set the desired parameters for CorrCA
params = {
    'gamma': 0,
    'k': 3,
    'n_surrogates': 200,
    'alpha': 0.05
}

# Run CorrCA analysis
W, ISC, A, Y, Yfull, ISC_thr = CorrCA.calc_corrca(eeg_epochs, **params)
```

### Epoch extraction

To extract epochs from a continuous EEG signal, you can use the `epoching_eeg` module. This module provides functions to:

* Load EEG data from a file.
* Extract epochs from the EEG data.
* Label the epochs with the corresponding conditions.

To use the `epoching_eeg` module, you can follow these steps:

1. Import the `epoching_eeg` module.
2. Load the EEG data using the `load_eeg_data()` function.
3. Extract epochs from the EEG data using the `extract_epochs()` function.
4. Label the epochs with the corresponding conditions using the `label_epochs()` function.

Here is an example of how to use the `epoching_eeg` module:

```
import epoching_eeg

# Load EEG data
eeg_data = epoching_eeg.load_eeg_data('EEG_data.edf')

# Extract epochs
epochs = epoching_eeg.extract_epochs(eeg_data, epoch_length=1, overlap=0.5)

# Label epochs
labels = epoching_eeg.label_epochs(epochs, conditions=['A', 'B'])
```

### EEG pre-processing

To pre-process EEG data, you can use the `eeg_preprocess` module. This module provides functions to:

* Filter the EEG data.
* Remove artifacts from the EEG data.
* Re-reference the EEG data.

To use the `eeg_preprocess` module, you can follow these steps:

1. Import the `eeg_preprocess` module.
2. Load the EEG data using the `load_eeg_data()` function.
3. Pre-process the EEG data using the `preprocess_eeg()` function.

Here is an example of how to use the `eeg_preprocess` module:

```
import eeg_preprocess

# Load EEG data
eeg_data = eeg_preprocess.load_eeg_data('EEG_data.edf')

# Pre-process EEG data
eeg_data = eeg_preprocess.preprocess_eeg(eeg_data, low_freq=1, high_freq=40, notch_freq=50)
```

## Architecture Overview

The `Analysis` module contains the following submodules:

* `CorrCA`: Contains the CorrCA algorithm implementation.
* `dreamer_read`: Contains functions for reading DREAMER data.
* `eeg_preprocess`: Contains functions for pre-processing EEG data.
* `epoching_eeg`: Contains functions for extracting epochs from continuous EEG data.

## Technologies Used

The following technologies were used in the development of this project:

* Python
* NumPy
* SciPy
* Pandas
* Matplotlib
* MNE
* DreamER