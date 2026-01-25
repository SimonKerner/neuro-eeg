import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mne

# 1. Standard BCI IV 2a Channel Order
BCI22_CHANNELS = [
    "Fz", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "P1", "Pz", "P2", "Poz"
]

# 2. LaBraM 64-Channel Montage (Simplified 10-20 Mapping)
# This maps your 22 channels to their approximate anatomical index 
# in the standard 64-channel montage LaBraM was trained on.
LABRAM_64_MAP = {
    "Fz": 2, "Fc3": 10, "Fc1": 11, "Fcz": 12, "Fc2": 13, "Fc4": 14,
    "C5": 26, "C3": 27, "C1": 28, "Cz": 29, "C2": 30, "C4": 31, "C6": 32,
    "Cp3": 41, "Cp1": 42, "Cpz": 43, "Cp2": 44, "Cp4": 45,
    "P1": 51, "Pz": 52, "P2": 53, "Poz": 59
}

class MotorImageryDataset(Dataset):
    def __init__(self, filenames, root_path="", is_train=True, normalization=True):
        self.is_train = is_train
        self.do_normalization = normalization
        
        # Load Raw Data
        self.data_all = []
        for fn in filenames:
            full_path = os.path.join(root_path, fn)
            if os.path.isfile(full_path):
                self.data_all.append(np.load(full_path))

        self.mi_types = {769: 'left', 770: 'right'}
        self.labels_string2int = {'left': 0, 'right': 1}
        
        # Extract and Preprocess all trials
        self.trials, self.labels = self.get_trials_all()

    def __len__(self):
        return len(self.labels)

    def normalize(self, data):
        """Z-score normalization per channel."""
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        return (data - mean) / (std + 1e-25)

    def pad_and_map(self, x):
        """
        Anatomically maps 22 channels into a 64x1600 frame.
        """
        padded_x = np.zeros((64, 1600))
        
        for i, ch_name in enumerate(BCI22_CHANNELS):
            target_idx = LABRAM_64_MAP[ch_name]
            # Place 4s trial (800 samples) into the buffer
            # We use 800 samples because data is resampled to 200Hz
            padded_x[target_idx, :800] = x[i, :800]
            
        return padded_x

    def get_trials_from_single_subj(self, sub_id):
        # 1. Extract raw data (22, Time)
        # S contains 22 channels + others; we take first 22
        raw_data = self.data_all[sub_id]['s'].T[:22, :]
        
        # 2. Resample from 250Hz to 200Hz (REQUIRED for LaBraM)
        # Use MNE for high-quality resampling
        info = mne.create_info(ch_names=BCI22_CHANNELS, sfreq=250, ch_types='eeg')
        raw = mne.io.RawArray(raw_data, info, verbose=False)
        raw.resample(200, verbose=False)
        resampled_data = raw.get_data()
        
        events_type = self.data_all[sub_id]['etyp'].T
        events_position = self.data_all[sub_id]['epos'].T
        
        # Adjust trigger positions for 200Hz (pos * 200 / 250)
        idxs = [i for i, x in enumerate(events_type[0]) if x == 768]
        trial_labels = [self.labels_string2int[self.mi_types[ev]] 
                        for ev in events_type[0] if ev in self.mi_types]

        trials, classes = [], []
        for j, index in enumerate(idxs):
            if j >= len(trial_labels): continue
            
            # Start 0.5s after cue to skip the visual 'flash'
            # (Original pos * 0.8 to adjust for 200Hz)
            start = int(events_position[0, index] * 0.8) + 100 
            stop = start + 800 # Take 4 seconds
            
            if stop <= resampled_data.shape[1]:
                trial = resampled_data[:, start:stop]
                # Scale to Microvolts
                if np.max(np.abs(trial)) < 1e-3: trial *= 1e6
                trials.append(trial)
                classes.append(trial_labels[j])
                
        return trials, classes

    def get_trials_all(self):
        trials_list, labels_list = [], []
        for sub_id in range(len(self.data_all)):
            t, l = self.get_trials_from_single_subj(sub_id)
            if t:
                trials_list.extend(t)
                labels_list.extend(l)
        return trials_list, labels_list

    def __getitem__(self, idx):
        x = self.pad_and_map(self.trials[idx])
        
        if self.do_normalization:
            x = self.normalize(x)
            
        # DATA AUGMENTATION: Add noise during training to stop overfitting
        if self.is_train:
            noise = np.random.normal(0, 0.02, x.shape)
            x = x + noise

        return {
            "inputs": torch.from_numpy(x).float(),
            "labels": torch.tensor(self.labels[idx]).long()
        }