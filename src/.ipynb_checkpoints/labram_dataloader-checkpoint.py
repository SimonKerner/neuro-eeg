import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mne

# --------------------------------------------------
# BCI Competition IV-2a channel order (22 channels)
# --------------------------------------------------
BCI22_CHANNELS = [
    "Fz",
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "P1", "Pz", "P2",
    "Poz"
]

def run_to_task_type(run):
    if run in [4, 8, 12]:
        return "imagined"
    elif run in [3, 7, 11]:
        return "real"
    return None

CLASS_MAP = {
    "T1": 0,  # Left hand
    "T2": 1   # Right hand
}

def clean_ch_name(ch):
    return ch.replace(".", "").capitalize()

class EEGMMIDBDataset(Dataset):
    """
    PhysioNet EEGMMIDB Dataloader updated for LaBraM (64x1600 alignment)
    """
    def __init__(
        self,
        root_path,
        subjects,
        runs,
        t_min=0.0,
        t_max=4.0, # We will pad this to 8 seconds (1600 samples)
        normalization=True
    ):
        self.root_path = root_path
        self.subjects = subjects
        self.runs = runs
        self.tmin = t_min
        self.tmax = t_max
        self.normalization = normalization

        self.data = []
        self.labels = []

        self._load_all_subjects()

    def pad_to_labram(self, x):
        """
        Transforms (22, T) data into (128, 1600) to satisfy the 
        pre-trained model's internal 128-channel requirement.
        """
        # 1. Initialize zero buffer (128 channels, 1600 timepoints)
        padded_x = np.zeros((64, 1600))
        
        # 2. Place your 22 channels into the first 22 rows
        # Ensure we don't exceed the 1600 temporal limit
        t_len = min(x.shape[1], 1600)
        padded_x[:22, :t_len] = x[:, :t_len]
        
        return padded_x

    def _load_all_subjects(self):
        for subj in self.subjects:
            subj_id = f"S{subj:03d}"
            for run in self.runs:
                self._load_single_run(subj_id, run)

    def _load_single_run(self, subj_id, run):
        fname = f"{subj_id}R{run:02d}.edf"
        edf_path = os.path.join(self.root_path, subj_id, fname)

        if not os.path.isfile(edf_path):
            return

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick("eeg")
        raw.rename_channels({ch: clean_ch_name(ch) for ch in raw.ch_names})

        # Ensure correct channel order (First 22 indices)
        raw.pick_channels(BCI22_CHANNELS)
        
        # Preprocessing: Match LaBraM Pre-training Specs
        raw.set_eeg_reference("average", verbose=False)
        raw.resample(200, verbose=False) # CRITICAL: Change 250 -> 200
        raw.filter(0.1, 75.0, verbose=False) # Broader filter for foundation models

        sfreq = raw.info["sfreq"] # Now 200
        seg_len = int((self.tmax - self.tmin) * sfreq)

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        task_type = run_to_task_type(run)
        if task_type is None:
            return

        for onset, _, code in events:
            label_name = [k for k, v in event_id.items() if v == code][0].split("/")[-1]
            if label_name not in CLASS_MAP:
                continue

            label = CLASS_MAP[label_name]
            start = int(onset + self.tmin * sfreq)
            stop = start + seg_len # e.g., 4s = 800 samples
            
            if stop > raw.n_times:
                continue

            # (22, T)
            segment = raw.get_data(start=start, stop=stop)
            
            # Scale to Microvolts (Expected by LaBraM)
            segment = segment * 1e6 
            
            # Channel-wise normalization (Z-score)
            if self.normalization:
                mean = segment.mean(axis=1, keepdims=True)
                std = segment.std(axis=1, keepdims=True)
                segment = (segment - mean) / (std + 1e-6)

            # --- APPLY LABRAM PADDING ---
            # Transforms (22, 800) -> (64, 1600)
            padded_segment = self.pad_to_labram(segment)
            
            self.data.append(torch.tensor(padded_segment, dtype=torch.float32))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "inputs": self.data[idx], 
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }