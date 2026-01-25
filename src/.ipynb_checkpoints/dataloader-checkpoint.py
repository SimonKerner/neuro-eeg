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

# --------------------------------------------------
# Run -> task type
# --------------------------------------------------
def run_to_task_type(run):
    if run in [4, 8, 12]:
        return "imagined"
    elif run in [3, 7, 11]:
        return "real"
    return None

# --------------------------------------------------
# Event -> class
# --------------------------------------------------
CLASS_MAP = {
    "T1": 0,  # Left hand
    "T2": 1   # Right hand
}

# --------------------------------------------------
# Channel name cleanup (EEGMMIDB)
# --------------------------------------------------
def clean_ch_name(ch):
    return ch.replace(".", "").capitalize()

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class EEGMMIDBDataset(Dataset):
    """
    PhysioNet EEGMMIDB
    mapped EXACTLY to BCI Competition IV-2a (22 channels)
    """

    def __init__(
        self,
        root_path,
        subjects,
        runs,
        t_min=0.0,
        t_max=2.0,
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

    # --------------------------------------------------
    def _load_all_subjects(self):
        for subj in self.subjects:
            subj_id = f"S{subj:03d}"
            for run in self.runs:
                self._load_single_run(subj_id, run)

    # --------------------------------------------------
    def _load_single_run(self, subj_id, run):
        fname = f"{subj_id}R{run:02d}.edf"
        edf_path = os.path.join(self.root_path, subj_id, fname)

        if not os.path.isfile(edf_path):
            return

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # --- EEG only
        raw.pick("eeg")

        # --- clean channel names
        raw.rename_channels({ch: clean_ch_name(ch) for ch in raw.ch_names})

        # --- pick + reorder EXACT BCI IV-2a channels
        missing = [ch for ch in BCI22_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise RuntimeError(f"Missing channels in {edf_path}: {missing}")

        raw.pick_channels(BCI22_CHANNELS)
        self.ch_names = raw.ch_names

        # --- preprocessing (NeuroGPT-safe)
        raw.set_eeg_reference("average", verbose=False)
        raw.resample(250, verbose=False)
        raw.notch_filter(60, verbose=False)
        raw.filter(0.5, 100.0, verbose=False)

        sfreq = raw.info["sfreq"]
        seg_len = int((self.tmax - self.tmin) * sfreq)

        # --- events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        task_type = run_to_task_type(run)
        if task_type is None:
            return

        for onset, _, code in events:
            label_name = None
            for k, v in event_id.items():
                if v == code:
                    label_name = k.split("/")[-1]
                    break

            if label_name not in CLASS_MAP:
                continue

            label = CLASS_MAP[label_name]

            start = int(onset + self.tmin * sfreq)
            stop = start + seg_len
            if stop > raw.n_times:
                continue

            # (22, T)
            segment = raw.get_data(start=start, stop=stop)
            segment = torch.tensor(segment, dtype=torch.float32)

            # --- channel-wise normalization
            if self.normalization:
                segment = (segment - segment.mean(dim=1, keepdim=True)) / (
                    segment.std(dim=1, keepdim=True) + 1e-6
                )

            self.data.append(segment)
            self.labels.append(label)

    # --------------------------------------------------
    def __len__(self):
        return len(self.data)

    # --------------------------------------------------
    def __getitem__(self, idx):
        full_segment = self.data[idx]  # (22, 1000)

        # split into two temporal chunks (NeuroGPT-style)
        chunk1 = full_segment[:, :500]
        chunk2 = full_segment[:, 500:1000]

        segment = torch.stack([chunk1, chunk2], dim=0)  # (2, 22, 500)

        return {
            "inputs": segment,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
