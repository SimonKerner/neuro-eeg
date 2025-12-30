import os
import numpy as np
import torch
from torch.utils.data import Dataset
import mne

# --------------------------------------------------
# NeuroGPT 22 selected channel
# --------------------------------------------------
NEUROGPT_CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T1","T3","C3","Cz","C4","T4","T2",
    "T5","P3","Pz","P4","T6","O1","Oz","O2"
]

# --------------------------------------------------
# Run - task type
# --------------------------------------------------
def run_to_task_type(run):
    if run in [4, 8, 12]:
        return "imagined"
    elif run in [3, 7, 11]:
        return "real"
    return None

# --------------------------------------------------
# (task_type, event) - class
# --------------------------------------------------
CLASS_MAP = {
    ("imagined", "T2"): 0,  # right imagined
    ("real",     "T2"): 1,  # right real
    ("imagined", "T1"): 2,  # left imagined
    ("real",     "T1"): 3,  # left real
}

# --------------------------------------------------
# Channel name cleanup (EEGMMIDB - standard)
# --------------------------------------------------
def clean_ch_name(ch):
    return ch.replace(".", "").capitalize()

# --------------------------------------------------
# Build NeuroGPT spatial projection matrix
# --------------------------------------------------
def build_neurogpt_projection(ch_names):
    """
    Build a fixed (22 x N) spatial projection matrix.
    """
    ch_names = [clean_ch_name(ch) for ch in ch_names]
    N = len(ch_names)
    P = np.zeros((22, N))

    def set_weight(row, names):
        idxs = [ch_names.index(n) for n in names if n in ch_names]
        if not idxs:
            return
        w = 1.0 / len(idxs)
        for i in idxs:
            P[row, i] = w

    # Exact matches
    set_weight(0, ["Fp1"])
    set_weight(1, ["Fp2"])
    set_weight(2, ["F7"])
    set_weight(3, ["F3"])
    set_weight(4, ["Fz"])
    set_weight(5, ["F4"])
    set_weight(6, ["F8"])

    # Virtual temporal channels
    set_weight(7,  ["Ft7", "C3"])    # T1
    set_weight(8,  ["T7", "C3"])     # T3
    set_weight(9,  ["C3"])
    set_weight(10, ["Cz"])
    set_weight(11, ["C4"])
    set_weight(12, ["T8", "C4"])     # T4
    set_weight(13, ["Ft8", "C4"])    # T2

    set_weight(14, ["Tp7", "P7"])    # T5
    set_weight(15, ["P3"])
    set_weight(16, ["Pz"])
    set_weight(17, ["P4"])
    set_weight(18, ["Tp8", "P8"])    # T6

    set_weight(19, ["O1"])
    set_weight(20, ["Oz"])
    set_weight(21, ["O2"])

    return torch.tensor(P, dtype=torch.float32)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class EEGMMIDBDataset(Dataset):
    """
    EEGMMIDB Dataset adapted for NeuroGPT
    4-class motor imagery / execution classification
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
        self.P = None  # projection matrix

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

        # --- normalize channel names
        rename_map = {ch: clean_ch_name(ch) for ch in raw.ch_names}
        raw.rename_channels(rename_map)

        # --- NeuroGPT-compatible preprocessing
        raw.set_eeg_reference("average", verbose=False)
        raw.resample(250, verbose=False)
        raw.notch_filter(60, verbose=False)
        raw.filter(0.5, 100.0, verbose=False)

        sfreq = raw.info["sfreq"]
        seg_len = int((self.tmax - self.tmin) * sfreq)

        # --- build projection once
        if self.P is None:
            self.P = build_neurogpt_projection(raw.ch_names)

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

            if label_name not in ("T1", "T2"):
                continue

            label = CLASS_MAP[(task_type, label_name)]

            start = int(onset + self.tmin * sfreq)
            stop = start + seg_len
            if stop > raw.n_times:
                continue

            segment = raw.get_data(start=start, stop=stop)
            segment = torch.tensor(segment, dtype=torch.float32)

            # --- spatial projection
            segment = self.P @ segment  # (22, T)

            # --- normalization
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
        return {
            "inputs": self.data[idx],   # (22, T)
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
