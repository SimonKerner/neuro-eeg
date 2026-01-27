import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

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
# Optional: map 22 channels into LaBraM's 64 montage indices (0..63)
# NOTE: This is a heuristic anatomical mapping; keep consistent if you use it.
# --------------------------------------------------
LABRAM_64_MAP = {
    "Fz": 2, "Fc3": 10, "Fc1": 11, "Fcz": 12, "Fc2": 13, "Fc4": 14,
    "C5": 26, "C3": 27, "C1": 28, "Cz": 29, "C2": 30, "C4": 31, "C6": 32,
    "Cp3": 41, "Cp1": 42, "Cpz": 43, "Cp2": 44, "Cp4": 45,
    "P1": 51, "Pz": 52, "P2": 53, "Poz": 59
}


def clean_ch_name(ch: str) -> str:
    # EEGMMIDB often has dots etc.
    return ch.replace(".", "").capitalize()


def run_to_task_type(run: int) -> Optional[str]:
    # EEGMMIDB convention
    if run in [4, 8, 12]:
        return "imagined"
    elif run in [3, 7, 11]:
        return "real"
    return None


# EEGMMIDB annotations commonly include T1/T2 for left/right
CLASS_MAP = {"T1": 0, "T2": 1}


@dataclass
class PreprocConfig:
    target_sfreq: float = 200.0
    notch_hz: float = 50.0
    l_freq: float = 0.1
    h_freq: float = 75.0
    reref: str = "average"   # "average" or None
    to_microvolts: bool = True


class EEGMMIDBLaBraMDataset(Dataset):
    """
    EEGMMIDB -> LaBraM-ready trials.

    Returns dict:
      {
        "inputs": FloatTensor [C, T]  (or [64, 1600] if pad_to_64=True),
        "labels": LongTensor [],
        "meta": { ... useful debug info ... }
      }

    Typical LaBraM paper-like setup:
      - resample to 200 Hz
      - notch 50 Hz
      - bandpass 0.1-75 Hz
      - unit uV

    Trial extraction:
      - uses events from annotations
      - extracts a window [t_min, t_max] seconds relative to event onset
    """

    def __init__(
        self,
        root_path: str,
        subjects: List[int],
        runs: List[int],
        t_min: float = 0.0,
        t_max: float = 4.0,
        normalization: bool = True,   # z-score per channel
        is_train: bool = True,
        add_noise_std: float = 0.0,   # e.g. 0.02 if inputs are normalized; 0.0 default
        preproc: PreprocConfig = PreprocConfig(),
        pad_to_64: bool = False,
        pad_time_len: int = 1600,     # buffer time length if pad_to_64=True
        pad_place_len: int = 800,     # how many samples to place (e.g. 4s*200Hz)
    ):
        self.root_path = root_path
        self.subjects = subjects
        self.runs = runs
        self.t_min = t_min
        self.t_max = t_max
        self.normalization = normalization
        self.is_train = is_train
        self.add_noise_std = float(add_noise_std)
        self.preproc = preproc
        self.pad_to_64 = pad_to_64
        self.pad_time_len = int(pad_time_len)
        self.pad_place_len = int(pad_place_len)

        self.data: List[np.ndarray] = []
        self.labels: List[int] = []
        self.metas: List[Dict[str, Any]] = []

        self._load_all()

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def _zscore_per_channel(x: np.ndarray) -> np.ndarray:
        # x: (C, T)
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        return (x - mean) / (std + 1e-6)

    @staticmethod
    def _pad_map_22_to_64(x_22: np.ndarray, place_len: int, total_len: int) -> np.ndarray:
        # x_22: (22, T)
        out = np.zeros((64, total_len), dtype=np.float32)
        for i, ch in enumerate(BCI22_CHANNELS):
            tgt = LABRAM_64_MAP[ch]
            out[tgt, :place_len] = x_22[i, :place_len]
        return out

    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw = raw.copy()
    
        # EEG only (newer API)
        raw.pick(picks="eeg")
    
        # clean names (EEGMMIDB)
        raw.rename_channels({ch: clean_ch_name(ch) for ch in raw.ch_names})
    
        # pick + reorder EXACT BCI22 (lock order explicitly)
        missing = [ch for ch in BCI22_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise RuntimeError(f"Missing channels: {missing}")
    
        raw.pick_channels(BCI22_CHANNELS, ordered=True)
    
        # reference
        if self.preproc.reref == "average":
            raw.set_eeg_reference("average", verbose=False)
    
        # resample -> target
        raw.resample(self.preproc.target_sfreq, verbose=False)
    
        # notch
        if self.preproc.notch_hz is not None:
            raw.notch_filter(float(self.preproc.notch_hz), verbose=False)
    
        # bandpass
        raw.filter(float(self.preproc.l_freq), float(self.preproc.h_freq), verbose=False)
    
        return raw

    def _load_all(self) -> None:
        for subj in self.subjects:
            subj_id = f"S{subj:03d}"
            for run in self.runs:
                task_type = run_to_task_type(run)
                if task_type is None:
                    continue

                fname = f"{subj_id}R{run:02d}.edf"
                edf_path = os.path.join(self.root_path, subj_id, fname)
                if not os.path.isfile(edf_path):
                    continue

                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                raw = self._preprocess_raw(raw)

                sfreq = float(raw.info["sfreq"])
                seg_len = int(round((self.t_max - self.t_min) * sfreq))

                events, event_id = mne.events_from_annotations(raw, verbose=False)

                # Build reverse map: code -> label string (T1/T2/...)
                code_to_name = {v: k.split("/")[-1] for k, v in event_id.items()}

                for onset, _, code in events:
                    label_name = code_to_name.get(code, None)
                    if label_name not in CLASS_MAP:
                        continue

                    label = CLASS_MAP[label_name]

                    start = int(round(onset + self.t_min * sfreq))
                    stop = start + seg_len
                    if stop > raw.n_times:
                        continue

                    x = raw.get_data(start=start, stop=stop).astype(np.float32)  # (22, T)

                    # convert to microvolts if data is in volts
                    if self.preproc.to_microvolts:
                        # MNE raw.get_data returns Volts for EDF EEG
                        x = x * 1e6

                    # channel-wise normalization
                    if self.normalization:
                        x = self._zscore_per_channel(x)

                    # optional pad/map to 64 channels
                    if self.pad_to_64:
                        place_len = min(self.pad_place_len, x.shape[1])
                        x64 = self._pad_map_22_to_64(x, place_len=place_len, total_len=self.pad_time_len)
                        x = x64

                    self.data.append(x)
                    self.labels.append(int(label))
                    self.metas.append(
                        {
                            "subject": subj_id,
                            "run": run,
                            "task_type": task_type,
                            "label_name": label_name,
                            "sfreq": sfreq,
                            "ch_names": list(BCI22_CHANNELS),
                            "t_min": self.t_min,
                            "t_max": self.t_max,
                            "path": edf_path,
                        }
                    )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.data[idx]  # (C, T) or (64, 1600)
        y = self.labels[idx]
        meta = self.metas[idx]

        # data augmentation (optional) — use only if normalized
        if self.is_train and self.add_noise_std > 0.0:
            x = x + np.random.normal(0.0, self.add_noise_std, size=x.shape).astype(np.float32)

        return {
            "inputs": torch.from_numpy(x).float(),
            "labels": torch.tensor(y, dtype=torch.long),
            "meta": meta,
        }