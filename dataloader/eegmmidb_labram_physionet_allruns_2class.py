# dataloader/eegmmidb_labram_dataset_allruns.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import mne


# --------------------------------------------------
# 22-channel motor-focused montage (BCI-style)
# --------------------------------------------------
BCI22_CHANNELS = [
    "Fz", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "P1", "Pz", "P2", "Poz"
]

# Maps BCI22 -> LaBraM 64-slot channel index space
LABRAM_64_MAP = {
    "Fz": 2, "Fc3": 10, "Fc1": 11, "Fcz": 12, "Fc2": 13, "Fc4": 14,
    "C5": 26, "C3": 27, "C1": 28, "Cz": 29, "C2": 30, "C4": 31, "C6": 32,
    "Cp3": 41, "Cp1": 42, "Cpz": 43, "Cp2": 44, "Cp4": 45,
    "P1": 51, "Pz": 52, "P2": 53, "Poz": 59
}


# --------------------------------------------------
# Runs: imagined vs real (EEGMMIDB)
# --------------------------------------------------
def run_to_task_type(run: int) -> Optional[str]:
    if run in [3, 7, 11]:
        return "real"
    if run in [4, 8, 12]:
        return "imagined"
    return None


# --------------------------------------------------
# Labels: left vs right only (ignore task_type)
# In EDF annotations, the event names are typically "T1" and "T2".
# --------------------------------------------------
LR_CLASS_MAP = {
    "T1": 0,  # Left
    "T2": 1,  # Right
}


# --------------------------------------------------
# Channel name cleanup (EEGMMIDB EDF channel names include dots etc.)
# --------------------------------------------------
def clean_ch_name(ch: str) -> str:
    # Examples: "Fc3." -> "Fc3"
    ch = ch.replace(".", "")
    # Normalize case like "FCZ" -> "Fcz" etc. (simple heuristic)
    ch = ch.capitalize()
    return ch


# --------------------------------------------------
# Preprocessing config
# --------------------------------------------------
@dataclass
class PreprocConfig:
    target_sfreq: float = 200.0
    notch_hz: Optional[float] = 50.0
    l_freq: Optional[float] = 0.1
    h_freq: Optional[float] = 75.0
    reref: Optional[str] = "average"  # "average" or None
    to_microvolts: bool = True


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class EEGMMIDBLaBraMAllRunsLRDataset(Dataset):
    """
    EEGMMIDB (PhysioNet) -> Left vs Right classification, pooling both:
      - imagined runs: 3,4
      - real runs:     7,8

    Output:
      inputs: torch.FloatTensor of shape (22, A, patch_size), default (22, 4, 200)
      labels: torch.LongTensor scalar (0=left, 1=right)
      meta: dict with subject/run/task_type/etc.

    Notes:
      - We explicitly restrict to BCI22_CHANNELS and reorder consistently.
      - We resample to 200Hz to match LaBraM patching assumptions (patch_size=200 => 1s).
      - We ignore imagined/real for the label, but keep it in meta.
    """

    def __init__(
        self,
        root_path: str,
        subjects: Optional[Sequence[int]],
        runs: Sequence[int] = (3, 4, 7, 8),
        t_min: float = 0.0,
        t_max: float = 4.0,
        patch_size: int = 200,
        normalization: bool = True,
        is_train: bool = False,
        add_noise_std: float = 0.0,
        preproc: Optional[PreprocConfig] = None,
    ):
        self.root_path = root_path
        if subjects is None:
            # auto-discover subjects from root_path (folders like S001, S002, ...)
            subs = []
            for name in os.listdir(self.root_path):
                if name.startswith('S') and len(name) == 4 and name[1:].isdigit():
                    subs.append(int(name[1:]))
            subs = sorted(subs)
            if not subs:
                raise RuntimeError(f'No subject folders found under: {self.root_path}')
            self.subjects = subs
        else:
            self.subjects = list(subjects)
        self.runs = list(runs)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.patch_size = int(patch_size)
        self.normalization = bool(normalization)
        self.is_train = bool(is_train)
        self.add_noise_std = float(add_noise_std)
        self.preproc = preproc or PreprocConfig()

        self.data: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.metas: List[Dict[str, Any]] = []

        self._load_all()

    # -----------------------------
    def __len__(self) -> int:
        return len(self.labels)

    # -----------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.data[idx]  # (22, T)
        y = self.labels[idx]
        meta = self.metas[idx]

        # channel-wise z-score
        if self.normalization:
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        # optional augmentation noise
        if self.is_train and self.add_noise_std > 0:
            x = x + torch.randn_like(x) * self.add_noise_std

        # reshape (22, T) -> (22, A, patch_size)
        T = x.shape[1]
        A = T // self.patch_size
        x = x[:, : A * self.patch_size].contiguous()
        x = x.view(22, A, self.patch_size)

        return {
            "inputs": x.float(),
            "labels": torch.tensor(y, dtype=torch.long),
            "meta": meta,
        }

    # -----------------------------
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
                if seg_len <= 0:
                    continue

                # events from annotations
                events, event_id = mne.events_from_annotations(raw, verbose=False)

                # iterate events
                for onset, _, code in events:
                    label_name = None
                    for k, v in event_id.items():
                        if v == code:
                            label_name = k.split("/")[-1]  # usually T1 / T2
                            break

                    if label_name not in LR_CLASS_MAP:
                        continue

                    label = LR_CLASS_MAP[label_name]

                    start = int(onset + self.t_min * sfreq)
                    stop = start + seg_len
                    if stop > raw.n_times:
                        continue

                    # (22, T)
                    segment = raw.get_data(start=start, stop=stop)
                    segment = torch.tensor(segment, dtype=torch.float32)

                    meta = {
                        "subject": subj,
                        "run": run,
                        "task_type": task_type,        # imagined / real (kept for analysis)
                        "label_name": label_name,      # T1 / T2
                        "sfreq": sfreq,
                        "ch_names": list(raw.ch_names),
                        "t_min": self.t_min,
                        "t_max": self.t_max,
                        "path": edf_path,
                    }

                    self.data.append(segment)
                    self.labels.append(int(label))
                    self.metas.append(meta)

    # -----------------------------
    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        # keep EEG channels
        raw.pick("eeg")

        # clean channel names
        raw.rename_channels({ch: clean_ch_name(ch) for ch in raw.ch_names})

        # ensure required channels exist
        missing = [ch for ch in BCI22_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise RuntimeError(f"Missing channels: {missing}")

        # pick + reorder 22 channels deterministically
        raw.pick(BCI22_CHANNELS)
        raw.reorder_channels(BCI22_CHANNELS)

        # re-reference
        if self.preproc.reref == "average":
            raw.set_eeg_reference("average", verbose=False)

        # resample
        if self.preproc.target_sfreq is not None:
            raw.resample(self.preproc.target_sfreq, verbose=False)

        # notch + bandpass
        if self.preproc.notch_hz is not None:
            raw.notch_filter(self.preproc.notch_hz, verbose=False)

        if self.preproc.l_freq is not None or self.preproc.h_freq is not None:
            raw.filter(l_freq=self.preproc.l_freq, h_freq=self.preproc.h_freq, verbose=False)

        # scale to microvolts if signals are in Volts
        if self.preproc.to_microvolts:
            data = raw.get_data()
            if np.nanmax(np.abs(data)) < 1e-3:
                raw._data = data * 1e6

        return raw