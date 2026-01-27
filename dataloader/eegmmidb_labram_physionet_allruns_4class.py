from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import mne

# --------------------------------------------------
# 22-channel order we use (BCI-style subset)
# --------------------------------------------------
BCI22_CHANNELS: List[str] = [
    "Fz",
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "P1", "Pz", "P2",
    "Poz",
]

# --------------------------------------------------
# Map these 22 channels into LaBraM's 64-index space (for input_chans)
# NOTE: We keep the *data* as 22 channels; this mapping is for positional embedding indexing.
# --------------------------------------------------
LABRAM_64_MAP: Dict[str, int] = {
    "Fz": 2, "Fc3": 10, "Fc1": 11, "Fcz": 12, "Fc2": 13, "Fc4": 14,
    "C5": 26, "C3": 27, "C1": 28, "Cz": 29, "C2": 30, "C4": 31, "C6": 32,
    "Cp3": 41, "Cp1": 42, "Cpz": 43, "Cp2": 44, "Cp4": 45,
    "P1": 51, "Pz": 52, "P2": 53, "Poz": 59,
}

# --------------------------------------------------
# EEGMMIDB annotation labels
# T1 = left fist, T2 = right fist
# --------------------------------------------------
LR_MAP: Dict[str, int] = {"T1": 0, "T2": 1}

# --------------------------------------------------
# Run -> task type (per your convention)
# --------------------------------------------------
def run_to_task_type(run: int) -> Optional[str]:
    if run in [3, 7, 11]:
        return "real"
    if run in [4, 8, 12]:
        return "imagined"
    return None

# --------------------------------------------------
# 4-class mapping: (task_type, left/right)
# 0 = real-left, 1 = real-right, 2 = imagined-left, 3 = imagined-right
# --------------------------------------------------
def to_4class(task_type: str, lr_id: int) -> int:
    if task_type == "real":
        return 0 + lr_id
    if task_type == "imagined":
        return 2 + lr_id
    raise ValueError(f"Unknown task_type: {task_type}")

# --------------------------------------------------
# Channel name cleanup (EEGMMIDB uses dotted names sometimes)
# --------------------------------------------------
def clean_ch_name(ch: str) -> str:
    return ch.replace(".", "").capitalize()

# --------------------------------------------------
# Preprocessing configuration
# --------------------------------------------------
@dataclass(frozen=True)
class PreprocConfig:
    target_sfreq: float = 200.0
    notch_hz: Optional[float] = 50.0
    l_freq: Optional[float] = 0.1
    h_freq: Optional[float] = 75.0
    reref: Optional[str] = "average"   # "average" or None
    to_microvolts: bool = True


class EEGMMIDBLaBraMAllRuns4ClassDataset(Dataset):
    """
    PhysioNet EEGMMIDB -> LaBraM input format

    Returns:
      inputs: (22, A=4, T=200)  float32
      labels: int64 (0..3)  (real/imagined x left/right)
      meta: dict of tensors/strings (collatable by DataLoader)
    """

    def __init__(
        self,
        root_path: str,
        subjects: Optional[List[int]],
        runs: List[int],
        t_min: float = 0.0,
        t_max: float = 4.0,
        patch_size: int = 200,
        normalization: bool = True,
        is_train: bool = True,
        add_noise_std: float = 0.0,
        preproc: Optional[PreprocConfig] = None,
    ):
        self.root_path = root_path
        self.runs = list(runs)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.patch_size = int(patch_size)
        self.normalization = bool(normalization)
        self.is_train = bool(is_train)
        self.add_noise_std = float(add_noise_std)
        self.preproc = preproc or PreprocConfig()

        # discover subjects if None
        if subjects is None:
            self.subjects = self._discover_subjects()
        else:
            self.subjects = list(subjects)

        self.data: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.metas: List[Dict[str, Any]] = []

        self._load_all()

    def _discover_subjects(self) -> List[int]:
        # EEGMMIDB has S001.. etc folders
        subs = []
        if not os.path.isdir(self.root_path):
            return subs
        for name in os.listdir(self.root_path):
            if name.startswith("S") and len(name) == 4 and name[1:].isdigit():
                subs.append(int(name[1:]))
        subs.sort()
        return subs

    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw.pick("eeg")

        # clean names then pick + reorder exact BCI22 subset
        raw.rename_channels({ch: clean_ch_name(ch) for ch in raw.ch_names})
        missing = [ch for ch in BCI22_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise RuntimeError(f"Missing channels: {missing}")

        # Use legacy ordered pick to guarantee order across MNE versions
        raw.pick_channels(BCI22_CHANNELS, ordered=True)

        if self.preproc.reref == "average":
            raw.set_eeg_reference("average", verbose=False)

        if self.preproc.target_sfreq is not None:
            raw.resample(self.preproc.target_sfreq, verbose=False)

        if self.preproc.notch_hz is not None:
            raw.notch_filter(self.preproc.notch_hz, verbose=False)

        if self.preproc.l_freq is not None or self.preproc.h_freq is not None:
            raw.filter(self.preproc.l_freq, self.preproc.h_freq, verbose=False)

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

                for onset, _, code in events:
                    label_name = None
                    for k, v in event_id.items():
                        if v == code:
                            label_name = k.split("/")[-1]
                            break
                    if label_name not in LR_MAP:
                        continue

                    lr_id = LR_MAP[label_name]
                    y4 = to_4class(task_type, lr_id)

                    start = int(onset + self.t_min * sfreq)
                    stop = start + seg_len
                    if stop > raw.n_times:
                        continue

                    x = raw.get_data(start=start, stop=stop)  # (22, T)
                    x = torch.tensor(x, dtype=torch.float32)

                    # scale if signal looks like Volts
                    if self.preproc.to_microvolts:
                        if float(x.abs().max()) < 1e-3:
                            x = x * 1e6

                    # normalize per channel
                    if self.normalization:
                        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

                    # reshape to (22, A, patch_size)
                    total_T = x.shape[-1]
                    A = total_T // self.patch_size
                    if A <= 0:
                        continue
                    use_T = A * self.patch_size
                    x = x[:, :use_T].contiguous()
                    x = x.view(x.shape[0], A, self.patch_size)  # (22, A, 200)

                    # optional noise augmentation
                    if self.is_train and self.add_noise_std > 0:
                        x = x + torch.randn_like(x) * self.add_noise_std

                    self.data.append(x)
                    self.labels.append(int(y4))
                    self.metas.append(
                        {
                            "subject": subj,
                            "run": run,
                            "task_type": task_type,
                            "label_name": label_name,
                            "sfreq": sfreq,
                            "ch_names": list(raw.ch_names),
                            "t_min": self.t_min,
                            "t_max": self.t_max,
                            "path": edf_path,
                        }
                    )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "inputs": self.data[idx],  # (22, A, 200)
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "meta": self.metas[idx],
        }from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import mne

# --------------------------------------------------
# 22-channel order we use (BCI-style subset)
# --------------------------------------------------
BCI22_CHANNELS: List[str] = [
    "Fz",
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "P1", "Pz", "P2",
    "Poz",
]

# --------------------------------------------------
# Map these 22 channels into LaBraM's 64-index space (for input_chans)
# NOTE: We keep the *data* as 22 channels; this mapping is for positional embedding indexing.
# --------------------------------------------------
LABRAM_64_MAP: Dict[str, int] = {
    "Fz": 2, "Fc3": 10, "Fc1": 11, "Fcz": 12, "Fc2": 13, "Fc4": 14,
    "C5": 26, "C3": 27, "C1": 28, "Cz": 29, "C2": 30, "C4": 31, "C6": 32,
    "Cp3": 41, "Cp1": 42, "Cpz": 43, "Cp2": 44, "Cp4": 45,
    "P1": 51, "Pz": 52, "P2": 53, "Poz": 59,
}

# --------------------------------------------------
# EEGMMIDB annotation labels
# T1 = left fist, T2 = right fist
# --------------------------------------------------
LR_MAP: Dict[str, int] = {"T1": 0, "T2": 1}

# --------------------------------------------------
# Run -> task type (per your convention)
# --------------------------------------------------
def run_to_task_type(run: int) -> Optional[str]:
    if run in [3, 7, 11]:
        return "real"
    if run in [4, 8, 12]:
        return "imagined"
    return None

# --------------------------------------------------
# 4-class mapping: (task_type, left/right)
# 0 = real-left, 1 = real-right, 2 = imagined-left, 3 = imagined-right
# --------------------------------------------------
def to_4class(task_type: str, lr_id: int) -> int:
    if task_type == "real":
        return 0 + lr_id
    if task_type == "imagined":
        return 2 + lr_id
    raise ValueError(f"Unknown task_type: {task_type}")

# --------------------------------------------------
# Channel name cleanup (EEGMMIDB uses dotted names sometimes)
# --------------------------------------------------
def clean_ch_name(ch: str) -> str:
    return ch.replace(".", "").capitalize()

# --------------------------------------------------
# Preprocessing configuration
# --------------------------------------------------
@dataclass(frozen=True)
class PreprocConfig:
    target_sfreq: float = 200.0
    notch_hz: Optional[float] = 50.0
    l_freq: Optional[float] = 0.1
    h_freq: Optional[float] = 75.0
    reref: Optional[str] = "average"   # "average" or None
    to_microvolts: bool = True


class EEGMMIDBLaBraMAllRuns4ClassDataset(Dataset):
    """
    PhysioNet EEGMMIDB -> LaBraM input format

    Returns:
      inputs: (22, A=4, T=200)  float32
      labels: int64 (0..3)  (real/imagined x left/right)
      meta: dict of tensors/strings (collatable by DataLoader)
    """

    def __init__(
        self,
        root_path: str,
        subjects: Optional[List[int]],
        runs: List[int],
        t_min: float = 0.0,
        t_max: float = 4.0,
        patch_size: int = 200,
        normalization: bool = True,
        is_train: bool = True,
        add_noise_std: float = 0.0,
        preproc: Optional[PreprocConfig] = None,
    ):
        self.root_path = root_path
        self.runs = list(runs)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.patch_size = int(patch_size)
        self.normalization = bool(normalization)
        self.is_train = bool(is_train)
        self.add_noise_std = float(add_noise_std)
        self.preproc = preproc or PreprocConfig()

        # discover subjects if None
        if subjects is None:
            self.subjects = self._discover_subjects()
        else:
            self.subjects = list(subjects)

        self.data: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.metas: List[Dict[str, Any]] = []

        self._load_all()

    def _discover_subjects(self) -> List[int]:
        # EEGMMIDB has S001.. etc folders
        subs = []
        if not os.path.isdir(self.root_path):
            return subs
        for name in os.listdir(self.root_path):
            if name.startswith("S") and len(name) == 4 and name[1:].isdigit():
                subs.append(int(name[1:]))
        subs.sort()
        return subs

    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw.pick("eeg")

        # clean names then pick + reorder exact BCI22 subset
        raw.rename_channels({ch: clean_ch_name(ch) for ch in raw.ch_names})
        missing = [ch for ch in BCI22_CHANNELS if ch not in raw.ch_names]
        if missing:
            raise RuntimeError(f"Missing channels: {missing}")

        # Use legacy ordered pick to guarantee order across MNE versions
        raw.pick_channels(BCI22_CHANNELS, ordered=True)

        if self.preproc.reref == "average":
            raw.set_eeg_reference("average", verbose=False)

        if self.preproc.target_sfreq is not None:
            raw.resample(self.preproc.target_sfreq, verbose=False)

        if self.preproc.notch_hz is not None:
            raw.notch_filter(self.preproc.notch_hz, verbose=False)

        if self.preproc.l_freq is not None or self.preproc.h_freq is not None:
            raw.filter(self.preproc.l_freq, self.preproc.h_freq, verbose=False)

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

                for onset, _, code in events:
                    label_name = None
                    for k, v in event_id.items():
                        if v == code:
                            label_name = k.split("/")[-1]
                            break
                    if label_name not in LR_MAP:
                        continue

                    lr_id = LR_MAP[label_name]
                    y4 = to_4class(task_type, lr_id)

                    start = int(onset + self.t_min * sfreq)
                    stop = start + seg_len
                    if stop > raw.n_times:
                        continue

                    x = raw.get_data(start=start, stop=stop)  # (22, T)
                    x = torch.tensor(x, dtype=torch.float32)

                    # scale if signal looks like Volts
                    if self.preproc.to_microvolts:
                        if float(x.abs().max()) < 1e-3:
                            x = x * 1e6

                    # normalize per channel
                    if self.normalization:
                        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

                    # reshape to (22, A, patch_size)
                    total_T = x.shape[-1]
                    A = total_T // self.patch_size
                    if A <= 0:
                        continue
                    use_T = A * self.patch_size
                    x = x[:, :use_T].contiguous()
                    x = x.view(x.shape[0], A, self.patch_size)  # (22, A, 200)

                    # optional noise augmentation
                    if self.is_train and self.add_noise_std > 0:
                        x = x + torch.randn_like(x) * self.add_noise_std

                    self.data.append(x)
                    self.labels.append(int(y4))
                    self.metas.append(
                        {
                            "subject": subj,
                            "run": run,
                            "task_type": task_type,
                            "label_name": label_name,
                            "sfreq": sfreq,
                            "ch_names": list(raw.ch_names),
                            "t_min": self.t_min,
                            "t_max": self.t_max,
                            "path": edf_path,
                        }
                    )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "inputs": self.data[idx],  # (22, A, 200)
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "meta": self.metas[idx],
        }