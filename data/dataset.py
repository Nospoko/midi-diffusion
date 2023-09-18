import random

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from data.quantizer import MidiQuantizer


def change_speed(dstart: np.ndarray, duration: np.ndarray, factor: float = None) -> tuple[np.ndarray, np.ndarray]:
    if not factor:
        slow = 0.8
        change_range = 0.4
        factor = slow + random.random() * change_range

    dstart /= factor
    duration /= factor
    return dstart, duration


def pitch_shift(pitch: np.ndarray, shift_threshold: int = 5) -> np.ndarray:
    # No more than given number of steps
    PITCH_LOW = 21
    PITCH_HI = 108
    low_shift = -min(shift_threshold, pitch.min() - PITCH_LOW)
    high_shift = min(shift_threshold, PITCH_HI - pitch.max())

    if low_shift > high_shift:
        shift = 0
    else:
        shift = random.randint(low_shift, high_shift + 1)
    pitch += shift

    return pitch


class MidiDataset(Dataset):
    def __init__(self, dataset: HFDataset, augmentation_percentage: float = 0.0):
        super().__init__()

        self.dataset = dataset
        self.augmentation_percentage = augmentation_percentage
        self.quantizer = MidiQuantizer(7, 7, 7)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.dataset[index]

        pitch = np.array(sequence["pitch"])
        dstart = np.array(sequence["dstart"])
        duration = np.array(sequence["duration"])
        dstart_bin = np.array(sequence["dstart_bin"])
        duration_bin = np.array(sequence["duration_bin"])
        # normalizing velocity between [-1, 1]
        velocity = (np.array(sequence["velocity"]) / 64) - 1
        velocity_bin = np.array(sequence["velocity_bin"])

        if np.any(np.isnan(dstart)):
            dstart = np.nan_to_num(dstart)

        # shift pitch augmentation
        if random.random() < self.augmentation_percentage:
            # max shift is octave down or up
            shift = random.randint(1, 12)
            pitch = pitch_shift(pitch, shift)

        # change tempo augmentation
        if random.random() < self.augmentation_percentage:
            dstart, duration = change_speed(dstart, duration)
            # change bins for new dstart and duration values
            dstart_bin = np.digitize(dstart, self.quantizer.dstart_bin_edges) - 1
            duration_bin = np.digitize(duration, self.quantizer.duration_bin_edges) - 1

        record = {
            "filename": sequence["midi_filename"],
            "pitch": torch.tensor(pitch[None, :], dtype=torch.long),
            "dstart": torch.tensor(dstart[None, :], dtype=torch.float),
            "duration": torch.tensor(duration[None, :], dtype=torch.float),
            "velocity": torch.tensor(velocity[None, :], dtype=torch.float),
            "dstart_bin": torch.tensor(dstart_bin, dtype=torch.long),
            "duration_bin": torch.tensor(duration_bin, dtype=torch.long),
            "velocity_bin": torch.tensor(velocity_bin, dtype=torch.long),
        }

        return record
