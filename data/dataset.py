import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


class MidiDataset(Dataset):
    def __init__(self, dataset: HFDataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.dataset[index]

        # wrap sequence with Tensor
        pitch = torch.tensor(sequence["pitch"], dtype=torch.long)
        dstart = torch.tensor(sequence["dstart"], dtype=torch.float32)
        duration = torch.tensor(sequence["duration"], dtype=torch.float32)
        # normalized velocity between [-1, 1]
        normalized_velocity = (torch.tensor(sequence["velocity"], dtype=torch.float32) / 64) - 1
        dstart_bin = torch.tensor(sequence["dstart_bin"], dtype=torch.long)
        duration_bin = torch.tensor(sequence["duration_bin"], dtype=torch.long)
        velocity_bin = torch.tensor(sequence["velocity_bin"], dtype=torch.long)

        record = {
            "filename": sequence["midi_filename"],
            "pitch": pitch[None, :],
            "dstart": dstart[None, :],
            "duration": duration[None, :],
            "velocity": normalized_velocity[None, :],
            "dstart_bin": dstart_bin,
            "duration_bin": duration_bin,
            "velocity_bin": velocity_bin,
        }

        return record
