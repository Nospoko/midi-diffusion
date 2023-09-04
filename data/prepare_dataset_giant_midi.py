import os
import math

import fortepyan as ff
from tqdm import tqdm
from datasets import Value, Dataset, Features, Sequence, DatasetDict, load_dataset

from data.quantizer import MidiQuantizer


def process_record(piece: ff.MidiPiece, sequence_len: int, quantizer: MidiQuantizer) -> list[dict]:
    piece_quantized = quantizer.quantize_piece(piece)

    midi_filename = piece_quantized.source["midi_filename"]

    record = []

    for subset in piece_quantized.df.rolling(window=sequence_len, step=sequence_len):
        # rolling sometimes creates subsets with shorter sequence length, they are filtered here
        if len(subset) != sequence_len:
            continue

        sequence = {
            "midi_filename": midi_filename,
            "pitch": subset.pitch.astype("int16").values.T,
            "dstart": subset.dstart.astype("float32"),
            "duration": subset.duration.astype("float32"),
            "velocity": subset.velocity.astype("int16"),
            "dstart_bin": subset.dstart_bin.astype("int8").values.T,
            "duration_bin": subset.duration_bin.astype("int8").values.T,
            "velocity_bin": subset.velocity_bin.astype("int8").values.T,
        }

        record.append(sequence)

    return record


def split_in_two(filenames: list[str], split_ratio: float = 0.8) -> tuple[list[str], list[str]]:
    ds_length = len(filenames)

    split = math.ceil(split_ratio * ds_length)

    return filenames[:split], filenames[split:]


def main():
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # hyperparameters
    sequence_len = 128

    hf_dataset_path = "roszcz/giant-midi-sustain"
    dataset = load_dataset(hf_dataset_path, split="train")

    quantizer = MidiQuantizer(
        n_dstart_bins=7,
        n_duration_bins=7,
        n_velocity_bins=7,
    )

    # get name of file, this will be used for split
    filenames = dataset["midi_filename"]

    # creating split for train, val, test
    train_filenames, val_test_filenames = split_in_two(filenames, split_ratio=0.8)
    val_filenames, test_filenames = split_in_two(val_test_filenames, split_ratio=0.5)

    train_records = []
    val_records = []
    test_records = []

    for record in tqdm(dataset, total=dataset.num_rows):
        midi_filename = record["midi_filename"]

        piece = ff.MidiPiece.from_huggingface(record)
        records = process_record(piece, sequence_len, quantizer)

        if midi_filename in train_filenames:
            train_records += records
        elif midi_filename in val_filenames:
            val_records += records
        elif midi_filename in test_filenames:
            test_records += records

    # building huggingface dataset
    features = Features(
        {
            "midi_filename": Value(dtype="string"),
            "pitch": Sequence(feature=Value(dtype="int16"), length=sequence_len),
            "dstart": Sequence(feature=Value(dtype="float32"), length=sequence_len),
            "duration": Sequence(feature=Value(dtype="float32"), length=sequence_len),
            "velocity": Sequence(feature=Value(dtype="int16"), length=sequence_len),
            "dstart_bin": Sequence(feature=Value(dtype="int8"), length=sequence_len),
            "duration_bin": Sequence(feature=Value(dtype="int8"), length=sequence_len),
            "velocity_bin": Sequence(feature=Value(dtype="int8"), length=sequence_len),
        }
    )

    # dataset = Dataset.from_list(records, features=features)
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_records, features=features),
            "validation": Dataset.from_list(val_records, features=features),
            "test": Dataset.from_list(test_records, features=features),
        }
    )

    # print(dataset["train"])
    dataset.push_to_hub("JasiekKaczmarczyk/giant-midi-sustain-quantized", token=token)


if __name__ == "__main__":
    main()
