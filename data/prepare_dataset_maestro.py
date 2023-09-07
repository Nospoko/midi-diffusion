import os

import fortepyan as ff
from tqdm import tqdm
from datasets import Value, Dataset, Features, Sequence, DatasetDict, load_dataset

from data.quantizer import MidiQuantizer


def process_dataset(dataset: Dataset, sequence_len: int, quantizer: MidiQuantizer) -> list[dict]:
    processed_records = []

    for record in tqdm(dataset, total=dataset.num_rows):
        # print(record)
        piece = ff.MidiPiece.from_huggingface(record)
        processed_record = process_record(piece, sequence_len, quantizer)

        processed_records += processed_record

    return processed_records


def process_record(piece: ff.MidiPiece, sequence_len: int, quantizer: MidiQuantizer) -> list[dict]:
    piece_quantized = quantizer.quantize_piece(piece)

    source = piece_quantized.source

    piece.df["next_start"] = piece.df.start.shift(-1)
    piece.df["dstart"] = piece.df.next_start - piece.df.start

    midi_filename = f"{source['composer']} {source['title']} {source['midi_filename']}"

    record = []

    for subset, subset_quantized in zip(
        piece.df.rolling(window=sequence_len, step=sequence_len),
        piece_quantized.df.rolling(window=sequence_len, step=sequence_len),
    ):
        # rolling sometimes creates subsets with shorter sequence length, they are filtered here
        if len(subset) != sequence_len:
            continue

        sequence = {
            "midi_filename": midi_filename,
            "pitch": subset.pitch.astype("int16").values.T,
            "dstart": subset.dstart.astype("float32"),
            "duration": subset.duration.astype("float32"),
            "velocity": subset.velocity.astype("int16"),
            "dstart_bin": subset_quantized.dstart_bin.astype("int8").values.T,
            "duration_bin": subset_quantized.duration_bin.astype("int8").values.T,
            "velocity_bin": subset_quantized.velocity_bin.astype("int8").values.T,
        }

        record.append(sequence)

    return record


def main():
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # hyperparameters
    sequence_len = 128

    dataset_path = "roszcz/maestro-v1-sustain"

    quantizer = MidiQuantizer(
        n_dstart_bins=7,
        n_duration_bins=7,
        n_velocity_bins=7,
    )

    train_dataset = load_dataset(dataset_path, split="train")
    val_dataset = load_dataset(dataset_path, split="validation")
    test_dataset = load_dataset(dataset_path, split="test")

    train_records = process_dataset(train_dataset, sequence_len=sequence_len, quantizer=quantizer)
    val_records = process_dataset(val_dataset, sequence_len=sequence_len, quantizer=quantizer)
    test_records = process_dataset(test_dataset, sequence_len=sequence_len, quantizer=quantizer)

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
    dataset.push_to_hub("JasiekKaczmarczyk/maestro-sustain-quantized", token=token)


if __name__ == "__main__":
    main()
