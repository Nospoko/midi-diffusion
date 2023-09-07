import os

import torch
import numpy as np
import pretty_midi
import torch.nn as nn
import fortepyan as ff
import pandas as pd
from omegaconf import OmegaConf
from datasets import load_dataset
from torch.utils.data import DataLoader
from fortepyan.audio import render as render_audio
from huggingface_hub.file_download import hf_hub_download

from sample import Generator
from data.dataset import MidiDataset
from data.quantizer import MidiQuantizer
from models.reverse_diffusion import Unet
from models.forward_diffusion import ForwardDiffusion
from models.velocity_time_encoder import VelocityTimeEncoder


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int):
    ds = load_dataset(dataset_name, split="validation")
    ds = MidiDataset(ds)
    dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataloader

def get_maestro_record(idx: int):
    ds = load_dataset("roszcz/maestro-v1-sustain", split="validation")
    quantizer = MidiQuantizer(7, 7, 7)

    record = ds[idx]
    midi_filename = f"{record['composer']} {record['title']}"
    piece = ff.MidiPiece.from_huggingface(record)

    piece.df["next_start"] = piece.df.start.shift(-1)
    piece.df["dstart"] = piece.df.next_start - piece.df.start

    piece_quantized = quantizer.quantize_piece(piece)

    sequence = {
        "filename": midi_filename,
        "pitch": torch.tensor(piece.df.pitch, dtype=torch.long)[None, None, :1024],
        "dstart": torch.tensor(piece.df.dstart, dtype=torch.float)[None, None, :1024],
        "duration": torch.tensor(piece.df.duration, dtype=torch.float)[None, None, :1024],
        "velocity": (torch.tensor(piece.df.velocity, dtype=torch.float)[None, None, :1024] / 64) - 1,
        "dstart_bin": torch.tensor(piece_quantized.df.dstart_bin, dtype=torch.long)[None, :1024],
        "duration_bin": torch.tensor(piece_quantized.df.duration_bin, dtype=torch.long)[None, :1024],
        "velocity_bin": torch.tensor(piece_quantized.df.velocity_bin, dtype=torch.long)[None, :1024],
    }

    return sequence



def denormalize_velocity(velocity: np.ndarray):
    return (velocity + 1) * 64


def to_midi_piece(pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray) -> ff.MidiPiece:
    record = {
        "pitch": pitch,
        "velocity": velocity,
        "dstart": dstart,
        "duration": duration,
    }

    df = pd.DataFrame(record)
    df["start"] = df.dstart.cumsum().shift(1).fillna(0)
    df["end"] = df.start + df.duration

    return ff.MidiPiece(df)
    

def compare_original_and_generated(
    gen: Generator,
    gen_ema: Generator,
    cfg: OmegaConf,
    conditioning_model: nn.Module = None,
    classifier_free_guidance_scale: float = 3.0,
    batch_size: int = 1,
):
    # loader = preprocess_dataset("JasiekKaczmarczyk/maestro-sustain-quantized", batch_size, 1)

    batch = get_maestro_record(idx=104)

    # grab only name without extension
    filename = batch["filename"]

    # unpack other attributes
    pitch = batch["pitch"][0][0].numpy()
    dstart = batch["dstart"][0][0].numpy()
    duration = batch["duration"][0][0].numpy()
    velocity = batch["velocity"]

    velocity_bin = batch["velocity_bin"].to(cfg.train.device)
    dstart_bin = batch["dstart_bin"].to(cfg.train.device)
    duration_bin = batch["duration_bin"].to(cfg.train.device)

    if conditioning_model is not None:
        with torch.no_grad():
            label_emb = conditioning_model(velocity_bin, dstart_bin, duration_bin)
    else:
        label_emb = None

    noise = torch.randn(velocity.size()).to(cfg.train.device)

    # sample velocities from standard and ema model
    fake_velocity = gen.sample(
        noise, label_emb=label_emb, intermediate_outputs=False, classifier_free_guidance_scale=classifier_free_guidance_scale
    )
    fake_velocity_ema = gen_ema.sample(
        noise, label_emb=label_emb, intermediate_outputs=False, classifier_free_guidance_scale=classifier_free_guidance_scale
    )

    velocity = denormalize_velocity(velocity[0][0].numpy())
    fake_velocity = denormalize_velocity(fake_velocity[0][0].cpu().numpy())
    fake_velocity_ema = denormalize_velocity(fake_velocity_ema[0][0].cpu().numpy())

    original_piece = to_midi_piece(pitch, dstart, duration, velocity)
    model_piece = to_midi_piece(pitch, dstart, duration, fake_velocity)
    model_ema_piece = to_midi_piece(pitch, dstart, duration, fake_velocity_ema)

    original_midi = original_piece.to_midi()
    model_midi = model_piece.to_midi()
    model_ema_midi = model_ema_piece.to_midi()

    # save as mp3
    render_midi_to_mp3(original_midi, f"tmp/mp3/{filename}-original.mp3")
    render_midi_to_mp3(model_midi, f"tmp/mp3/{filename}-model.mp3")
    render_midi_to_mp3(model_ema_midi, f"tmp/mp3/{filename}-model-ema.mp3")

    # save as midi
    original_midi.write(f"tmp/midi/{filename}-original.midi")
    model_midi.write(f"tmp/midi/{filename}-model.midi")
    model_ema_midi.write(f"tmp/midi/{filename}-model-ema.midi")


# def to_midi(pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray, track_name: str = "piano"):
#     track = pretty_midi.PrettyMIDI()
#     piano = pretty_midi.Instrument(program=0, name=track_name)

#     previous_start = 0.0

#     for p, ds, d, v in zip(pitch, dstart, duration, velocity):
#         start = previous_start + ds
#         end = start + d
#         previous_start = start

#         note = pretty_midi.Note(
#             velocity=int(v),
#             pitch=int(p),
#             start=start,
#             end=end,
#         )

#         piano.notes.append(note)

#     track.instruments.append(piano)

#     return track


def render_midi_to_mp3(midi_file: pretty_midi.PrettyMIDI, filepath: str) -> str:
    render_audio.midi_to_mp3(midi_file, filepath)

    return filepath


if __name__ == "__main__":
    checkpoint = torch.load(
        "checkpoints/midi-diffusion-2023-09-05-13-18-params-15.27M.ckpt"
        # "checkpoints/midi-diffusion-2023-09-04-22-33-params-8.601025M.ckpt"
    )

    cfg = checkpoint["config"]

    # model
    model = Unet(
        in_channels=cfg.models.unet.in_out_channels,
        out_channels=cfg.models.unet.in_out_channels,
        dim=cfg.models.unet.dim,
        dim_mults=cfg.models.unet.dim_mults,
        kernel_size=cfg.models.unet.kernel_size,
        resnet_block_groups=cfg.models.unet.num_resnet_groups,
    ).to(cfg.train.device)

    ema_model = Unet(
        in_channels=cfg.models.unet.in_out_channels,
        out_channels=cfg.models.unet.in_out_channels,
        dim=cfg.models.unet.dim,
        dim_mults=cfg.models.unet.dim_mults,
        kernel_size=cfg.models.unet.kernel_size,
        resnet_block_groups=cfg.models.unet.num_resnet_groups,
    ).to(cfg.train.device)

    # forward diffusion
    forward_diffusion = ForwardDiffusion(
        beta_start=cfg.models.forward_diffusion.beta_start,
        beta_end=cfg.models.forward_diffusion.beta_end,
        timesteps=cfg.models.forward_diffusion.timesteps,
        schedule_type=cfg.models.forward_diffusion.schedule_type,
    ).to(cfg.train.device)

    cond_model_ckpt = torch.load(hf_hub_download(repo_id=cfg.paths.cond_model_repo_id, filename=cfg.paths.cond_model_ckpt_path))
    cfg_cond_model = cond_model_ckpt["config"]

    conditioning_model = (
        VelocityTimeEncoder(
            num_embeddings=cfg_cond_model.models.velocity_time_encoder.num_embeddings,
            embedding_dim=cfg_cond_model.models.velocity_time_encoder.embedding_dim,
            output_embedding_dim=cfg_cond_model.models.velocity_time_encoder.output_embedding_dim,
            num_attn_blocks=cfg_cond_model.models.velocity_time_encoder.num_attn_blocks,
            num_attn_heads=cfg_cond_model.models.velocity_time_encoder.num_attn_heads,
            attn_ffn_expansion=cfg_cond_model.models.velocity_time_encoder.attn_ffn_expansion,
            dropout_rate=cfg_cond_model.models.velocity_time_encoder.dropout_rate,
        )
        .eval()
        .to(cfg.train.device)
    )

    model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema_model"])
    forward_diffusion.load_state_dict(checkpoint["forward_diffusion"])
    conditioning_model.load_state_dict(cond_model_ckpt["velocity_time_encoder"])

    gen = Generator(model, forward_diffusion)
    gen_ema = Generator(ema_model, forward_diffusion)

    compare_original_and_generated(gen, gen_ema, cfg, conditioning_model, batch_size=1)
