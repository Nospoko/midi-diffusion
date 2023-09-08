import torch
import numpy as np
import pretty_midi
import pandas as pd
import torch.nn as nn
import fortepyan as ff
import os
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
from models.pitch_encoder import PitchEncoder

def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int):
    ds = load_dataset(dataset_name, split="validation")
    ds = MidiDataset(ds)
    dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataloader


def get_maestro_record(query: str | int):
    ds = load_dataset("roszcz/maestro-v1-sustain", split="validation")
    quantizer = MidiQuantizer(7, 7, 7)

    if isinstance(query, int):
        record = ds[query]
    else:
        idx_query = [i for i, record in enumerate(ds) if str.lower(query) in str.lower(f"{record['composer']} {record['title']}")]
        # get first record
        idx = idx_query[0]
        record = ds[idx]

    midi_filename = f"{record['composer']} {record['title']}"
    piece = ff.MidiPiece.from_huggingface(record)

    piece.df["next_start"] = piece.df.start.shift(-1)
    piece.df["dstart"] = piece.df.next_start - piece.df.start

    piece_quantized = quantizer.quantize_piece(piece)

    sequence = {
        "filename": midi_filename,
        "pitch": torch.tensor(piece.df.pitch, dtype=torch.long)[None, :1024],
        "dstart": torch.tensor(piece.df.dstart, dtype=torch.float)[:1024],
        "duration": torch.tensor(piece.df.duration, dtype=torch.float)[:1024],
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


def eval_interpolation(
    gen: Generator,
    gen_ema: Generator,
    query: str | int,
    cfg: OmegaConf,
    conditioning_models: tuple[nn.Module] = None,
    classifier_free_guidance_scale: float = 3.0,
    interpolation_weight: float = 0.5,
):
    # loader = preprocess_dataset("JasiekKaczmarczyk/maestro-sustain-quantized", batch_size, 1)

    batch_pitch = get_maestro_record(query[0])
    batch_dynamics = get_maestro_record(query[1])

    # grab only name without extension
    filename_pitch = batch_pitch["filename"]
    filename_dynamics = batch_dynamics["filename"]

    filename = f"pitch: {filename_pitch}, dynamics: {filename_dynamics}"

    # unpack other attributes
    pitch = batch_pitch["pitch"].to(cfg.train.device)
    dstart = batch_dynamics["dstart"].numpy()
    duration = batch_dynamics["duration"].numpy()
    velocity = batch_dynamics["velocity"]

    velocity_bin = batch_dynamics["velocity_bin"].to(cfg.train.device)
    dstart_bin = batch_dynamics["dstart_bin"].to(cfg.train.device)
    duration_bin = batch_dynamics["duration_bin"].to(cfg.train.device)

    if conditioning_models is not None:
        with torch.no_grad():
            velocity_time_conditioning_model, pitch_conditioning_model = conditioning_models

            label_emb_dynamics = velocity_time_conditioning_model(velocity_bin, dstart_bin, duration_bin)
            label_emb_pitch = pitch_conditioning_model(pitch)
            label_emb = torch.lerp(label_emb_dynamics, label_emb_pitch, interpolation_weight)
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

    fake_velocity = torch.clip(fake_velocity, -1, 1)
    fake_velocity_ema = torch.clip(fake_velocity_ema, -1, 1)

    pitch = pitch[0].cpu().numpy()

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
    render_midi_to_mp3(original_midi, f"tmp/mp3/{filename}-simple-interpolation.mp3")
    render_midi_to_mp3(model_midi, f"tmp/mp3/{filename}-model.mp3")
    render_midi_to_mp3(model_ema_midi, f"tmp/mp3/{filename}-model-ema.mp3")

    # save as midi
    original_midi.write(f"tmp/midi/{filename}-simple-interpolation.midi")
    model_midi.write(f"tmp/midi/{filename}-model.midi")
    model_ema_midi.write(f"tmp/midi/{filename}-model-ema.midi")


def render_midi_to_mp3(midi_file: pretty_midi.PrettyMIDI, filepath: str) -> str:
    render_audio.midi_to_mp3(midi_file, filepath)

    return filepath


if __name__ == "__main__":
    makedir_if_not_exists("tmp/mp3")
    makedir_if_not_exists("tmp/midi")

    checkpoint = torch.load(
        hf_hub_download("JasiekKaczmarczyk/midi-diffusion", filename="midi-diffusion-2023-09-07-11-10-params-15.27M.ckpt")
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

    velocity_time_conditioning_model = (
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

    pitch_conditioning_model = (
        PitchEncoder(
            num_embeddings=cfg_cond_model.models.pitch_encoder.num_embeddings,
            embedding_dim=cfg_cond_model.models.pitch_encoder.embedding_dim,
            output_embedding_dim=cfg_cond_model.models.pitch_encoder.output_embedding_dim,
            num_attn_blocks=cfg_cond_model.models.pitch_encoder.num_attn_blocks,
            num_attn_heads=cfg_cond_model.models.pitch_encoder.num_attn_heads,
            attn_ffn_expansion=cfg_cond_model.models.pitch_encoder.attn_ffn_expansion,
            dropout_rate=cfg_cond_model.models.pitch_encoder.dropout_rate,
        )
        .eval()
        .to(cfg.train.device)
    )

    model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema_model"])
    forward_diffusion.load_state_dict(checkpoint["forward_diffusion"])
    velocity_time_conditioning_model.load_state_dict(cond_model_ckpt["velocity_time_encoder"])
    pitch_conditioning_model.load_state_dict(cond_model_ckpt["pitch_encoder"])

    conditioning_models = (velocity_time_conditioning_model, pitch_conditioning_model)

    gen = Generator(model, forward_diffusion)
    gen_ema = Generator(ema_model, forward_diffusion)

    # query = "Chopin"
    queries = ["Chopin", "Liszt"]

    eval_interpolation(gen, gen_ema, queries, cfg, conditioning_models)
