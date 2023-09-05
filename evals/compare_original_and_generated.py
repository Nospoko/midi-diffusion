import os

import torch
import numpy as np
import pretty_midi
import torch.nn as nn
from omegaconf import OmegaConf
from datasets import load_dataset
from torch.utils.data import DataLoader
from fortepyan.audio import render as render_audio
from huggingface_hub.file_download import hf_hub_download

from sample import Generator
from data.dataset import MidiDataset
from models.reverse_diffusion import Unet
from models.forward_diffusion import ForwardDiffusion
from models.velocity_time_encoder import VelocityTimeEncoder


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int):
    ds = load_dataset(dataset_name, split="validation")
    ds = MidiDataset(ds)
    dataloader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataloader


def denormalize_velocity(velocity: np.ndarray):
    return (velocity + 1) * 64


def compare_original_and_generated(
    gen: Generator,
    gen_ema: Generator,
    cfg: OmegaConf,
    conditioning_model: nn.Module = None,
    classifier_free_guidance_scale: float = 3.0,
    batch_size: int = 1,
):
    loader = preprocess_dataset("JasiekKaczmarczyk/maestro-sustain-quantized", batch_size, 1)

    batch = next(iter(loader))

    # grab only name without extension
    filename = " ".join(batch["filename"])
    filename = filename.split(".")[0]

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

    render_midi_to_mp3(f"{filename}-original.mp3", pitch, dstart, duration, velocity)
    render_midi_to_mp3(f"{filename}-standard.mp3", pitch, dstart, duration, fake_velocity)
    render_midi_to_mp3(f"{filename}-ema.mp3", pitch, dstart, duration, fake_velocity_ema)


def to_midi(pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray, track_name: str = "piano"):
    track = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name=track_name)

    previous_start = 0.0

    for p, ds, d, v in zip(pitch, dstart, duration, velocity):
        start = previous_start + ds
        end = start + d
        previous_start = start

        note = pretty_midi.Note(
            velocity=int(v),
            pitch=int(p),
            start=start,
            end=end,
        )

        piano.notes.append(note)

    track.instruments.append(piano)

    return track


def render_midi_to_mp3(filename: str, pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray) -> str:
    midi_filename = os.path.basename(filename)
    mp3_path = os.path.join("tmp", midi_filename)

    if not os.path.exists(mp3_path):
        track = to_midi(pitch, dstart, duration, velocity)
        render_audio.midi_to_mp3(track, mp3_path)

    return mp3_path


if __name__ == "__main__":
    checkpoint = torch.load("checkpoints/midi-diffusion-2023-09-04-22-33-params-8.601025M.ckpt")

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
