import os
import argparse

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import fortepyan as ff
from omegaconf import OmegaConf
from datasets import load_dataset
from huggingface_hub.file_download import hf_hub_download
from tqdm import tqdm

from sample import Generator
from data.quantizer import MidiQuantizer
from models.reverse_diffusion import Unet
from models.forward_diffusion import ForwardDiffusion
from models.velocity_time_encoder import VelocityTimeEncoder


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def process_record(record: dict, seq_len: int = 1024):
    quantizer = MidiQuantizer(7, 7, 7)

    midi_filename = f"{record['composer']} {record['title']}"
    piece = ff.MidiPiece.from_huggingface(record)

    piece.df["next_start"] = piece.df.start.shift(-1)
    piece.df["dstart"] = piece.df.next_start - piece.df.start

    piece_quantized = quantizer.quantize_piece(piece)

    sequence = {
        "filename": midi_filename,
        "pitch": torch.tensor(piece.df.pitch, dtype=torch.long)[:seq_len],
        "dstart": torch.tensor(piece.df.dstart, dtype=torch.float)[:seq_len],
        "duration": torch.tensor(piece.df.duration, dtype=torch.float)[:seq_len],
        "velocity": (torch.tensor(piece.df.velocity, dtype=torch.float)[None, None, :seq_len] / 64) - 1,
        "dstart_bin": torch.tensor(piece_quantized.df.dstart_bin, dtype=torch.long)[None, :seq_len],
        "duration_bin": torch.tensor(piece_quantized.df.duration_bin, dtype=torch.long)[None, :seq_len],
        "velocity_bin": torch.tensor(piece_quantized.df.velocity_bin, dtype=torch.long)[None, :seq_len],
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


def process_original_and_generated(
    gen: Generator,
    gen_ema: Generator,
    record: dict,
    cfg: OmegaConf,
    save_dir: str,
    conditioning_model: nn.Module = None,
    classifier_free_guidance_scale: float = 3.0,
):
    batch = process_record(record, seq_len=1024)

    # grab only name without extension
    filename = batch["filename"]

    # unpack other attributes
    pitch = batch["pitch"].numpy()
    dstart = batch["dstart"].numpy()
    duration = batch["duration"].numpy()
    velocity = batch["velocity"]

    velocity_bin = batch["velocity_bin"].to(cfg.train.device)
    dstart_bin = batch["dstart_bin"].to(cfg.train.device)
    duration_bin = batch["duration_bin"].to(cfg.train.device)

    if conditioning_model is not None:
        with torch.no_grad():
            conditioning_embedding = conditioning_model(velocity_bin, dstart_bin, duration_bin)
    else:
        conditioning_embedding = None

    noise = torch.randn(velocity.size()).to(cfg.train.device)

    # sample velocities from standard and ema model
    fake_velocity = gen.sample(
        noise,
        conditioning_embeding=conditioning_embedding,
        intermediate_outputs=False,
        classifier_free_guidance_scale=classifier_free_guidance_scale,
    )
    fake_velocity_ema = gen_ema.sample(
        noise,
        conditioning_embeding=conditioning_embedding,
        intermediate_outputs=False,
        classifier_free_guidance_scale=classifier_free_guidance_scale,
    )

    fake_velocity = torch.clip(fake_velocity, -1, 1)
    fake_velocity_ema = torch.clip(fake_velocity_ema, -1, 1)

    velocity = denormalize_velocity(velocity[0][0].numpy())
    fake_velocity = denormalize_velocity(fake_velocity[0][0].cpu().numpy())
    fake_velocity_ema = denormalize_velocity(fake_velocity_ema[0][0].cpu().numpy())

    original_piece = to_midi_piece(pitch, dstart, duration, velocity)
    model_piece = to_midi_piece(pitch, dstart, duration, fake_velocity)
    model_ema_piece = to_midi_piece(pitch, dstart, duration, fake_velocity_ema)

    original_midi = original_piece.to_midi()
    model_midi = model_piece.to_midi()
    model_ema_midi = model_ema_piece.to_midi()

    # save as midi
    original_midi.write(f"{save_dir}/{filename}-original.midi")
    model_midi.write(f"{save_dir}/{filename}-model.midi")
    model_ema_midi.write(f"{save_dir}/{filename}-model-ema.midi")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="tmp/midi")
    args = parser.parse_args()

    makedir_if_not_exists(args.save_dir)

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

    dataset = load_dataset("roszcz/maestro-v1-sustain", split="validation")

    for record in tqdm(dataset, total=dataset.num_rows):
        process_original_and_generated(
            gen=gen, 
            gen_ema=gen_ema, 
            record=record, 
            cfg=cfg, 
            save_dir=args.save_dir, 
            conditioning_model=conditioning_model
        )


if __name__ == "__main__":
    main()