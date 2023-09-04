import os
import random

import hydra
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from omegaconf import OmegaConf
from datasets import load_dataset
from huggingface_hub import upload_file
from torch.utils.data import Subset, DataLoader
from huggingface_hub.file_download import hf_hub_download

from data.dataset import MidiDataset
from models.reverse_diffusion import Unet
from models.ema import ExponentialMovingAverage
from models.forward_diffusion import ForwardDiffusion
from models.velocity_time_encoder import VelocityTimeEncoder


def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(dataset_name: str, batch_size: int, num_workers: int, *, overfit_single_batch: bool = False):
    train_ds = load_dataset(dataset_name, split="train")
    train_ds = MidiDataset(train_ds)

    val_ds = load_dataset(dataset_name, split="validation")
    val_ds = MidiDataset(val_ds)

    test_ds = load_dataset(dataset_name, split="test")
    test_ds = MidiDataset(test_ds)

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))
        val_ds = Subset(val_ds, indices=range(batch_size))
        test_ds = Subset(test_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def forward_step(
    model: Unet,
    forward_diffusion: ForwardDiffusion,
    conditioning_model: nn.Module,
    batch: dict[str, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    x = batch["velocity"].to(device)

    velocity_bin = batch["velocity_bin"].to(device)
    dstart_bin = batch["dstart_bin"].to(device)
    duration_bin = batch["duration_bin"].to(device)

    batch_size = x.shape[0]

    # sample t
    t = torch.randint(0, forward_diffusion.timesteps, size=(batch_size,), dtype=torch.long, device=device)

    # noise batch
    x_noisy, added_noise = forward_diffusion(x, t)

    # conditional or unconditional
    if random.random() > 0.1:
        with torch.no_grad():
            label_emb = conditioning_model(velocity_bin, dstart_bin, duration_bin)
    else:
        label_emb = None

    # get predicted noise
    predicted_noise = model(x_noisy, t, label_emb)

    # get loss value for batch
    loss = F.mse_loss(predicted_noise, added_noise)

    return loss


def save_checkpoint(
    model: Unet, ema_model: Unet, forward_diffusion: ForwardDiffusion, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str
):
    # saving models
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "forward_diffusion": forward_diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        f=save_path,
    )


def upload_to_huggingface(ckpt_save_path: str, cfg: OmegaConf):
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # upload model to hugging face
    upload_file(ckpt_save_path, path_in_repo=f"{cfg.logger.run_name}.ckpt", repo_id=cfg.paths.hf_repo_id, token=token)


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)

    # dataset
    train_dataloader, val_dataloader, _ = preprocess_dataset(
        dataset_name=cfg.train.dataset_name,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # logger
    wandb.init(project="midi-diffusion", name=cfg.logger.run_name, dir=cfg.paths.log_dir)

    device = torch.device(cfg.train.device)

    # get trained conditioning model
    if cfg.paths.cond_model_repo_id is not None and cfg.paths.cond_model_ckpt_path is not None:
        cond_model_ckpt = torch.load(
            hf_hub_download(repo_id=cfg.paths.cond_model_repo_id, filename=cfg.paths.cond_model_ckpt_path)
        )
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
            .to(device)
        )

        conditioning_model.load_state_dict(cond_model_ckpt["velocity_time_encoder"])

        # freeze layers
        conditioning_model.requires_grad_(False)

    # model
    unet = Unet(
        in_channels=cfg.models.unet.in_out_channels,
        out_channels=cfg.models.unet.in_out_channels,
        dim=cfg.models.unet.dim,
        dim_mults=cfg.models.unet.dim_mults,
        label_embedding_dim=cfg_cond_model.models.velocity_time_encoder.output_embedding_dim,
        kernel_size=cfg.models.unet.kernel_size,
        resnet_block_groups=cfg.models.unet.num_resnet_groups,
    ).to(device)

    # forward diffusion
    forward_diffusion = ForwardDiffusion(
        beta_start=cfg.models.forward_diffusion.beta_start,
        beta_end=cfg.models.forward_diffusion.beta_end,
        timesteps=cfg.models.forward_diffusion.timesteps,
        schedule_type=cfg.models.forward_diffusion.schedule_type,
    ).to(device)

    # exponential moving average
    # there will be two models saved one standard and second with ema
    ema = ExponentialMovingAverage(unet, beta=0.995, warmup_steps=2000)

    # setting up optimizer
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # load checkpoint if specified in cfg
    if cfg.paths.load_ckpt_path is not None:
        checkpoint = torch.load(cfg.paths.load_ckpt_path)

        unet.load_state_dict(checkpoint["model"])
        forward_diffusion.load_state_dict(checkpoint["forward_diffusion"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # ckpt specifies directory and name of the file is name of the experiment in wandb
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}.ckpt"

    # step counts for logging to wandb
    step_count = 0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        unet.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        loss_epoch = 0.0

        for batch_idx, batch in train_loop:
            # metrics returns loss and additional metrics if specified in step function
            loss = forward_step(unet, forward_diffusion, conditioning_model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # applying exponential moving average
            ema.update(unet)

            train_loop.set_postfix(loss=loss.item())

            step_count += 1
            loss_epoch += loss.item()

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                # log metrics
                wandb.log({"train/loss": loss.item()}, step=step_count)

                # save model and optimizer states
                save_checkpoint(unet, ema.ema_model, forward_diffusion, optimizer, cfg, save_path=save_path)

        training_metrics = {"train/loss_epoch": loss_epoch / len(train_dataloader)}

        # val epoch
        unet.eval()
        val_loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False)
        loss_epoch = 0.0

        with torch.no_grad():
            for batch_idx, batch in val_loop:
                # metrics returns loss and additional metrics if specified in step function
                loss = forward_step(unet, forward_diffusion, conditioning_model, batch, device)

                val_loop.set_postfix(loss=loss.item())

                loss_epoch += loss.item()

        val_metrics = {"val/loss_epoch": loss_epoch / len(val_dataloader)}

        metrics = training_metrics | val_metrics
        wandb.log(metrics, step=step_count)

    # save model at the end of training
    save_checkpoint(unet, ema.ema_model, forward_diffusion, optimizer, cfg, save_path=save_path)

    wandb.finish()

    # upload model to huggingface if specified in cfg
    if cfg.paths.hf_repo_id is not None:
        upload_to_huggingface(save_path, cfg)


if __name__ == "__main__":
    wandb.login()

    train()