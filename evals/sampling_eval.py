import torch
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from huggingface_hub.file_download import hf_hub_download

from sample import Generator
from train import preprocess_dataset
from models.reverse_diffusion import Unet
from models.forward_diffusion import ForwardDiffusion
from models.velocity_time_encoder import VelocityTimeEncoder


def eval_generation(
    gen: Generator,
    cfg: OmegaConf,
    conditioning_model: nn.Module = None,
    classifier_free_guidance_scale: float = 3.0,
):
    batch_size = 2
    loader, _, _ = preprocess_dataset("JasiekKaczmarczyk/giant-midi-sustain-quantized", batch_size, 1, overfit_single_batch=True)

    batch = next(iter(loader))

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

    # if intermediate_outputs=True returns dict of intermediate signals else only denoised signal
    fake_velocity = gen.sample(
        noise, label_emb=label_emb, intermediate_outputs=True, classifier_free_guidance_scale=classifier_free_guidance_scale
    )

    # idx for plotted intermediate image
    idx_to_plot = [255, 127, 63, 10, 0]
    plotted_velocities = [fake_velocity[i].cpu() for i in idx_to_plot]
    plotted_velocities += [velocity]

    # plot fake
    fig, axes = plt.subplots(batch_size, len(plotted_velocities), figsize=(20, 10))
    titles = [f"Timestep: {t}" for t in idx_to_plot] + ["Original"]

    # iterate over batches
    for i, ax_rows in enumerate(axes):
        # iterate over timesteps
        for j, ax in enumerate(ax_rows):
            v = plotted_velocities[j][i][0]

            sns.lineplot(v, ax=ax)
            ax.set_title(titles[j])
            ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    checkpoint = torch.load("checkpoints/overfit-single-batch-2023-09-04-17-12.ckpt")

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

    eval_generation(gen, cfg, conditioning_model, classifier_free_guidance_scale=3.0)

    eval_generation(gen, cfg, classifier_free_guidance_scale=0)

    # eval_generation(gen_ema, conditioning_model, cfg)
