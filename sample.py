import torch
from tqdm import tqdm

from models.reverse_diffusion import Unet
from models.forward_diffusion import ForwardDiffusion


class Generator:
    def __init__(self, model: Unet, forward_diffusion: ForwardDiffusion):
        self.model = model.eval()
        self.forward_diffusion = forward_diffusion.eval()
        self.timesteps = forward_diffusion.timesteps

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        conditioning_embeding: torch.Tensor = None,
        timesteps: int = None,
        intermediate_outputs: bool = False,
        classifier_free_guidance_scale: int = 3,
    ) -> torch.Tensor | dict[torch.Tensor]:
        outputs = {}

        # specify diffusion timesteps
        if timesteps is None:
            timesteps = self.forward_diffusion.timesteps

        # reversing diffusion process
        for i in tqdm(reversed(range(timesteps)), total=timesteps, leave=False):
            # generating timestep tensor of size (batch_size, )
            t = torch.ones(x.shape[0], device=x.device, dtype=torch.long) * i

            # predict noise
            predicted_noise = self.model(x, t, conditioning_embeding)

            # classifier free guidance
            if conditioning_embeding is not None:
                uncond_predicted_noise = self.model(x, t, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, classifier_free_guidance_scale)

            # get params diffusion params for timestep
            betas_t = self.forward_diffusion.betas[t][:, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.forward_diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None]
            sqrt_recip_alphas_t = self.forward_diffusion.sqrt_recip_alphas[t][:, None, None]
            posterior_variance_t = self.forward_diffusion.posterior_variance[t][:, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
                + torch.sqrt(posterior_variance_t) * noise
            )

            # append intermediate results
            if intermediate_outputs:
                outputs[i] = x

        return outputs if len(outputs) > 0 else x
