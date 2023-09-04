import copy

import torch.nn as nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, beta: float = 0.9999, warmup_steps: int = 2000):
        super().__init__()

        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval().requires_grad_(False)

        self.beta = beta
        self.warmup_steps = warmup_steps
        self.step = 0

    def update_average(self, new, old):
        if old is None:
            return new

        return old * self.beta + (1 - self.beta) * new

    def update_model_average(self, model: nn.Module):
        for model_params, ema_params in zip(model.parameters(), self.ema_model.parameters()):
            new_weight, old_weight = model_params.data, ema_params.data

            ema_params.data = self.update_average(new_weight, old_weight)

    def reset_parameters(self, model: nn.Module):
        self.ema_model.load_state_dict(model.state_dict())

    def update(self, model: nn.Module):
        if self.step < self.warmup_steps:
            self.reset_parameters(model)
        else:
            self.update_model_average(model)

        self.step += 1
