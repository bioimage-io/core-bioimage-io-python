import torch


class Upscaler(torch.nn.Module):
    def forward(self, x):
        return torch.nn.Upsample(scale_factor=4, mode="nearest")(x)
