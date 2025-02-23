import torch
import torch.nn as nn

class MeanShift(nn.Module):
    def __init__(self, pixel_range, n_channels, rgb_mean=None, rgb_std=None, sign=-1):
        super(MeanShift, self).__init__()

        if rgb_mean is None and rgb_std is None:
            if n_channels == 1:
                rgb_mean = [0.5]
                rgb_std =[1.0]
            elif n_channels == 3:
                # rgb_mean = (0.4488, 0.4371, 0.4040)
                rgb_mean = (0.5, 0.5, 0.5)
                rgb_std = (1.0, 1.0, 1.0)

        self.shifter = nn.Conv2d(n_channels, n_channels, 1, 1, 0)
        std = torch.Tensor(rgb_std)
        self.shifter.weight.data = torch.eye(n_channels).view(n_channels, n_channels, 1, 1) / std.view(n_channels, 1, 1, 1)
        self.shifter.bias.data = sign * pixel_range * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x
