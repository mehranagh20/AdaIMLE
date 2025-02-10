from math import sqrt

import torch
from torch import nn


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias_init=None, bias_start_dim=None):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        # linear.weight.data.normal_()
        linear.bias.data.zero_()
        
        if bias_init is not None and bias_start_dim is not None:
            linear.bias.data[bias_start_dim:] = bias_init

        self.linear = linear

    def forward(self, input):
        return self.linear(input)


class MappingNetowrk(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        # Change constants to trainable parameters
        self.min_logvar = nn.Parameter(torch.tensor(-1.0))
        self.max_logvar = nn.Parameter(torch.tensor(2.0))

        f_layers = []
        s_layers = []
        for i in range(5):
            f_layers.append(EqualLinear(code_dim, code_dim))
            f_layers.append(nn.LeakyReLU(0.2))
            s_layers.append(EqualLinear(code_dim, code_dim))
            s_layers.append(nn.LeakyReLU(0.2))
        
        f_layers.append(EqualLinear(code_dim, code_dim * 2))
        f_layers.append(EqualLinear(code_dim * 2, code_dim * 2))
        f_layers.append(EqualLinear(code_dim * 2, code_dim * 2, bias_init=-0.5, bias_start_dim=code_dim))

        s_layers.append(EqualLinear(code_dim, code_dim * 2))
        s_layers.append(EqualLinear(code_dim * 2, code_dim * 2))
        s_layers.append(EqualLinear(code_dim * 2, code_dim * 2, bias_init=-0.5, bias_start_dim=code_dim))

        self.f_style = nn.Sequential(*f_layers)
        self.s_style = nn.Sequential(*s_layers)

    def forward(
        self,
        input,
        l1,
        l2,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):
        mean, logvar = self.f_style(input).chunk(2, dim=1)
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        std = torch.exp(logvar)
        s1 = mean + l1 * std

        mean, logvar = self.s_style(s1).chunk(2, dim=1)
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        std = torch.exp(logvar)
        s2 = mean + l2 * std

        return s2

    # def mean_style(self, input):

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = input
        if input.shape[3] > 1:
            out = self.norm(input)
        out = gamma * out + beta
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, channel, 1, 1))

    def forward(self, image, spatial_noise):
        # Convert spatial noise to half precision and move to GPU
        if spatial_noise.device != self.weight.device:
            spatial_noise = spatial_noise.half().to(self.weight.device, non_blocking=True)
        return image + (self.weight * spatial_noise).to(image.dtype)
