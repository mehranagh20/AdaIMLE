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
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        # linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = linear

    def forward(self, input):
        return self.linear(input)


class MappingNetowrk(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        f_layers = [PixelNorm()]
        s_layers = [PixelNorm()]
        for i in range(4):
            f_layers.append(EqualLinear(code_dim, code_dim))
            f_layers.append(nn.LeakyReLU(0.2))
            s_layers.append(EqualLinear(code_dim, code_dim))
            s_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(EqualLinear(code_dim, code_dim * 2))
        f_layers.append(nn.LeakyReLU(0.2))
        f_layers.append(EqualLinear(code_dim * 2, code_dim * 2))
        s_layers.append(EqualLinear(code_dim, code_dim * 2))
        s_layers.append(nn.LeakyReLU(0.2))
        s_layers.append(EqualLinear(code_dim * 2, code_dim * 2))

        self.first = nn.Sequential(*f_layers)
        self.second = nn.Sequential(*s_layers)

    def forward(
        self,
        input,
        f_latent,
        s_latent,
    ):

        out = self.first(input)
        mean, logstd = out.chunk(2, dim=1)
        std = torch.exp(logstd).clamp(max=1.0)
        print('first', torch.mean(std), torch.mean(mean), torch.mean(logstd), torch.mean(f_latent))
        f_sample = mean + std * f_latent

        out = self.second(f_sample)
        mean, logstd = out.chunk(2, dim=1)
        std = torch.exp(logstd).clamp(max=1.0)
        print('second', torch.mean(std), torch.mean(mean), torch.mean(logstd), torch.mean(s_latent))
        s_sample = mean + std * s_latent

        return s_sample

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
        return image + self.weight * spatial_noise
