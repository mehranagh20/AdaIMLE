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
        
        # Improved initialization with mean-specific scaling
        if any(n.startswith('final') for n,_ in self.named_parameters()):
            gain = sqrt(2.0)  # Higher gain for final layers
            nn.init.kaiming_normal_(linear.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            linear.weight.data.mul_(gain)
        
        linear.bias.data.zero_()
        
        if bias_init is not None and bias_start_dim is not None:
            linear.bias.data[bias_start_dim:] = bias_init

        self.linear = linear

    def forward(self, input):
        return self.linear(input)


class MappingNetowrk(nn.Module):
    def __init__(self, code_dim=512, rank=4):
        super().__init__()
        self.code_dim = code_dim
        self.rank = rank
        
        # Store layers properly with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                EqualLinear(code_dim, code_dim),
                nn.LeakyReLU(0.2),
                PixelNorm()
            ) for _ in range(8)
        ])
        
        # Add final processing layers
        self.final = nn.Sequential(
            EqualLinear(code_dim, code_dim * 4),
            nn.LeakyReLU(0.2),
            EqualLinear(code_dim * 4, code_dim * (2 + rank))
        )

        # Add special initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Special initialization for mean outputs
        with torch.no_grad():
            # Final layer weights for mean component
            final_layer = self.final[-1].linear
            fan_in = final_layer.weight.size(1)
            
            # Initialize mean weights with larger variance
            mean_weights = final_layer.weight[:self.code_dim]
            mean_weights.normal_(0, sqrt(2.0 / fan_in) * 5)
            
            # Initialize bias for mean outputs
            final_layer.bias[:self.code_dim].zero_()

    def forward(self, input, l1, l2):
        # Process through all layers
        x = input
        for layer in self.layers:
            x = layer(x)
        
        # Get final output
        output = self.final(x)
        
        # Split into components
        mean = output[:, :self.code_dim]
        logvar = output[:, self.code_dim:2*self.code_dim]
        low_rank_flat = output[:, 2*self.code_dim:2*self.code_dim + self.code_dim*self.rank]
        
        # Rest of the forward pass remains the same
        low_rank = low_rank_flat.view(-1, self.code_dim, self.rank)
        std_diag = torch.exp(0.5 * logvar)
        
        sample = mean + torch.bmm(low_rank, l2.unsqueeze(-1)).squeeze(-1) + std_diag * l1
        return sample


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
