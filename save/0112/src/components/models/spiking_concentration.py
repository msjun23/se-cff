import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import snntorch as snn
from snntorch import surrogate, utils
from snntorch import functional as SF

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.1),
        ])

    def forward(self, x):
        return self.block(x)


class SpikingBlock(nn.Module):
    def __init__(self, beta, spike_grad):
        super(SpikingBlock, self).__init__()
        self.spiking = nn.Sequential(*[
            snn.Leaky(beta=beta, 
                      spike_grad=spike_grad)
        ])
        
    def forward(self, x):
        return self.spiking(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(UpBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = ConvBlock(in_channels=2 * out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = ConvBlock(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)

    def forward(self, x, other):
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.conv1(x)
        x = self.conv2(torch.cat([x, other], 1))
        x = self.conv3(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = ConvBlock(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = ConvBlock(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)

    def forward(self, x):
        x = F.avg_pool2d(x, (2, 2))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class SpikingConcentrationNet(nn.Module):
    def __init__(self, in_channels, base_channels=32, attention_method='hard'):
        super(SpikingConcentrationNet, self).__init__()
        # Neuron and simulation params
        self.beta = 0.5
        self.spike_grad = surrogate.fast_sigmoid(slope=75)
        
        self.attention_method = attention_method
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=base_channels,
                               kernel_size=(3, 3),
                               padding=(1, 1))
        self.conv2 = ConvBlock(in_channels=base_channels,
                               out_channels=base_channels,
                               kernel_size=(3, 3),
                               padding=(1, 1))
        self.down1 = DownBlock(base_channels, base_channels * 2, (3, 3), (1, 1))
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, (3, 3), (1, 1))
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, (3, 3), (1, 1))
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, (3, 3), (1, 1))
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, (3, 3), (1, 1))
        self.up3 = UpBlock(base_channels * 2, base_channels, (3, 3), (1, 1))
        self.last_conv = nn.Conv2d(in_channels=base_channels,
                                   out_channels=in_channels,
                                   kernel_size=(3, 3),
                                   padding=(1, 1))
        
        # self.lif1 = SpikingBlock(beta=self.beta, spike_grad=self.spike_grad)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [time_step batch_size C H W]
        time_step = x.size(0)           # time_step = 10
        rec = []
        # mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()
        
        for step in range(time_step):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            
            c1 = self.conv1(x[step])    # x[step]: [batch_size C H W]
            spk1, mem1 = self.lif1(c1, mem1)
            c2 = self.conv2(mem1)
            spk2, mem2 = self.lif2(c2, mem2)
            
            d1 = self.down1(mem2)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            
            out = self.up1(d3, d2)
            out = self.up2(out, d1)
            out = self.up3(out, mem2)
            out = self.last_conv(out)
            
            rec.append(out)
            
        out = torch.cat(rec, dim=1)

        x = rearrange(x, 't b c h w -> b (c t) h w')
        if self.attention_method == 'hard':
            hard_attention = out.max(dim=1)[1]

            new_x = x[
                torch.arange(x.size(0), device='cuda').view(x.size(0), 1, 1, 1),
                torch.stack([hard_attention] * x.size(1), dim=1),
                torch.arange(x.size(2), device='cuda').view(1, 1, x.size(2), 1),
                torch.arange(x.size(3), device='cuda').view(1, 1, 1, x.size(3)),
            ]
            new_x = new_x.squeeze(dim=4).contiguous()
        elif self.attention_method == 'soft':
            soft_attention = F.softmax(out, dim=1)
            new_x = x * soft_attention
            new_x = new_x.sum(dim=1, keepdim=True).contiguous()
        else:
            raise NotImplementedError

        return new_x

