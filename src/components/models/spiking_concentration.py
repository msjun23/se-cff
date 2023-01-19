import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate, utils

from einops import rearrange

class ConvSNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False, 
                 beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=0.75)):
        super(ConvSNN, self).__init__()
        # self.block = nn.Sequential(*[
        #     nn.Conv2d(in_channels=in_channels, 
        #               out_channels=out_channels, 
        #               kernel_size=kernel_size, 
        #               padding=padding, 
        #               bias=bias), 
        #     nn.BatchNorm2d(out_channels), 
        #     snn.Leaky(beta=beta, 
        #               spike_grad=spike_grad, 
        #               init_hidden=True, 
        #               output=True)
        # ])
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              padding=padding, 
                              bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.lif  = snn.Leaky(beta=beta, 
                              spike_grad=spike_grad, 
                              init_hidden=False, 
                              output=False)
        self.mem = self.lif.init_leaky()
        
    def forward(self, x):
        # mem = self.lif.init_leaky()
        # utils.reset(self.block)
        
        out = self.conv(x)
        out = self.bn(out)
        spk, self.mem = self.lif(out, self.mem)
        # spk, mem = self.block(x)
        
        return spk, self.mem
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(UpBlock, self).__init__()
        self.csnn1 = ConvSNN(in_channels=in_channels, 
                             out_channels=out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding)
        self.csnn2 = ConvSNN(in_channels=2*out_channels, 
                             out_channels=out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding)
        self.csnn3 = ConvSNN(in_channels=out_channels, 
                             out_channels=out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding)
        
    def forward(self, x, past):
        out = F.interpolate(x, scale_factor=(2,2))
        spk, mem = self.csnn1(out)
        spk, mem = self.csnn2(torch.cat([mem, past], dim=1))
        spk, mem = self.csnn3(mem)
        
        return mem
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DownBlock, self).__init__()
        self.csnn1 = ConvSNN(in_channels=in_channels, 
                             out_channels=out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding)
        self.csnn2 = ConvSNN(in_channels=out_channels, 
                             out_channels=out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding)
        self.csnn3 = ConvSNN(in_channels=out_channels, 
                             out_channels=out_channels, 
                             kernel_size=kernel_size, 
                             padding=padding)
        
    def forward(self, x):
        out = F.avg_pool2d(x, (2,2))
        spk, mem = self.csnn1(out)
        spk, mem = self.csnn2(mem)
        spk, mem = self.csnn3(mem)
        
        return mem

class SpikingConcentrationNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, attention_method='soft'):
        super(SpikingConcentrationNet, self).__init__()
        
        b1_channels = 2 * base_channels
        b2_channels = 2 * b1_channels
        b3_channels = 2 * b2_channels
        
        self.attention_method = attention_method
        
        self.csnn1 = ConvSNN(in_channels=in_channels, out_channels=base_channels, kernel_size=(3,3), padding=(1,1))
        self.csnn2 = ConvSNN(in_channels=base_channels, out_channels=base_channels, kernel_size=(3,3), padding=(1,1))
        
        self.down1 = DownBlock(in_channels=base_channels, out_channels=b1_channels, kernel_size=(3,3), padding=(1,1))
        self.down2 = DownBlock(in_channels=b1_channels, out_channels=b2_channels, kernel_size=(3,3), padding=(1,1))
        self.down3 = DownBlock(in_channels=b2_channels, out_channels=b3_channels, kernel_size=(3,3), padding=(1,1))
        
        self.up1   = UpBlock(in_channels=b3_channels, out_channels=b2_channels, kernel_size=(3,3), padding=(1,1))
        self.up2   = UpBlock(in_channels=b2_channels, out_channels=b1_channels, kernel_size=(3,3), padding=(1,1))
        self.up3   = UpBlock(in_channels=b1_channels, out_channels=base_channels, kernel_size=(3,3), padding=(1,1))
        
        self.last_csnn = ConvSNN(in_channels=base_channels, out_channels=in_channels, kernel_size=(3,3), padding=(1,1))
        
    def forward(self, x):
        # x = x.transpose(0, 1)       # [timestep, batch_size, h, w]
        # x = x.unsqueeze(dim=2)      # [timestep, batch_size, 1, h, w], 1=in_channels(c)
        time_step = x.size(0)       # timestep
        rec = []
        
        for step in range(time_step):
            spk1, mem1 = self.csnn1(x[step])    # x[step]: [batch_size, 1, h, w]
            spk2, mem2 = self.csnn2(mem1)
            
            b1 = self.down1(mem2)
            b2 = self.down2(b1)
            b3 = self.down3(b2)
            
            out = self.up1(b3, b2)
            out = self.up2(out, b1)
            out = self.up3(out, mem2)
            
            spk_out, mem_out = self.last_csnn(out)  # [batch_size, 1, h, w]
            
            rec.append(mem_out)                     # [[batch_size, 1, h, w] * step]
            
        out = torch.cat(rec, dim=1) # [batch_size, timestep, h, w]
        
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
    