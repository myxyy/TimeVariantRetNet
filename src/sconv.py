import torch
import torch.nn as nn
import numpy as np

class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float, dropout: float, dtype, scale: float):
        super().__init__()
        scale = 1e-2
        self.linear_1 = nn.Linear(dim, dim_ff_hidden, bias=True, dtype=dtype)
        nn.init.normal_(self.linear_1.weight, std=dim**-0.5*scale)
        nn.init.constant_(self.linear_1.bias, 0)
        self.linear_2 = nn.Linear(dim_ff_hidden, dim, bias=True, dtype=dtype)
        nn.init.normal_(self.linear_2.weight, std=dim_ff_hidden**-0.5*scale)
        nn.init.constant_(self.linear_2.bias, 0)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class SConv(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, dropout: float, scale: float):
        super().__init__()
        scale = 1e-2
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.linear_in = nn.Linear(dim, dim_hidden, dtype=torch.cfloat, bias=False)
        nn.init.normal_(self.linear_in.weight, std=dim**-0.5*scale)
        self.linear_out = nn.Linear(dim_hidden, dim, dtype=torch.cfloat, bias=False)
        nn.init.normal_(self.linear_out.weight, std=dim_hidden**-0.5*scale)
        #self.phazor = nn.Parameter(torch.randn(dim_hidden, dtype=torch.cfloat))
        self.phazor = nn.Parameter(torch.exp(torch.rand(dim_hidden) * np.pi * 2.0j) * (1-torch.rand(dim_hidden)*1e-3))
        #self.phazor_init = nn.Parameter(torch.randn(dim, 2))
        #self.angle = nn.Parameter(torch.randn(dim))
        #self.angle_init = nn.Parameter(torch.randn(dim))
        #self.act = nn.SiLU()
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(dim_hidden, dtype=torch.cfloat))
        self.dropout = nn.Dropout(dropout)
        self.is_refresh = True

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        #x = x.to(torch.cfloat)
        #print(f'testtesttest:{x.dtype}')
        #print(f'testtesttest:{self.linear_in.weight.dtype}')
        x = self.linear_in(x)
        if self.last_conv is None:
            self.last_conv = self.last_conv_init.unsqueeze(0).expand(batch, self.dim_hidden)
        else:
            self.last_conv = self.last_conv.detach()
        #angle = self.angle.float() * np.pi
        #phazor = torch.view_as_complex(torch.stack((torch.cos(angle), torch.sin(angle)), dim=1))
        #angle_init = self.angle_init.float() * np.pi
        #phazor_init = torch.view_as_complex(torch.stack((torch.cos(angle_init), torch.sin(angle_init)), dim=1))
        phazor = self.phazor
        #phazor_init = self.phazor_init
        #phazor = phazor / phazor.abs() * torch.exp(-phazor.abs())
        phazor = phazor / torch.clamp(phazor.abs(), min=1.0)
        #phazor = phazor / (1 + 1e-2 * (torch.arange(self.dim_hidden, device=x.device) + 1)/self.dim)
        phazor_progression = torch.pow(phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)) # (len, dim)
        filter = phazor_progression# * phazor_init.unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len*2, dim=0) # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len*2, dim=1) # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1,0,len) # (batch, len, dim)
        conv_with_past = conv_filter_x + self.last_conv.unsqueeze(1)*phazor_progression.unsqueeze(0)*phazor.unsqueeze(0).unsqueeze(0)
        if self.is_refresh:
            self.last_conv = conv_with_past[:,-1,:]
        
        y = self.linear_out(conv_with_past).real
        y = self.dropout(y)
        return y

    def reset_hidden(self):
        self.last_conv = None

    def randomize_init(self):
        self.last_conv = torch.randn(self.dim_hidden, dtype=torch.cfloat, device=self.linear_in.weight.device)

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class SConvNetBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dim_sc_hidden: int, dropout: float, dtype, scale: float):
        super().__init__()
        self.dtype = dtype
        self.spiral_conv = SConv(dim, dim_sc_hidden, dropout, scale)
        self.ffn = FFN(dim, dim_ff_hidden, dropout, dtype, scale)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False, dtype=dtype)

    def forward(self, x):
        x_ = x
        x = self.layer_norm(x)
        x = x.to(torch.cfloat)
        x = self.spiral_conv(x)
        x = x.to(self.dtype)
        x = self.layer_norm(x)
        #x = x + x_

        #x_ = x
        x = self.ffn(x)
        #x = self.layer_norm(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.spiral_conv.reset_hidden()

    def randomize_init(self):
        self.spiral_conv.randomize_init()

    def set_is_refresh(self, is_refresh):
        self.spiral_conv.set_is_refresh(is_refresh)

class SConvNet(nn.Module):
    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dim_sc_hidden: int, dropout: float, vocab_size: int, dtype, devices):
        super().__init__()
        self.devices = devices
        self.vocab_size = vocab_size
        self.token_in = nn.Linear(vocab_size, dim, device=devices[0], dtype=dtype)
        nn.init.normal_(self.token_in.weight, std=vocab_size**-0.5)
        nn.init.constant_(self.token_in.bias, 0)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1], dtype=dtype)
        nn.init.normal_(self.token_out.weight, std=dim**-0.5)
        nn.init.constant_(self.token_out.bias, 0)
        self.block_list = nn.ModuleList([SConvNetBlock(dim, dim_ff_hidden, dim_sc_hidden, dropout, dtype, 1.0) for i in range(depth)])
        for i, block in enumerate(self.block_list):
            block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (len(self.devices) * i) // len(self.block_list)

    def forward(self, x):
        x = self.token_in(x)
        for i, block in enumerate(self.block_list):
            if i > 0 and self.device_index(i) != self.device_index(i-1):
                x = x.to(self.devices[self.device_index(i)])
            x = block(x)
        x = self.token_out(x)
        return x 

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

    def randomize_init(self):
        for block in self.block_list:
            block.randomize_init()

    def set_is_refresh(self, is_refresh):
        for block in self.block_list:
            block.set_is_refresh(is_refresh)

    def module_list(self):
        blistlist = []
        for _ in self.devices:
            blistlist.append([])
        for i, block in enumerate(self.block_list):
            blistlist[self.device_index(i)].append(block)
        mlist = []
        for blist in blistlist:
            mlist.append(nn.Sequential(*blist))
        mlist[0] = nn.Sequential(self.token_in, mlist[0])
        mlist[-1] = nn.Sequential(mlist[-1], self.token_out)
        return mlist
        
    