import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim_ff_hidden, bias=True)
        self.linear_2 = nn.Linear(dim_ff_hidden, dim, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class SpiralConvConvBlock(nn.Module):
    def __init__(self, dim: int, dim_hidden: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.linear_in = nn.Linear(dim, dim_hidden)
        self.linear_out = nn.Linear(dim_hidden, dim)
        self.phazor = nn.Parameter(torch.randn(dim_hidden, 2))
        self.phazor_init = nn.Parameter(torch.randn(dim_hidden, 2))
        self.act = nn.SiLU()
        self.last_conv = None # (batch, dim)
        self.is_refresh = True
        self.dropout = nn.Dropout(dropout)

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dtype = x.dtype
        x = self.linear_in(x).float()
        if self.last_conv is None:
            self.last_conv = torch.randn((batch, self.dim_hidden), dtype=torch.cfloat, device=x.device)
        phazor = torch.view_as_complex(self.phazor.float())
        phazor_init = torch.view_as_complex(self.phazor_init.float())
        phazor = phazor / phazor.abs() * torch.exp(-phazor.abs())
        phazor_progression = torch.pow(phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)) # (len, dim)
        filter = phazor_progression * phazor_init.unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len*2, dim=0) # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len*2, dim=1) # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1,0,len) # (batch, len, dim)
        conv_with_past = conv_filter_x + self.last_conv.detach().unsqueeze(1)*phazor_progression.unsqueeze(0)*phazor.unsqueeze(0).unsqueeze(0)
        if self.is_refresh:
            self.last_conv = conv_with_past[:,-1,:]
        
        y = conv_with_past.real.to(dtype)
        y = self.act(y)
        y = self.linear_out(y)
        y = self.dropout(y)
        return y

    def reset_hidden(self):
        self.last_conv = None

    def randomize_init(self):
        self.last_conv = torch.randn(self.dim_hidden, dtype=torch.cfloat, device=self.linear_in.weight.device)

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class SpiralConvBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dim_sc_hidden: int, dropout: float):
        super().__init__()
        self.spiral_conv = SpiralConvConvBlock(dim, dim_sc_hidden, dropout)
        self.ffn = FFN(dim, dim_ff_hidden, dropout)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        x_ = x
        x = self.layer_norm(x)
        x = self.spiral_conv(x)
        x = x + x_

        x_ = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.spiral_conv.reset_hidden()

    def randomize_init(self):
        self.spiral_conv.randomize_init()

    def set_is_refresh(self, is_refresh):
        self.spiral_conv.set_is_refresh(is_refresh)

class SpiralConv(nn.Module):
    def __init__(self, depth: int, dim: int, dim_ff_hidden: int, dim_sc_hidden: int, dropout: float, devices):
        super().__init__()
        self.devices = devices
        self.block_list = nn.ModuleList([SpiralConvBlock(dim, dim_ff_hidden, dim_sc_hidden, dropout) for _ in range(depth)])
        for i, block in enumerate(self.block_list):
            block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (len(self.devices) * i) // len(self.block_list)

    def forward(self, x):
        for i, block in enumerate(self.block_list):
            if i > 0 and self.device_index(i) != self.device_index(i-1):
                x = x.to(self.devices[self.device_index(i)])
            x = block(x)
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
        return mlist
        
    