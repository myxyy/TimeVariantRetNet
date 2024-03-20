import torch
import torch.nn as nn
import numpy as np

class FFN(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: float, dtype):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim_ff_hidden, bias=True, dtype=dtype)
        nn.init.normal_(self.linear_1.weight, std=dim**-0.5)
        nn.init.constant_(self.linear_1.bias, 0)
        self.linear_2 = nn.Linear(dim_ff_hidden, dim, bias=True, dtype=dtype)
        nn.init.normal_(self.linear_2.weight, std=dim_ff_hidden**-0.5)
        nn.init.constant_(self.linear_2.bias, 0)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x

class SioConv(nn.Module):
    def __init__(self, dim: int, dtype):
        super().__init__()
        self.dim = dim
        self.linear_a = nn.Linear(dim, dim*2, bias=False)

        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))
        self.is_refresh = True

    #(batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dim = self.dim
        dtype = x.dtype

        if self.last_hidden is None:
            self.last_hidden = self.last_hidden_init.unsqueeze(0).expand(batch, dim)
        else:
            self.last_hidden = self.last_hidden.detach()

        a = torch.view_as_complex(self.linear_a(x).float().view(batch, len, dim, 2)) # (batch, len, dim)
        a_sqr_mag = a.real * a.real + a.imag * a.imag
        a = a * torch.rsqrt(a_sqr_mag) * torch.exp(-a_sqr_mag)
        a_ln = torch.log(a)
        a_ln_tri = a_ln.transpose(2,1).unsqueeze(2).expand(batch, dim, len, len).triu() # (batch, dim, len, len)
        a_ln_tri_fft = torch.fft.fft(a_ln_tri, n=len*2, dim=3)
        ones_fft = torch.fft.fft(torch.ones(len, len, device=x.device), n=len*2, dim=1)
        a_ln_tri_conv = torch.fft.ifft(a_ln_tri_fft * ones_fft.unsqueeze(0).unsqueeze(1)).narrow(3,0,len) # (batch, dim, len, len)
        c = torch.exp(a_ln_tri_conv).triu(diagonal=-1) # (batch, dim, len, len)

        x = x.cfloat().transpose(2,1) # (batch, dim, len)
        x_mat = x.roll(1, dims=2).unsqueeze(2).repeat(1, 1, len, 1)
        x_mat[:,:,:,0] = self.last_hidden.unsqueeze(2)
        h = (x_mat * c).sum(3) # (batch, dim, len)
        h[:,:,-1] += x[:,:,-1]
        h = h.transpose(2,1) # (batch, len, dim)
        if self.is_refresh:
            self.last_hidden = h[:,-1,:]
        return h.real.to(dtype)

    def reset_hidden(self):
        self.last_hidden = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh




class SConv(nn.Module):
    def __init__(self, dim: int, dtype):
        super().__init__()
        self.dim = dim
        self.phazor_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))
        self.phazor = nn.Parameter(torch.exp(2.0j * np.pi * torch.arange(dim) / dim) * torch.abs(torch.randn(dim)))
        self.last_conv = None # (batch, dim)
        self.last_conv_init = nn.Parameter(torch.randn(dim, dtype=torch.cfloat))
        self.is_refresh = True

    # (batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dtype = x.dtype

        x = x.to(torch.cfloat)
        if self.last_conv is None:
            self.last_conv = self.last_conv_init.unsqueeze(0).expand(batch, self.dim)
        else:
            self.last_conv = self.last_conv.detach()
        phazor = self.phazor
        phazor = torch.exp(-phazor.real*phazor.real-phazor.imag*phazor.imag) * torch.exp(1.0j * phazor.angle())
        phazor_progression = torch.pow(phazor.unsqueeze(0), torch.arange(len, device=x.device).unsqueeze(1)) # (len, dim)
        filter = phazor_progression * self.phazor_init.unsqueeze(0)
        filter_fft = torch.fft.fft(filter, n=len*2, dim=0) # (len*2, dim)
        x_fft = torch.fft.fft(x, n=len*2, dim=1) # (batch, len*2, dim)
        conv_filter_x = torch.fft.ifft(filter_fft.unsqueeze(0) * x_fft, dim=1).narrow(1,0,len) # (batch, len, dim)
        conv_with_past = conv_filter_x + self.last_conv.unsqueeze(1)*phazor_progression.unsqueeze(0)*phazor.unsqueeze(0).unsqueeze(0)
        if self.is_refresh:
            self.last_conv = conv_with_past[:,-1,:]

        y = conv_with_past
        y = y.real.to(dtype)
        return y

    def reset_hidden(self):
        self.last_conv = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class CConv(nn.Module):
    def __init__(self, dim: int, clen: int, dtype):
        super().__init__()
        self.dim = dim
        self.clen = clen
        self.last_input_init = nn.Parameter(torch.randn((clen, dim), dtype=torch.float))
        self.filter= nn.Parameter(torch.randn((clen, dim), dtype=torch.float) / dim)
        self.last_input = None
        self.is_refresh = True

    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dtype = x.dtype

        x = x.to(torch.float)

        if self.last_input is None:
            self.last_input = self.last_input_init.unsqueeze(0).expand(batch, self.clen, self.dim)
        else:
            self.last_input = self.last_input.detach()
        
        x_with_last = torch.cat((self.last_input, x), dim=1)
        fft_x_with_last = torch.fft.rfft(x_with_last, n=(self.clen+len)*2, dim=1)
        fft_filter = torch.fft.rfft(self.filter, n=(self.clen+len)*2, dim=0)
        conv_x_with_last_filter = torch.fft.irfft(fft_x_with_last * fft_filter.unsqueeze(0), dim=1).narrow(1,self.clen,len)
        if self.is_refresh:
            self.last_input = x_with_last.narrow(1,len,self.clen)
        return conv_x_with_last_filter.to(dtype)

    def reset_hidden(self):
        self.last_input = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh


class SioConvNetBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, dim_time, dropout: float, dtype):
        super().__init__()
        self.dtype = dtype 
        self.spiral_conv = SioConv(dim_time, dtype)
        self.linear_time_in = nn.Linear(dim, dim_time)
        self.linear_time_out = nn.Linear(dim_time, dim)
        self.ffn_sc = FFN(dim, dim_ff_hidden, dtype)
        self.layer_norm_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.layer_norm_ffn_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.act = nn.SiLU()
        self.fc = nn.Linear(dim, dim_time, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.layer_norm_sc_in(x)
        y = self.fc(x)
        y = self.act(y)
        x = self.spiral_conv(self.linear_time_in(x))
        x = x * y
        x = self.linear_time_out(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm_ffn_sc_in(x)
        x = self.ffn_sc(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.spiral_conv.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.spiral_conv.set_is_refresh(is_refresh)

class SioConvNet(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        dim_time: int,
        dropout: float,
        vocab_size: int,
        devices,
        dtype=torch.float,
        token_in_out_parameter_corr = 3.0,
    ):
        super().__init__()
        self.devices = devices
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.token_in = nn.Linear(vocab_size, dim, device=devices[0], dtype=dtype)
        nn.init.normal_(self.token_in.weight, std=vocab_size**-0.5)
        nn.init.constant_(self.token_in.bias, 0)
        self.token_out = nn.Linear(dim, vocab_size, device=devices[-1], dtype=dtype)
        nn.init.normal_(self.token_out.weight, std=dim**-0.5)
        nn.init.constant_(self.token_out.bias, 0)
        self.block_list = nn.ModuleList([SioConvNetBlock(dim, dim_ff_hidden, dim_time, dropout, dtype) for _ in range(depth)])
        self.layer_norm_last = nn.LayerNorm(dim, elementwise_affine=True, bias=True, device=devices[-1], dtype=dtype)

        self.token_in_out_parameter_corr = token_in_out_parameter_corr
        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_layer_norm_last = sum(p.numel() for p in self.layer_norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters_corr = (self.num_parameters_per_block * depth) + self.num_parameters_layer_norm_last + (self.num_parameters_token_in + self.num_parameters_token_out) * self.token_in_out_parameter_corr

        for i, block in enumerate(self.block_list):
            self.block_list[i] = block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (int)((len(self.devices) * (i * self.num_parameters_per_block + self.num_parameters_token_in * self.token_in_out_parameter_corr)) / self.num_parameters_corr)

    def forward(self, x):
        x = self.token_in(x)
        for i, block in enumerate(self.block_list):
            x = x.to(self.devices[self.device_index(i)])
            x = block(x)
        x = x.to(self.devices[-1])
        x = self.layer_norm_last(x)
        x = self.token_out(x)
        return x 

    def reset_hidden(self):
        for block in self.block_list:
            block.reset_hidden()

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
        mlist[-1] = nn.Sequential(mlist[-1], self.layer_norm_last, self.token_out)
        return mlist
        
    