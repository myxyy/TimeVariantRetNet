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

class SioConvLayer(nn.Module):
    def __init__(self, dim: int, inner_dim: int, dtype):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim 
        self.linear_a = nn.Linear(dim, inner_dim*2, bias=False)
        self.linear_x = nn.Linear(dim, inner_dim*2, bias=False)
        self.mat_v = nn.Parameter(torch.randn(inner_dim, inner_dim, dtype=torch.cfloat))
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

        self.last_hidden = None
        self.last_hidden_init = nn.Parameter(torch.randn(inner_dim, dtype=torch.cfloat))
        self.is_refresh = True

    #(batch, len, dim) -> (batch, len, dim)
    def forward(self, x):
        batch = x.shape[0]
        len = x.shape[1]
        dim = self.dim
        inner_dim = self.inner_dim
        dtype = x.dtype

        if self.last_hidden is None:
            self.last_hidden = self.last_hidden_init.unsqueeze(0).expand(batch, inner_dim)
        else:
            self.last_hidden = self.last_hidden.detach()

        a = torch.view_as_complex(self.linear_a(x.float()).view(batch, len, inner_dim, 2)) # (batch, len, inner_dim)
        a_sqr_mag = a.real * a.real + a.imag * a.imag
        a = a * torch.rsqrt(a_sqr_mag) * torch.sigmoid(torch.log(a_sqr_mag))

        x = torch.view_as_complex(self.linear_x(x.float()).view(batch, len, inner_dim, 2))
        x_sqr_mag = x.real * x.real + x.imag * x.imag
        x = x * torch.rsqrt(x_sqr_mag) * x_sqr_mag
        x = x.view(batch, len, inner_dim)

        if len == 1:
            vav_inv = torch.matmul(self.mat_v.unsqueeze(0).unsqueeze(1) * a.unsqueeze(3), torch.inverse(self.mat_v))
            h = (torch.matmul(vav_inv, self.last_hidden.unsqueeze(1).unsqueeze(3)).squeeze(3) + x)
            if self.is_refresh:
                self.last_hidden = h[:,-1,:]
            return self.linear_out(h.real).to(dtype)

        a_ln = torch.log(a)
        a_ln_tri = a_ln.transpose(2,1).unsqueeze(2).expand(batch, inner_dim, len, len).triu() # (batch, inner_dim, len, len)
        a_ln_tri_fft = torch.fft.fft(a_ln_tri, n=len*2, dim=3)
        ones_fft = torch.fft.fft(torch.ones(len, len, device=x.device), n=len*2, dim=1)
        a_ln_tri_conv = torch.fft.ifft(a_ln_tri_fft * ones_fft.unsqueeze(0).unsqueeze(1)).narrow(3,0,len) # (batch, inner_dim, len, len)
        c = torch.exp(a_ln_tri_conv).triu(diagonal=-1) # (batch, inner_dim, len, len)
        mat_v_unsqueeze = self.mat_v.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        c = torch.matmul(mat_v_unsqueeze * c.permute(0,2,3,1).unsqueeze(3), torch.inverse(mat_v_unsqueeze)) # (batch, len, len, inner_dim, inner_dim)

        x = x.transpose(2,1) # (batch, inner_dim, len)
        x_roll = x.roll(1, dims=2) # (batch, inner_dim, len)
        x_roll[:,:,0] = self.last_hidden
        h = torch.einsum("blmno,bol->bmn", c, x_roll).transpose(2,1) # (btach, inner_dim, len)
        h[:,:,-1] += x[:,:,-1]
        h = h.transpose(2,1).reshape(batch, len, inner_dim) # (batch, len, inner_dim)
        if self.is_refresh:
            self.last_hidden = h[:,-1,:]
        return self.linear_out(h.real).to(dtype)

    def reset_hidden(self):
        self.last_hidden = None

    def set_is_refresh(self, is_refresh):
        self.is_refresh = is_refresh

class SioConvBlock(nn.Module):
    def __init__(self, dim: int, dim_ff_hidden: int, inner_dim: int, dropout: float, dtype):
        super().__init__()
        self.dtype = dtype 

        self.layer_norm_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.sioconv = SioConvLayer(dim, inner_dim, dtype)

        self.layer_norm_ffn_sc_in = nn.LayerNorm(dim, elementwise_affine=True, bias=True, dtype=dtype)
        self.ffn_sc = FFN(dim, dim_ff_hidden, dtype)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_ = x
        x = self.layer_norm_sc_in(x)
        x = self.sioconv(x)
        x = self.dropout(x)
        x = x + x_

        x_ = x
        x = self.layer_norm_ffn_sc_in(x)
        x = self.ffn_sc(x)
        x = self.dropout(x)
        x = x + x_

        return x

    def reset_hidden(self):
        self.sioconv.reset_hidden()

    def set_is_refresh(self, is_refresh):
        self.sioconv.set_is_refresh(is_refresh)

class SioConv(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        dim_ff_hidden: int,
        inner_dim: int,
        dropout: float,
        vocab_size: int,
        devices,
        dtype=torch.float,
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
        self.block_list = nn.ModuleList([SioConvBlock(dim, dim_ff_hidden, inner_dim, dropout, dtype) for _ in range(depth)])
        self.layer_norm_last = nn.LayerNorm(dim, elementwise_affine=True, bias=True, device=devices[-1], dtype=dtype)

        self.num_parameters_token_in = sum(p.numel() for p in self.token_in.parameters())
        self.num_parameters_per_block = sum(p.numel() for p in self.block_list[0].parameters())
        self.num_parameters_layer_norm_last = sum(p.numel() for p in self.layer_norm_last.parameters())
        self.num_parameters_token_out = sum(p.numel() for p in self.token_out.parameters())
        self.num_parameters = (self.num_parameters_per_block * depth) + self.num_parameters_layer_norm_last + (self.num_parameters_token_in + self.num_parameters_token_out)

        for i, block in enumerate(self.block_list):
            self.block_list[i] = block.to(devices[self.device_index(i)])

    def device_index(self, i):
        return (int)((len(self.devices) * (i * self.num_parameters_per_block + self.num_parameters_token_in)) / self.num_parameters)

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
        
    