from hyena import Hyena

import pytorch_lightning as pl
from timm.models.layers import trunc_normal_
import torchvision.transforms as transforms
import torch
from torchmetrics import MeanMetric
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

class Lang(pl.LightningModule):
    logger: TensorBoardLogger
    def __init__(self, model, len, depth=4, dropout=0.1, vocab_size=256, dim=256, dim_pos=256, enable_profiling=False, batch_size=16):
        super().__init__()
        self.save_hyperparameters()
        self.enable_profiling=enable_profiling
        self.len = len
        self.vocab_size = vocab_size
        self.hyena = model(len, dim, depth, dim_pos, dropout)
        self.token_in = nn.Linear(vocab_size, dim)
        self.token_out = nn.Linear(dim, vocab_size)
        self.batch_size = batch_size
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.num_parameters**-0.5)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def training_step(self, batch, batch_idx):
        data, next = batch
        x = nn.functional.one_hot(data.long(), self.vocab_size).float()
        x_next = next
        if self.enable_profiling:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ]
            ) as p:
                x_hat = self.token_out(self.hyena(self.token_in(x)))
            print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        else:
            x_hat = self.token_out(self.hyena(self.token_in(x)))
        loss = nn.CrossEntropyLoss()(x_hat.view(-1,self.vocab_size), x_next.view(-1).long())
        self.log("train_loss", loss, on_epoch=False, prog_bar=True)
        return loss

    def forward(self, x):
        x = nn.functional.one_hot(x.long(), self.vocab_size).float()
        x_hat = self.token_out(self.hyena(self.token_in(x)))
        x_hat = x_hat.softmax(2)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

model = Lang(
    Hyena,
    1024,
    dim=256,
    depth=16,
    batch_size=16,
)