import torchvision
from text_loader import TextDataset
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from hydra.utils import instantiate
from tqdm import tqdm
from torch.distributed.pipeline.sync import Pipe

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
    transforms = torchvision.transforms.Compose([])
    dataset = TextDataset(cfg.train.text, cfg.train.length, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    ckpt_path = cfg.train.weight
    ckpt_path = ckpt_path if os.path.isfile(ckpt_path) else None
    #trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=cfg.train.max_epochs, log_every_n_steps=cfg.train.log_every_n_steps, logger=[TensorBoardLogger('./')])
    model = instantiate(cfg.model)
    devices = cfg.train.devices
    model = model(devices=devices)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))

    model = nn.Sequential(*model.module_list())
    model = Pipe(model, chunks=2)
    model.train()
    #trainer.fit(model, train_dataloaders=dataloader, ckpt_path=ckpt_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #print(f"parameters:{model.num_parameters}")
    pbar = tqdm(dataloader)
    i = 0
    for batch in pbar:
        optimizer.zero_grad()

        text, text_next = batch
        text = text.to(devices[0])
        text_next = text_next.to(devices[-1])
        text = nn.functional.one_hot(text.long(), 256).float()

        text_hat = model(text).local_value()

        loss = nn.CrossEntropyLoss()(text_hat.view(-1,256), text_next.view(-1).long())
 
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        i += 1
    torch.save(model.state_dict(), cfg.train.weight)

if __name__ == '__main__':
    main()