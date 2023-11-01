import torchvision
from text_loader import TextDataset
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from hydra.utils import instantiate
from tqdm import tqdm

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    transforms = torchvision.transforms.Compose([])
    dataset = TextDataset(cfg.train.text, cfg.train.length, transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    ckpt_path = cfg.train.weight
    ckpt_path = ckpt_path if os.path.isfile(ckpt_path) else None
    #trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=cfg.train.max_epochs, log_every_n_steps=cfg.train.log_every_n_steps, logger=[TensorBoardLogger('./')])
    model = instantiate(cfg.model)
    model = model(devices=cfg.train.devices)
    #trainer.fit(model, train_dataloaders=dataloader, ckpt_path=ckpt_path)
    optimizer = model.configure_optimizers()
    print(f"parameters:{model.num_parameters}")
    pbar = tqdm(dataloader)
    i = 0
    for batch in pbar:
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        i += 1
    torch.save(model.state_dict(), cfg.train.weight)

if __name__ == '__main__':
    main()