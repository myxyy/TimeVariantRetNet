# TimeVariantRetNet

## Overview
https://qiita.com/3405691582/items/c6fa00e58181b6bb6ca5

## Usage
Adjust model parameters by editing `configs/model/retnet.yaml`.
Before train or predict, specify cuda device by editing `devices` parameter of `configs/train/default.yaml` and `configs/predict/default.yaml`.
### Train
```
poetry run python src/train.py train.text=text/foo.txt train.weight=weight/bar.ckpt
```
### Predict
```
poetry run python src/predict.py predict.weight=weight/bar.ckpt
```