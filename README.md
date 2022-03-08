# DKN pytorch 
# Introduce 
- 
# Dataset
- use the news dataset from MIND dataset 
- download from [here](https://msnews.github.io/)
- data preprocess: python preprocess.py
```
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip

unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d test
```
# Train
- python train.py