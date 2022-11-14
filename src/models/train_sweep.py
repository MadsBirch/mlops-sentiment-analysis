import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.train_utils import (evaluate_one_epoch, get_dataloaders,
                                    get_model, get_optimizer, train_one_epoch)

# set device to Apple M1 GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# load hyper parameters to sweep over from config file
with open('conf/conf_train_sweep.yaml') as file:
      config = yaml.load(file, Loader=yaml.FullLoader)

# initialize wandb sweep
wandb.login(key = "bd3fd38b22f78a0420f42abfc8b978d7ae49d44c")
sweep_id = wandb.sweep(sweep=config, project='mlops_dtu')

def train():
  
  # initialize wandb
  run = wandb.init(config=config)

  # read data files from path
  train_set = torch.load('data/processed/train_set.pth')
  val_set = torch.load('data/processed/val_set.pth')
  test_set = torch.load('data/processed/test_set.pth')
  
  # create data loaders
  trainloader, valloader, _ = get_dataloaders(train_set, val_set, test_set, bs = wandb.config.batch_size)

  # init model
  model = get_model(dropout=wandb.config.dropout)
  
  # define optimizer
  optimizer = get_optimizer(model, lr = wandb.config.lr, 
                            weight_decay = wandb.config.weight_decay, 
                            optimizer=wandb.config.optimizer)
  
  for e in range(wandb.config.epochs):
    print(f'[EPOCH]: {e+1:3d}')

    train_loss, train_acc = train_one_epoch(model, trainloader, optimizer)
    val_loss, val_acc = evaluate_one_epoch(model, valloader)
    
    wandb.log({
        'epoch': e, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': val_acc, 
        'val_loss': val_loss
      })

# Start sweep
wandb.agent(sweep_id, function=train, count=20)  