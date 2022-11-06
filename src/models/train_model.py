import random
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models.model import BertSentiment

# set device to Apple M1 GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_dataloaders(train_set, val_set, bs):
  trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
  valloader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0)
  return trainloader, valloader

def get_model(dropout = float):
  return BertSentiment(n_classes=3, dropout=dropout).to(device)

def get_optimizer(model, lr, optimizer = 'adam'):
  if optimizer == 'adam':
    return optim.Adam(model.parameters(), lr=lr)
  
  elif optimizer == 'sgd':
    return optim.SGD(model.parameters(), momentum=0.9, lr=lr)  

def train_one_epoch(model, trainloader, optimizer, loss_fn):
  
  train_loss = 0
  n_correct = 0
  total = 0
  
  model.train()
  for batch in tqdm(trainloader):
    
    optimizer.zero_grad()
    
    # move data to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # get model outout and calc. loss
    output = model(input_ids, attention_mask)
    loss = loss_fn(output, labels)
    train_loss += loss.item()
    
    preds = torch.argmax(F.softmax(output, dim = 1),dim=1)
    n_correct += (preds == labels).sum().item()
    total += labels.size(0)
    
    loss.backward()
    optimizer.step()
  
  loss_epoch = train_loss/len(trainloader)
  acc = (n_correct/total)*100
  
  return loss_epoch, acc

def evaluate_one_epoch(model, valloader, loss_fn):
  
  test_loss = 0
  n_correct = 0
  total = 0
  
  model.eval()
  with torch.no_grad():
      for batch in tqdm(valloader):
          
          # move data to device
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
                
          output = model(input_ids, attention_mask)
          loss = loss_fn(output, labels)
          test_loss += loss.item()
          
          preds = torch.argmax(F.softmax(output, dim = 1),dim=1)
          n_correct += (preds == labels).sum().item()
          total += labels.size(0)
  
  loss_epoch = test_loss/len(valloader)
  acc = (n_correct/total)*100
  
  return loss_epoch, acc


# load hyper parameters to sweep over from config file
with open('src/models/conf/conf_train.yaml') as file:
      config = yaml.load(file, Loader=yaml.FullLoader)

# initialize wandb sweep
sweep_id = wandb.sweep(sweep=config, project='mlops_dtu')

def train():
  
  # initialize wndb
  run = wandb.init(config=config)
  
  # define hyper parameters for wandb sweep
  lr  =  wandb.config.lr
  dropout = wandb.config.dropout
  optimizer = wandb.config.optimizer
  
  # define constant hyper parametners
  bs = 256
  epochs = 5
  
  # read data files from path
  train_set = torch.load('data/processed/train_set.pth')
  val_set = torch.load('data/processed/val_set.pth')
  
  # create data loaders
  trainloader, valloader = get_dataloaders(train_set, val_set, bs = bs)

  # init model
  model = get_model(dropout=dropout)
  
  # define criterion and optimizer
  loss_fn = nn.CrossEntropyLoss().to(device)
  optimizer = get_optimizer(model, lr, optimizer=optimizer)
  
  for e in range(epochs):
    print(f'[EPOCH]: {e+1:3d}')

    train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, loss_fn)
    val_loss, val_acc = evaluate_one_epoch(model, valloader, loss_fn)
    
    wandb.log({
        'epoch': e, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': val_acc, 
        'val_loss': val_loss
      })

# Start sweep
wandb.agent(sweep_id, function=train, count=10)
  

  
  
  
  
  
  