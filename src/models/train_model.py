import random
import wandb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model import BertSentiment
# from model import BertSentiment

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# set device to Apple M1 GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# set up wandb sweep
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'test_acc'
               },
    'parameters': {
        'dropout': {'values': [0.1, 0.2, 0.3]},
        'lr': {'values': [1e-3, 1e-4, 1e-5]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project='mlops_dtu')

# read data files from path
data_path = "data/processed"
train_set = torch.load(data_path + "/train_set.pth")
test_set = torch.load(data_path + "/test_set.pth")


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

def evaluate_one_epoch(model, testloader, loss_fn):
  
  test_loss = 0
  n_correct = 0
  total = 0
  
  model.eval()
  with torch.no_grad():
      for batch in tqdm(testloader):
          
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
  
  loss_epoch = test_loss/len(testloader)
  acc = (n_correct/total)*100
  
  return loss_epoch, acc

def main():
  run = wandb.init()
  
  # define hyper parameters
  lr  =  wandb.config.lr
  dropout = wandb.config.dropout
  bs = 256
  epochs = 5
  
  # create data loaders
  trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
  testloader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=0)

  # init model
  model = BertSentiment(n_classes=3, dropout=dropout).to(device)

  # define criterion and optimizer
  loss_fn = nn.CrossEntropyLoss().to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  
  for e in range(epochs):
    print(f'[EPOCH]: {e+1:3d}')

    train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, loss_fn)
    test_loss, test_acc = evaluate_one_epoch(model, testloader, loss_fn)
    
    wandb.log({
        'epoch': e, 
        'train_acc': train_acc,
        'train_loss': train_loss, 
        'val_acc': test_acc, 
        'val_loss': test_loss
      })
    
# Start sweep job.
wandb.agent(sweep_id, function=main, count=4)