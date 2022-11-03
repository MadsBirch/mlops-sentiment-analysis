import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import hydra

from src.models.model import BertSentiment

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# set device to Apple M1 GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

@hydra.train(config_name="train.yaml")


def main(cfg):
    print(cfg.hyperparameters.batch_size, cfg.hyperparameters.lr , cfg.hyperparameters.epochs)

if __name__ == "__main__":
    main

# hyper parameters
class hp:
  batch_size = 24
  lr = 1e-5
  num_epochs = 10
  
# read data files from path
data_path = 'data/processed'
train_set = torch.load(data_path+'/train_set.pth')
test_set = torch.load(data_path+'/test_set.pth')

# create data loaders
train_loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=True, num_workers=0)

# init model
model = BertSentiment(n_classes=3, dropout=0.2).to(device)

# define criterion and optimizer
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=hp.lr)

# BASIC TRAINNIG LOOP
LOSS_LIST = []

for e in range(hp.num_epochs):
  print(f'[EPOCH]: {e+1:3d}')
  
  train_loss = 0
  model.train()
  for batch in tqdm(train_loader):
    
    optimizer.zero_grad()
    
    # move data to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # get model outout and calc. loss
    output = model(input_ids, attention_mask)
    loss = loss_fn(output, labels)
    train_loss += loss.item()
    
    loss.backward()
    optimizer.step()
  
  train_loss_epoch = train_loss/len(train_loader)
  print(f'[LOSS]: {train_loss_epoch:.2f}')
  LOSS_LIST.append(train_loss_epoch)

# plot training loss
plt.plot(np.arange(1,hp.num_epochs+1, dtype=int),LOSS_LIST)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('reports/figures'+'/train_loss.png')

# save model
torch.save(model.state_dict(), 'models/model.pth')