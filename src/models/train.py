import wandb
import yaml

import torch
import torch.nn as nn

from src.models.train_utils import get_dataloaders, get_model, get_optimizer, train_one_epoch, evaluate_one_epoch

# set device to Apple M1 GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# load hyper parameters to sweep over from config file
with open('conf/conf_train.yaml') as file:
      config = yaml.load(file, Loader=yaml.FullLoader)
      
# initialize wandb
wandb.init(config = config, project="mlops_dtu")

def train():
  
  print(f'[INFO] Training model with the parameters: \n {config}')
  
  # declare hyper parameters for wandb sweep
  lr  =  wandb.config.lr
  dropout = wandb.config.dropout
  optimizer = wandb.config.optimizer
  bs = wandb.config.batch_size
  epochs = wandb.config.epochs
  
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
  
  # training loop
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
  
  # save final model
  torch.save(model.state_dict(), 'models/final_model.pth')

if __name__ == "__main__":
    train()