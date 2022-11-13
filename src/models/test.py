import click
import torch
import yaml

import wandb
from src.models.train_utils import (evaluate_one_epoch, get_dataloaders,
                                    get_model)

# load hyper parameters to sweep over from config file
with open('conf/conf_train.yaml') as file:
      config = yaml.load(file, Loader=yaml.FullLoader)

@click.command()
@click.argument('model_path')
def test(model_path):
       
    # load relevant hyper parameters from training run
    dropout = config['dropout']
    
    # load testset and get dataloader
    train_set = torch.load('data/processed/train_set.pth')
    test_set = torch.load('data/processed/test_set.pth')
    _, testloader = get_dataloaders(train_set, test_set, bs = 256)
    
    # load trained model from path
    model = get_model(dropout=dropout)
    model.load_state_dict(torch.load(model_path))

    # test model
    _, acc = evaluate_one_epoch(model, testloader)

    # print accuracy
    print(f'Accuracy on the test set: {acc:.2f} %')
    
if __name__ == "__main__":
    test()