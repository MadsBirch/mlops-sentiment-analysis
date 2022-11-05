from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import hydra 
from src.models.model import BertSentiment
from omegaconf import DictConfig

data_path = 'data/processed'

@hydra.main(config_path="conf", config_name="conf_predict")
def main(cfg: DictConfig, display = True):
    # load testset and define dataloader
    test_set = torch.load(data_path+'/test_set.pth')
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size , num_workers=0)

    # load trained model from path
    model = BertSentiment(n_classes= cfg.n_classes, dropout=cfg.dropout)
    model.load_state_dict(torch.load('models/model.pth'))

    # test function
    model.eval().to('cpu')

    test_loss = 0
    n_correct = 0
    total = 0

    loss_fn = nn.CrossEntropyLoss()

    TEST_ACC = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            
            review = batch['review']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            
            preds = torch.argmax(F.softmax(outputs, dim = 1),dim=1)
            n_correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = (n_correct/total)*100

    if display:
        print(f'Accuracy on the test set: {acc:.2f} %')
    return

if __name__ == "__main__":
    main()
