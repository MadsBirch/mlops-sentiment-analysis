import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model import BertSentiment

# set device to Apple M1 GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss().to(device)


def get_dataloaders(train_set, val_set, test_set, bs: int):
    trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    valloader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=0)
    testloader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=0)
    return trainloader, valloader, testloader


def get_model(dropout=0.3):
    return BertSentiment(n_classes=3, dropout=dropout).to(device)


def get_optimizer(model, lr: float, weight_decay: float, optimizer=str):
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer == "sgd":
        return optim.SGD(
            model.parameters(), momentum=0.9, lr=lr, weight_decay=weight_decay
        )

    else:
        raise ValueError("Illegal optimizer! Specify optimizer as 'sgd' or 'adam'")


def train_one_epoch(model, trainloader, optimizer):

    train_loss = 0
    n_correct = 0
    total = 0

    model.train()
    for batch in tqdm(trainloader):

        optimizer.zero_grad()

        # move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # get model outout and calc. loss
        output = model(input_ids, attention_mask)
        loss = loss_fn(output, labels)
        train_loss += loss.item()

        preds = torch.argmax(F.softmax(output, dim=1), dim=1)
        n_correct += (preds == labels).sum().item()
        total += labels.size(0)

        loss.backward()
        optimizer.step()

    loss_epoch = train_loss / len(trainloader)
    acc = (n_correct / total) * 100

    return loss_epoch, acc


def evaluate_one_epoch(model, valloader):

    test_loss = 0
    n_correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(valloader):

            # move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, attention_mask)
            loss = loss_fn(output, labels)
            test_loss += loss.item()

            preds = torch.argmax(F.softmax(output, dim=1), dim=1)
            n_correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss_epoch = test_loss / len(valloader)
    acc = (n_correct / total) * 100

    return loss_epoch, acc
