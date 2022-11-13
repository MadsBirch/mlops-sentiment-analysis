import os
import torch
import pytest
import torch.utils.data as data

"""
We will now test the dataset splits are of correct length.

There is a total of 20,473 samples in the dataset.

We split the dataset into train and test with a split of 80/20, which leaves us with 
16378 samples in the trainset and 4095 in the test set.

The training set is further split into training set and a validation set with a 90/10 split,
which leaves us with 14740 samples in the train set and 1638 samples in the validation set.

"""
N_train = 14740
N_test = 4095
N_val = 1638

@pytest.mark.skipif(not os.path.exists('data/processed/train_set.pth'), reason="Data files not found")
def test_data_splits_and_sample_shape():
    train_set = torch.load('data/processed/train_set.pth')
    val_set = torch.load('data/processed/val_set.pth')
    test_set = torch.load('data/processed/test_set.pth')
    
    dataloader = data.DataLoader(train_set, batch_size=1)
    out_dict = next(iter(dataloader))
    input_ids = out_dict['input_ids']
    
    assert len(train_set) == N_train and len(test_set) == N_test and len(val_set) == N_val, "Dataset splits did not have expected lengths"
    assert input_ids.shape[1] == 24, "Max length is not value"

if __name__ == "__main__":
    test_data_splits_and_sample_shape()