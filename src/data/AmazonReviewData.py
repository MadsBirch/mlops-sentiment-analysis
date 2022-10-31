import torch
from torch.utils.data import Dataset


class AmazonReviewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = self.df["review"].iloc[idx]
        sentiment = torch.tensor(int(self.df["sentiment"].iloc[idx]), dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        return {
            "review": review,
            "input_ids": encoding.input_ids.flatten(),
            "attention_mask": encoding.attention_mask.flatten(),
            "labels": sentiment,
        }
