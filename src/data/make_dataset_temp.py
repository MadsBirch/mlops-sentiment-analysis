import pandas as pd
import gzip, json

from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from src.data.AmazonReviewData import AmazonReviewsDataset

raw_data_path = "data/raw/"

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)


def sentiment_map(x):
  if x < 3:
    return 0
  
  elif x > 3:
    return 2
  
  else:
    return 1


def get_pandas_DF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
 
  return pd.DataFrame.from_dict(df, orient='index')


def preprocess_data(raw_data_path, tokenizer, max_len = 256, train_split = 0.7):
  
  # get pandas df
  df = get_pandas_DF(raw_data_path+'reviews_Automotive_5.json.gz')
  
  # subset columns and rename to more intuitive names 
  df = df[['overall', 'reviewText']]
  df = df.rename(columns={'overall': 'sentiment', 'reviewText': 'review'})
  
  # do sentiment mapping
  df.sentiment = df.sentiment.apply(sentiment_map)
  
  # split into train test set
  train_df, test_df = train_test_split(df, train_size=train_split, random_state=0)
  
  train_set = AmazonReviewsDataset(train_df, tokenizer=tokenizer, max_len=max_len)
  test_set = AmazonReviewsDataset(test_df, tokenizer=tokenizer, max_len=max_len)
    
  return train_df, test_df, train_set, test_set


def get_dataloaders(batch_size = 32, num_workers = 0):
  _, _, train_set, test_set = preprocess_data(raw_data_path=raw_data_path, tokenizer = tokenizer)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  valid_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
  
  return train_loader, valid_loader