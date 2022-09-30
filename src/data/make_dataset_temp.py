import pandas as pd
import gzip, json

from sklearn.model_selection import train_test_split
from src.data.AmazonReviewData import AmazonReviewsDataset

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
  df_train, df_test = train_test_split(df, train_size=train_split, random_state=0)
  
  train_set = AmazonReviewsDataset(df_train, tokenizer=tokenizer, max_len=max_len)
  test_set = AmazonReviewsDataset(df_test, tokenizer=tokenizer, max_len=max_len)
    
  return train_set, test_set