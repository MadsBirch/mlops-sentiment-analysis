# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import gzip, json, os, urllib

import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from src.data.AmazonReviewData import AmazonReviewsDataset


# define a pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# load zipped json file
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

# create pandas DF
def get_pandas_DF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
 
  return pd.DataFrame.from_dict(df, orient='index')

# map rating into sentiment scores
def sentiment_map(x):
  if x < 3:
    return 0
  
  elif x > 3:
    return 2
  
  else:
    return 1


def preprocess_data(raw_data_path, max_len = 24, train_split = 0.7, val_split = 0.1):
  
  # get pandas df
  df = get_pandas_DF(raw_data_path+'/reviews_Automotive_5.json.gz')
  
  # subset columns and rename to more intuitive names 
  df = df[['overall', 'reviewText']]
  df = df.rename(columns={'overall': 'sentiment', 'reviewText': 'review'})
  
  # do sentiment mapping
  df.sentiment = df.sentiment.apply(sentiment_map)
  
  # split into train, validation and test set
  train_df, test_df = train_test_split(df, train_size=train_split, random_state=0)
  train_df, val_df = train_test_split(train_df, train_size=0.9, random_state=0)
  
  train_set = AmazonReviewsDataset(train_df, tokenizer=tokenizer, max_len=max_len)
  val_set = AmazonReviewsDataset(val_df, tokenizer=tokenizer, max_len=max_len)
  test_set = AmazonReviewsDataset(test_df, tokenizer=tokenizer, max_len=max_len)
    
  return train_set, val_set, test_set


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # download dataset file if not in folder
    url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz'
    name = url.split("/")[-1] 
    filename = os.path.join(input_filepath, name)
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    
    # get train and test set from input path -> save to output path
    train_set, val_set, test_set = preprocess_data(input_filepath)
    torch.save(train_set, output_filepath+'/train_set.pth')
    torch.save(train_set, output_filepath+'/val_set.pth')
    torch.save(test_set, output_filepath+'/test_set.pth')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
