import torch
import torch.nn as nn
from transformers import BertModel


class BertSentiment(nn.Module):
  def __init__(self, n_classes: int, dropout = 0.3, bert_out_dim = 768):
    super(BertSentiment, self).__init__()
    self.bert_out_dim = bert_out_dim
    self.bert = BertModel.from_pretrained("bert-base-cased")
    self.dropout = nn.Dropout(dropout)
    self.output = nn.Linear(bert_out_dim, n_classes)

  def forward(self, input_ids, attention_mask):
    output = self.bert(input_ids, attention_mask)
    pooled_output = output['pooler_output']
    x = self.dropout(pooled_output)
    out = self.output(x)
    
    return out