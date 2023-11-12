import re
import contractions
import torch
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, DebertaTokenizerFast, AutoTokenizer
import numpy as np

# Apply the necessary preprocessing steps on the text 
def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    # Replace '&amp;' with '&'
    # Remove trailing whitespace
    # Remove words that contain only digits
    # Remove contractions, example: I'll --> I will
    text = re.sub('RT '+r'(@.*?)[\s]', '', text)
    text = re.sub(r'(@.*?)[\s]', '', text)
    text = re.sub(r'#','',text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    #text = re.sub(r'\b\d+\b','', text)
    text = contractions.fix(text)
    return text

# Use the appropriate tokenizer for the chosen bert model 
# and apply the tokenizer to each text of the dataset. 
def preprocessing_for_bert(data,tokenizer):
  input_ids = []
  attention_masks = []

  encoded_texts = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
  MAX_LEN = max([len(text) for text in encoded_texts])

  for sent in data:
      encoded_sent = tokenizer.encode_plus(
          text=sent,
          add_special_tokens=True,
          max_length=MAX_LEN,
          padding='max_length',
          return_attention_mask=True
          )
      input_ids.append(encoded_sent.get('input_ids'))
      attention_masks.append(encoded_sent.get('attention_mask'))

  input_ids = torch.tensor(input_ids)
  attention_masks = torch.tensor(attention_masks)

  return input_ids, attention_masks


# Choose the correct tokenizer based on the specific bert model 
def choose_tokenizer(MODEL):
  if MODEL in ['roberta-base','roberta-large']:
    tokenizer = RobertaTokenizer.from_pretrained(MODEL)
  elif MODEL=='albert-base-v2':
    tokenizer = AlbertTokenizer.from_pretrained(MODEL)
  elif MODEL in ['xlm-roberta-base','nlpaueb/bert-base-greek-uncased-v1']:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
  elif MODEL=='microsoft/deberta-base':
    tokenizer = DebertaTokenizerFast.from_pretrained(MODEL)
  elif MODEL in ['bert-base-cased','bert-base-uncased']:
    tokenizer = BertTokenizer.from_pretrained(MODEL,do_lower_case = False)
  return tokenizer

