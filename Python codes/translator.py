from deep_translator import GoogleTranslator
import h5py
import numpy as np
import pandas as pd
import sys
import re
import contractions
from tqdm import tqdm

### Read hdf5 file ### 
def read_hdf5(path):
    read_file = h5py.File(path, 'r')

    feature_names = list(read_file.keys())
    loaded_data = []

    for name in feature_names:
        dataset = read_file[name][:]
        if dataset.dtype == np.dtype('object'):
            dataset = np.array([x.decode('utf-8') for x in dataset])
        loaded_data.append((name, dataset))

    return loaded_data

### Load MVSA-Single dataset from stored file ###
def load_mvsa_data(path,mode):
    data = read_hdf5(path)
    if mode == 1: #multimodal
      for x in data:
          if x[0] == 'texts':
              texts = x[1]
          if x[0] == 'multimodal-labels':
              labels = x[1]
          if x[0] == 'images':
              images = x[1]
          if x[0] == 'text-labels':
              text_labels = x[1]
          if x[0] == 'image-labels':
              image_labels = x[1]
      return texts, images, labels, text_labels, image_labels

    elif mode == 2: # text only
      for x in data:
          if x[0] == 'texts':
              texts = x[1]
          if x[0] == 'text-labels':
              text_labels = x[1]
      return texts,text_labels

    elif mode == 3: # image only
      for x in data:
          if x[0] == 'images':
              images = x[1]
          if x[0] == 'image-labels':
              image_labels = x[1]
      return images,image_labels
    
### Apply text preprocessing techniques ###
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
    
### SELECT LANGUAGE FOR TRANSLATION: el,de, ru, es, hi ###
TARGET_LANGUAGE = 'hi'

### USE GOOGLE TRANSLATOR FOR THE TRANSLATION AND STORE THE TRANSLATED TEXTS TO A FILE NAMED translated.npy ###
# official_texts, images, labels, text_labels, image_labels = load_mvsa_data('mvsa-single-4511_multimodal.hdf5',1)
# np.save('official.npy',official_texts)
# texts = [text_preprocessing(text) for text in official_texts]
# np.save('texts.npy',texts)
official_texts = np.load('official.npy')
texts = np.load('texts.npy')
translated_texts = []
for text in tqdm(texts):
    translated_texts = np.append(translated_texts,GoogleTranslator(source='auto', target=TARGET_LANGUAGE).translate(text=text))
np.save('translated.npy',translated_texts)


### SOME TEXTS ARE HAVING ONLY SPACES OR SPECIAL CHARACTERS, IN THIS CASE FILL THEM AS THEY WERE ON THE OFFICIAL LANGUAGE ###
df = pd.DataFrame(data=zip(texts,official_texts,translated_texts),columns=('English','Official','Greek'))
df.to_excel(f'{TARGET_LANGUAGE}_texts.xlsx')

for (i,text) in enumerate(translated_texts):
    if text==None:
        translated_texts[i] = texts[i]

np.save(f'{TARGET_LANGUAGE}_translated.npy',translated_texts)

