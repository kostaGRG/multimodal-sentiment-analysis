import re
import contractions
import torch

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """

    text = re.sub('RT '+r'(@.*?)[\s]', '', text)
    text = re.sub(r'(@.*?)[\s]', '', text)
    text = re.sub(r'#','',text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\b\d+\b','', text)
    text = contractions.fix(text)
    return text

def preprocessing_for_bert(text,tokenizer):
    input_ids = []
    attention_masks = []

    encoded_text = tokenizer.encode(text, add_special_tokens=True)
    MAX_LEN = len(encoded_text)

    encoded_sent = tokenizer.encode_plus(
        text=text,
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