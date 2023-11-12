import preprocessing
import torch
import numpy as np
import os 
import classes
import torch.nn as nn

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}
LABEL_DICT = ['Negative', 'Neutral', 'Positive']
englishBert,greekBert,imageVit,multimodalModel,englishTokenizer,greekTokenizer,imageProcessor = classes.load_models()


def detect_language(text):
    greek_characters = set("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω")
    greek_count = 0
    for char in text:
        if char in greek_characters:
            greek_count += 1
    return greek_count > 0.2*len(text)

def allowed_file(filename):
    # Extract the file extension from the filename
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension[1:] in ALLOWED_EXTENSIONS

def makePredictions(text,image):
    textExists = text != None
    imageExists = image != None
    textLabel = '-'
    imageLabel = '-'
    finalLabel = '-'
    probs = {"text":[],
             "image":[],
             "final":[]
             }
    softmax = nn.Softmax(dim=1)
    if textExists:
        isGreek = detect_language(text)
        text = preprocessing.text_preprocessing(text)
        if isGreek:
            input_ids, attention_masks = preprocessing.preprocessing_for_bert(text,greekTokenizer)
            with torch.no_grad():
                textPred = greekBert(input_ids,attention_masks)
                textLogits = greekBert.bert(input_ids,attention_masks)[0][:,0,:]
        else:
            input_ids, attention_masks = preprocessing.preprocessing_for_bert(text,englishTokenizer)
            with torch.no_grad():
                textPred = englishBert(input_ids,attention_masks)
                textLogits = englishBert.bert(input_ids,attention_masks)[0][:,0,:]
        textLabel = np.argmax(textPred)
        textLabel = LABEL_DICT[textLabel]
        probs["text"] = torch.mul(softmax(textPred).squeeze(),100).tolist()
    
    if imageExists:
        image = imageProcessor(image, return_tensors='pt')['pixel_values'][0]
        image = image[np.newaxis,...]
        with torch.no_grad():
            imagePred = imageVit(image)
            imageLogits = imageVit.vit(image)[0][:,0,:]
        imageLabel = np.argmax(imagePred)
        imageLabel = LABEL_DICT[imageLabel]
        probs["image"] = torch.mul(softmax(imagePred).squeeze(),100).tolist()
        print(probs["image"])
    if imageExists and textExists:
        multimodal = np.concatenate((textLogits,imageLogits),1)
        multimodal = torch.tensor(multimodal)
        with torch.no_grad():
            finalPred = multimodalModel(multimodal)
        finalLabel = np.argmax(finalPred)
        finalLabel = LABEL_DICT[finalLabel]
        probs["final"] = torch.mul(softmax(finalPred).squeeze().squeeze(),100).tolist()
    return textLabel, imageLabel, finalLabel, probs
