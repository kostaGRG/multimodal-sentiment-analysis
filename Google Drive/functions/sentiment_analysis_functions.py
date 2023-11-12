import h5py
import torch
import numpy as np
import os
from huggingface_hub import HfApi
from transformers import BertModel, RobertaModel, AlbertModel, DebertaModel, ViTModel, BeitModel, AutoModel
import torch.nn as nn
from torch.optim import AdamW
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import text_functions
from sklearn.model_selection import train_test_split

# Read data that are stored on hdf5 file
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

# Load the stored dataset, providing the path and the selected mode
# If mode=1, use both texts and images.
# If mode=2, use only texts.
# If mode=3, use only images.
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
  
# Map labels from {negative, neutral, positive} to {0,1,2}
def map_labels(labels):
    label_dict = {}
    label_dict['negative'] = 0
    label_dict['neutral'] = 1
    label_dict['positive'] = 2

    labels = np.array([label_dict[value] for value in labels])
    return labels,label_dict

# Create batches for the given array with the specified batch size
def get_batches(array, batch_size):
    num_batches = len(array) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch = array[start:end]
        yield batch
    # Handle the remaining elements if the array length is not divisible by batch_size
    if len(array) % batch_size != 0:
        yield array[num_batches * batch_size:]
        
# Save model to local file, storing the classifier and the tokenizer of the model
def save_model_to_local_file(SAVE_PATH, classifier, tokenizer):
    os.makedirs(SAVE_PATH)
    classifier.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    
# Connect to the Hugging Face API for uploading models to hugging face
def upload_to_hugging_face(SAVE_PATH,REPO_NAME,create_repo=False):
    api = HfApi()
    
    if create_repo:
        api.create_repo(repo_id=REPO_NAME, private=True)

    api.upload_folder(
        folder_path=SAVE_PATH,
        repo_id=REPO_NAME,
        # repo_type="space",
    )
    
# Create dataloaders for the data and the labels given as arguments. 
# Type='text' is used for text data and
# type='image' is used for image and multimodal data.
def create_dataloaders(data, labels, batch_size, generator, SHUFFLE, type, tokenizer=None):
  if type=='text':
    input_ids, attention_masks = text_functions.preprocessing_for_bert(data,tokenizer)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
  elif type=='image':
    dataset = TensorDataset(data, labels)
    
  dataloader = DataLoader(
      dataset,
      shuffle=SHUFFLE,
      batch_size=batch_size,
      generator=generator
  )

  return dataloader
    
# Split data to train/validation/test sets with percentages 80/10/10 with specified random seed.
def split_data(data,labels,seed_val):
  train_texts, X_rem, train_labels, y_rem = train_test_split(data,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=seed_val,
                                                    stratify=labels)


  test_texts, valid_texts, test_labels, valid_labels = train_test_split(X_rem,
                                                  y_rem,
                                                  test_size=0.5,
                                                  random_state=seed_val,
                                                  stratify=y_rem)
  

  train_labels = torch.tensor(train_labels)
  test_labels = torch.tensor(test_labels)
  valid_labels = torch.tensor(valid_labels)

  return train_texts,test_texts,valid_texts,train_labels,test_labels,valid_labels
    
    
# Class for text models (BERT, RoBERTa, ALBERT, DeBERTa)
# FC NN with dropout is used as the classifier.
# There is also the option of freezing the layers of the BERT model, used for some experiments.
class BertClassifier(nn.Module):

  def __init__(self, MODEL, dropout, freeze_bert=False):
      super(BertClassifier, self).__init__()
      # Specify hidden size of BERT, hidden size of our classifier, and number of labels

      if MODEL=='roberta-large':
        D_in, H, D_out = 1024, 400, 3
      else:
        D_in, H, D_out = 768, 100, 3

      if MODEL in ['roberta-base','roberta-large']:
        self.bert = RobertaModel.from_pretrained(MODEL)
      elif MODEL=='albert-base-v2':
        self.bert = AlbertModel.from_pretrained(MODEL)
      elif MODEL=='microsoft/deberta-base':
        self.bert = DebertaModel.from_pretrained(MODEL)
      elif MODEL in ['bert-base-cased','bert-base-uncased']:
        self.bert = BertModel.from_pretrained(MODEL)
      elif MODEL in ['xlm-roberta-base','nlpaueb/bert-base-greek-uncased-v1']:
        self.bert = AutoModel.from_pretrained(MODEL)

      # Instantiate an one-layer feed-forward classifier
      self.classifier = nn.Sequential(
          nn.Linear(D_in, H),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(H, D_out)
      )

      # Freeze the BERT model
      if freeze_bert:
          for param in self.bert.parameters():
              param.requires_grad = False

  def forward(self, input_ids, attention_mask):
      # Feed input to BERT
      outputs = self.bert(input_ids=input_ids,
                          attention_mask=attention_mask)

      # Extract the last hidden state of the token `[CLS]` for classification task
      last_hidden_state_cls = outputs[0][:, 0, :]

      # Feed input to classifier to compute logits
      logits = self.classifier(last_hidden_state_cls)

      return logits

  def save_model(self, path):
    self.bert.save_pretrained(path)
    torch.save(self.classifier.state_dict(), os.path.join(path, "classifier.pt"))
    
# Class for image models that are using the transformer's architecture (ViT, BEiT, DINO) 
# FC NN with dropout is used as the classifier.
class VisualTransformer(nn.Module):

  def __init__(self,MODEL,HIDDEN_LAYER,dropout):
      super(VisualTransformer, self).__init__()
      D_in, H, D_out = 768, 100, 3

      if MODEL == 'microsoft/beit-base-patch16-224-pt22k-ft22k':
        self.vit = BeitModel.from_pretrained(MODEL)
      else:
        self.vit = ViTModel.from_pretrained(MODEL)

      # Instantiate an one-layer feed-forward classifier
      if HIDDEN_LAYER:
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, D_out)
        )
      else:
        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_out)
        )

  def forward(self, images):
      outputs = self.vit(images)
      last_hidden_state_cls = outputs[0][:, 0, :]
      logits = self.classifier(last_hidden_state_cls)
      return logits

  def save_model(self, path):
    self.vit.save_pretrained(path)

# Initialize the chosen model, specified by the parameter MODEL.
# Also initialize optimizer, loss function and LR scheduler. 
def initialize_model(device, lr, dataloader_train, scheduler_name, MODEL, dropout, HIDDEN_LAYER, type='text', epochs=10):

    if type=='text':
        model = BertClassifier(MODEL, dropout, freeze_bert=False)
    elif type=='image':
        model = VisualTransformer(MODEL,HIDDEN_LAYER,dropout)

    # Tell PyTorch to run the model on GPU
    model.to(device)

    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=lr
                      )
    criterion = nn.CrossEntropyLoss()


    # Total number of training steps
    total_steps = len(dataloader_train) * epochs


    if scheduler_name == 'warmup':
      total_steps = len(dataloader_train) * epochs
      scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
    elif scheduler_name == 'reduce':
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=1, threshold=1e-4,verbose=True)
    elif scheduler_name == 'exponential':
      scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=True)
    return model, criterion, optimizer, scheduler

# Train the selected model. Training hyperparameters are given as arguments.
# With the PRINT option you can select if you will print the results on each epoch or not.
# With the CONFIDENCT_ACC you can monitor how confident are the predictions of the model.
# Type parameter must be specified so the function will know if a text or image model is used.
# If val_dataloader is not None, evaluation on the validation set is enabled each epoch. 
def train(model, train_dataloader, optimizer, scheduler, loss_fn, experiment=None, device='cpu', type='text', val_dataloader=None, test_dataloader=None, epochs=10, evaluation=False, FREEZE=False,CONFIDENT_ACC=False,PRINT=True):

    if CONFIDENT_ACC:
       confidence_60_percent = []
       confidence_70_percent = []
       confidence_80_percent = []
       confidence_90_percent = []

    if PRINT:
        print("Start training...\n")
    for epoch_i in tqdm(range(epochs)):
        if type =='image' and FREEZE and (epoch_i==2):
          for name,param in model.named_parameters():
              if ("classifier" not in name) and ("pooler" not in name):
                param.requires_grad = False
        progress_bar = tqdm(train_dataloader,
                        desc='Epoch {:1d}'.format(epoch_i),
                        leave=False,
                        disable=False)
        total_loss = 0
        model.train()

        for batch in progress_bar:
            if type=='text':
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            elif type=='image':
                images, b_labels = batch
                images  = images.to(device)
                b_labels = b_labels.to(device)
                
            model.zero_grad()
            optimizer.zero_grad()

            if type=='text':
                logits = model(b_input_ids, b_attn_mask)
            elif type=='image':
                logits = model(images)
                
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        if PRINT:
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            tqdm.write(f'\nEpoch {epoch_i}')
            tqdm.write(f'Training loss: {avg_train_loss}')

        if evaluation == True:
            val_loss, predictions, val_labels = evaluate(model, loss_fn, device, val_dataloader, type)
            val_f1 = f1_score(val_labels, predictions, average = 'weighted')
            val_acc = accuracy_score(val_labels, predictions)
            if PRINT:
                tqdm.write(f'Validation loss: {val_loss}')
                tqdm.write(f'Validation F1 Score (weighted): {val_f1}')
                tqdm.write(f'Validation Accuracy Score: {val_acc}')
            if experiment is not None:
                metrics = {
                    'train loss': avg_train_loss,
                    'validation loss': val_loss,
                    'f1 score': val_f1,
                    'accuracy': val_acc
                }
                experiment.log_metrics(metrics, epoch=epoch_i)

            test_loss, predictions, test_labels = evaluate(model, loss_fn, device, test_dataloader, type)
            test_f1 = f1_score(test_labels, predictions, average = 'weighted')
            test_acc = accuracy_score(test_labels, predictions)
            if PRINT:
                tqdm.write(f'Test F1 Score (weighted): {test_f1}')
                tqdm.write(f'Test Accuracy Score: {test_acc}')
            if experiment is not None:
                metrics = {
                    'train loss': avg_train_loss,
                    'validation loss': val_loss,
                    'f1 score': val_f1,
                    'accuracy': val_acc,
                    'test f1 score': test_f1,
                    'test accuracy score': test_acc
                }
                experiment.log_metrics(metrics, epoch=epoch_i)

            if CONFIDENT_ACC:
               probs, true_values = predict(model,val_dataloader,device,type)
               confidence_60_percent.append(calculate_confident_accuracy(predictions=probs,true_values=true_values,threshold=0.6)[0])
               confidence_70_percent.append(calculate_confident_accuracy(predictions=probs,true_values=true_values,threshold=0.7)[0])
               confidence_80_percent.append(calculate_confident_accuracy(predictions=probs,true_values=true_values,threshold=0.8)[0])
               confidence_90_percent.append(calculate_confident_accuracy(predictions=probs,true_values=true_values,threshold=0.9)[0])


        else:
            if experiment is not None:
                experiment.log_metric('train loss',avg_train_loss, epoch=epoch_i)
        scheduler.step()
        if PRINT:
            print("\n")

    print("Training complete!")

    if CONFIDENT_ACC:
       return predictions, test_labels, confidence_60_percent, confidence_70_percent, confidence_80_percent, confidence_90_percent
    return predictions, test_labels

# Evaluate the model performance on the validation set.
def evaluate(model, loss_fn, device, val_dataloader, type):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    val_loss = np.array([])
    true_values = np.array([])
    predictions = np.array([])

    for batch in tqdm(val_dataloader):
        if type=='text':
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        elif type=='image':
            images, b_labels = tuple(t.to(device) for t in batch)
                
        with torch.no_grad():
            if type=='text':
                logits = model(b_input_ids, b_attn_mask)
            elif type=='image':
                logits = model(images)
                
        loss = loss_fn(logits, b_labels)
        val_loss = np.append(val_loss,loss.item())

        logits = logits.detach().cpu().numpy()
        true_values = np.append(true_values,b_labels.detach().cpu().numpy().flatten())
        predictions = np.append(predictions,np.argmax(logits,axis=1).flatten())
    val_loss = np.mean(val_loss)

    return val_loss, predictions, true_values

# Predict sentiment based on the input, using the test set.
def predict(model, test_dataloader, device, type):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. 
    # The dropout layers are disabled during the test time.
    model.eval()

    all_logits = []
    true_values = np.array([])
    # For each batch in our test set...
    for batch in test_dataloader:
        if type=='text':
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        elif type=='image':
            images, b_labels = tuple(t.to(device) for t in batch)
                
        with torch.no_grad():
            if type=='text':
                logits = model(b_input_ids, b_attn_mask)
            elif type=='image':
                logits = model(images)
                
        all_logits.append(logits)
        true_values = np.append(true_values,b_labels.detach().cpu().numpy().flatten())
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()


    return probs, true_values

# Calculate features for the data inside the dataloader using the specified model.
# This function is called after the complete training of the model.
def calculate_logits(model, dataloader, device, type):
    model.to(device)
    model.eval()
    logits = []
    for batch in tqdm(dataloader):
        if type =='text':
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        elif type=='image':
            b_images,b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            if type=='text':
                logits.append(model(b_input_ids, b_attn_mask)[0][:,0,:])
            elif type=='image':  
                logits.append(model(b_images)[0][:,0,:])

    logits = torch.cat(logits,dim=0)
    logits = logits.cpu().numpy()

    return logits   


# Calculate how confident is the model, based on the chosen threshold=[0,1]
def calculate_confident_accuracy(predictions, true_values, threshold):
  total_predictions = len(predictions)
  correct_confident_predictions = 0
  confident_predictions = 0
  unconfident_predictions = 0
  correct_unconfident_predictions = 0

  for pred, true_val in zip(predictions, true_values):
      predicted_class = np.argmax(pred)  # Get the class with the highest probability
      if pred[predicted_class] >= threshold and predicted_class == true_val:
          correct_confident_predictions += 1
      elif pred[predicted_class] < threshold:
          unconfident_predictions += 1
          if predicted_class == true_val:
            correct_unconfident_predictions += 1

  confident_predictions = len(predictions) - unconfident_predictions
  if confident_predictions == 0:
    confident_accuracy = 0
    unconfident_accuracy = 1
  elif unconfident_predictions == 0:
    confident_accuracy = 1
    unconfident_accuracy = 0
  else:
    confident_accuracy = correct_confident_predictions / confident_predictions
    unconfident_accuracy = correct_unconfident_predictions / unconfident_predictions
  print(f"Percentage of confident predictions = {confident_predictions/total_predictions}")
  return confident_accuracy, unconfident_accuracy

# Calculate the accuracy of the model for each of the available classes
def accuracy_per_class(preds, labels,label_dict):
  label_dict_inverse = {v: k for k, v in label_dict.items()}

  preds_flat = np.argmax(preds, axis=0).flatten()
  labels_flat = labels.flatten()

  for label in np.unique(labels_flat):
      y_preds = preds_flat[labels_flat==label]
      y_true = labels_flat[labels_flat==label]
      print(f'Class: {label_dict_inverse[label]}')
      print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')

# Print the accuracy of the model for each class
def print_stats(bert_probs_test,test_dataloader_labels,labels_dict):
  accuracy_per_class(bert_probs_test,test_dataloader_labels,labels_dict)
  confident_accuracy,unconfident_accuracy = calculate_confident_accuracy(bert_probs_test,test_dataloader_labels, threshold=0.75)
  print(f"accuracy on confident predictions= {confident_accuracy}")
  print(f"accuracy on non confident predictions= {unconfident_accuracy}")
  return