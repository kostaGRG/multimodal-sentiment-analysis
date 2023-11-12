import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

# Load text and image features and their labels from the PATH.
# If VADER=True use the vader features too.
def load_data(PATH,VADER):
  images = np.load(PATH+'image_logits.npy')
  texts = np.load(PATH+'text_logits.npy')
  labels = np.load(PATH+'labels.npy')
  multimodal = np.concatenate((texts,images),1)
  if VADER:
    vader = np.load(PATH+'vader_values.npy')
    multimodal = np.concatenate((multimodal,vader),1)
  return multimodal, labels

# Fully connected neural network with 4 hidden layers
class FiveLayersFC(nn.Module):
    def __init__(self, input_size, hidden_size, last_hidden_size, output_size):
        super(FiveLayersFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, last_hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(last_hidden_size, last_hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(last_hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

# Fully connected neural network with 3 hidden layers
class FourLayersFC(nn.Module):
    def __init__(self, input_size, hidden_size, last_hidden_size, output_size):
        super(FourLayersFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, last_hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(last_hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

# Fully connected neural network with 2 hidden layers
class ThreeLayersFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayersFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Multi-head attention and fully connected neural network
class AttentionFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
      super(AttentionFC, self).__init__()

      self.attention = nn.MultiheadAttention(input_size, num_heads)

      self.fc = nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, output_size)
      )

    def forward(self, x):
      attended_x, _ = self.attention(x, x, x)  # self-attention
      output = self.fc(attended_x)

      return output
    
# Initialize the chosen model, based on the value of MODEL. 
# Also initialize loss function, optimizer and learning rate scheduler.
def initialize_model(dataloader_train,lr=2e-5,epochs=10,hidden_size=500,last_hidden_size=100,num_heads=4,input_size=1536,output_size=3,MODEL='4 layers',device='cpu',scheduler_name='warmup'):

    if MODEL == '4 layers':
      model = FourLayersFC(input_size,hidden_size,last_hidden_size,output_size)
    elif MODEL == '3 layers':
      model = ThreeLayersFC(input_size,hidden_size,output_size)
    elif MODEL == '5 layers':
      model = FiveLayersFC(input_size,hidden_size,last_hidden_size,output_size) 
    else:
      model = AttentionFC(input_size,hidden_size,output_size,num_heads)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                      lr=lr
                      )

    if scheduler_name == 'warmup':
      total_steps = len(dataloader_train) * epochs
      scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
    if scheduler_name == 'reduce':
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=1, threshold=1e-4,verbose=True)
    elif scheduler_name == 'exponential':
      scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=True)
    return model, loss_fn, optimizer, scheduler
