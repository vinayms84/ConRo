import random
from torch.utils.data.sampler import Sampler
import torch
import torchvision  
import torch.nn.functional as F  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader  
from tqdm import tqdm  
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.preprocessing import StandardScaler
from torch import linalg as LA
import numpy as np
import math
import gensim
import os
import nltk
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
nltk.download('punkt')




class Seq_Dataset(Dataset):
  def __init__(self, dataset, labels):

    labels=labels.flatten()
    self.x=torch.from_numpy(dataset[:,:,:])
    self.y=torch.from_numpy(labels).type(torch.LongTensor)
    self.n_samples=dataset.shape[0]


  def __getitem__(self,index):
    return self.x[index], self.y[index]

  def __len__(self):
    return self.n_samples

features=50
seq_length=25
input_size = features
hidden_size = features
num_layers = 2
num_classes = features
sequence_length = seq_length



# LSTM 

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).float()
    
    
    
  def forward(self, x):
    # Set initial hidden and cell states
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    # Forward propagate LSTM
    rnn_out, _ = self.lstm(x, (h0,c0))
    out = torch.mean(rnn_out, axis = 1)
    

    return out, _

net = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
save_path = './encoder.pth'
new_word2vec = gensim.models.Word2Vec.load("word2vec.model")
new_net = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
new_net.load_state_dict(torch.load(save_path))
new_net.eval()

def distribution_centers(seq_length,features):
  ignore = {".DS_Store", ".txt"}
  session_count=0
  for root, dirs, files in os.walk("sampled_sessions"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()
             
              for line in Lines:
                
                session_count=session_count+1
  
  print('session_count: ', session_count)
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  session_number=0
  for root, dirs, files in os.walk("sampled_sessions"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()
              for line in Lines:
                acts, _, label = line.split(';')
                session_label_old.append(label.strip('\n'))
                sequence_number=0
                for act in acts.split(','):
                  if sequence_number<seq_length:
                    x=new_word2vec.wv.get_vector(act.lower())
                    for i in range(features):
                      dataset[session_number][sequence_number][i]=x[i]
                    sequence_number=sequence_number+1
                session_number=session_number+1

  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, session_label

def get_distribution_centers(data, target):
  n = target.shape[0]
  count_1 = 0
  count_0 = 0
  data=data.detach().cpu().numpy()
  target=target.detach().cpu().numpy()
  for i in range(n):
    if target[i]==1:
      count_1 = count_1+1
    else:
      count_0 = count_0+1

   

  malicious = np.zeros([count_1,data.shape[1]], dtype=np.float32)
  normal = np.zeros([count_0,data.shape[1]], dtype=np.float32)
  
  
  index_1=0
  index_0=0
  for k in range(n):
    if target[k]==1:
      malicious[index_1]=data[k]
      index_1=index_1+1
    else:
      normal[index_0]=data[k]
      index_0 = index_0+1

   

  V1=np.mean(malicious, axis=0)
  V0=np.mean(normal, axis=0)
 

  V1=torch.from_numpy(V1)
  V0=torch.from_numpy(V0)

  return V1, V0

data_vector, label= distribution_centers(seq_length, features)
dataset=Seq_Dataset(data_vector,label)
dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
  encode,_ = new_net(data)
V1,V0 = get_distribution_centers(encode, targets)

def test_dataset(seq_length,features):
  ignore = {".DS_Store", ".txt"}
  session_count=0
  for root, dirs, files in os.walk("test_data"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()
             
              for line in Lines:
                
                session_count=session_count+1
  
  print('sesion_count: ', session_count)
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  session_number=0
  for root, dirs, files in os.walk("test_data"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()
              for line in Lines:
                acts, _, label = line.split(';')
                session_label_old.append(label.strip('\n'))
                sequence_number=0
                for act in acts.split(','):
                  if sequence_number<seq_length:
                    x=new_word2vec.wv.get_vector(act.lower())
                    for i in range(features):
                      dataset[session_number][sequence_number][i]=x[i]
                    sequence_number=sequence_number+1
                session_number=session_number+1

  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, session_label

data_vector, label= test_dataset(seq_length, features)
dataset = Seq_Dataset(data_vector,label)
dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
  encode,_ = new_net(data)

def calculate_accuracy(data, target, V1, V0):
  
  count=0
  pred_label=np.zeros(target.shape[0], dtype=np.float32)
  for i in range(target.shape[0]):
    if LA.norm(data[i]-V1)<= LA.norm(data[i]-V0):
      pred_label[i]=1
    else:
      pred_label[i]=0


  pred_label=torch.from_numpy(pred_label)

  return pred_label, (torch.sum(torch.eq(pred_label, target))/target.shape[0])*100

def calculate_performance_metrics(data, target, V1, V0):
  
  count=0
  pred_label=np.zeros(target.shape[0], dtype=np.float32)
  for i in range(target.shape[0]):
    if LA.norm(data[i]-V1)<= LA.norm(data[i]-V0):
      pred_label[i]=1
    else:
      pred_label[i]=0


  pred_label=torch.from_numpy(pred_label)

  tp = 0
  fp = 0
  fn = 0
  tn = 0

  for i in range(target.shape[0]):
    if (pred_label[i] == target[i]) & (target[i]==1):
      tp=tp+1
    elif (pred_label[i] != target[i]) & (target[i]==0):
      fp=fp+1
    elif (pred_label[i] != target[i]) & (target[i]==1):
      fn=fn+1
    elif (pred_label[i] == target[i]) & (target[i]==0):
      tn=tn+1


  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = 2*((precision*recall)/(precision+recall))
  fpr = fp/(tn+fp)

  
  return precision, recall, f1, fpr

def calculate_auc_scores(data, target, pred_label, V1, V0):

  pred_label=np.zeros(target.shape[0], dtype=np.float32)
  for i in range(target.shape[0]):
    if LA.norm(data[i]-V1)<= LA.norm(data[i]-V0):
      pred_label[i]=1
    else:
      pred_label[i]=0

  target=target.detach().cpu().numpy()
  roc_auc = roc_auc_score(target, pred_label)
  avg_precision = average_precision_score(target, pred_label)

  return roc_auc, avg_precision

pred_label, accuracy=calculate_accuracy(encode, targets, V1, V0)
precision, recall, f1, fpr = calculate_performance_metrics(encode, targets, V1, V0)
roc_auc, avg_precision = calculate_auc_scores(encode, targets, pred_label, V1, V0)
print(f"Accuracy: {accuracy}")
print(f"Precision, recall, f1, fpr: {precision, recall, f1, fpr}")
print(f"auc_roc, auc_pr: {roc_auc, avg_precision}")
