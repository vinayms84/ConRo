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


def build_word_to_vector():
  data = []
  Label= []
  ignore = {".DS_Store", ".txt"}
  session_count=0
  for root, dirs, files in os.walk("all_sessions"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()
             
              for line in Lines:
                acts, _, label = line.split(';')
                temp = []
                session_count=session_count+1
               
                for act in acts.split(','):
                  temp.append(act.lower())
                  
                data.append(temp)
                Label.append(label.strip('\n'))


  return data

features=50
seq_length=25

data=build_word_to_vector()

word2Vec_model = gensim.models.Word2Vec(
                    data, 
                    sg=1, 
                    vector_size=features, 
                    min_count=1
                     
                    )

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

input_size = features
hidden_size = features
num_layers = 2
num_classes = features
sequence_length = seq_length
learning_rate = 0.005
batch_size = 120
num_epochs = 10

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
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

def set_A(data, target, index):
  n=data.shape[0]
  m=n-1
  data=data.detach().cpu().numpy()
  target=target.detach().cpu().numpy()
  A=np.zeros([m,data.shape[1]], dtype=np.float32)
  Alabel=np.zeros(m, dtype=np.float32)
  j=0
  for i in range(n):
    if i != index:
      A[j]=data[i]
      Alabel[j]=target[i]
      j=j+1
  A=torch.from_numpy(A)
  Alabel=torch.from_numpy(Alabel) 
  return A, Alabel

def set_B(A, Alabel, label):
  n=A.shape[0]
  A=A.detach().cpu().numpy()
  Alabel=Alabel.detach().cpu().numpy()
  count=0
  for i in range(n):
    if Alabel[i]==label:
      count=count+1
  
  B=np.zeros([count,A.shape[1]], dtype=np.float32)
  Blabel=np.zeros(count, dtype=np.float32)

  count=0
  for i in range(n):
    if Alabel[i]==label:
      B[count]=A[i]
      Blabel[count]=Alabel[i]
      count=count+1

  B=torch.from_numpy(B)
  Blabel=torch.from_numpy(Blabel) 
  return B, Blabel

def normal_distribution_center_stage1(data, target):
  n=target.shape[0]
  count_1 = 0
  data=data.detach().cpu().numpy()
  target=target.detach().cpu().numpy()
  for i in range(n):
    if target[i]==0:
      count_1 = count_1+1
     
  normal = np.zeros([count_1,data.shape[1]], dtype=np.float32)

  index_1=0
  for k in range(n):
    if target[k]==0:
      normal[index_1]=data[k]
      index_1=index_1+1
  

  V0=np.mean(normal, axis=0)

  V0=torch.from_numpy(V0)
  

  return V0

def loss_pair(query, A, B, index, alpha=1):
  cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  numerator=torch.exp((cos(query.view(1,features),B[index,:].view(1,features)))/alpha)
  denom=0
  for k in range(A.shape[0]):
    denom=denom+torch.exp((cos(query.view(1,features),A[k,:].view(1,features)))/alpha)
    
  score=-torch.log(torch.div(numerator,denom))

  return score

def contrastive_loss(values, target):
  batch_loss=0
  for i in range(values.shape[0]):
    if target[i] == 0:
      query=values[i]
      A, Alabel = set_A(values, target, i)
      B, Blabel = set_B(A, Alabel, target[i])
      loss=0
      for j in range(B.shape[0]):
        loss=loss+loss_pair(query, A, B, j)
      if B.shape[0] > 0:
        loss=loss/B.shape[0]
      batch_loss=batch_loss+loss
    
  return batch_loss

def DeepSVDD_loss(values, target):
  batch_loss = 0
  V0 = normal_distribution_center_stage1(values, target)
  for i in range(values.shape[0]):
    if target[i] == 0:
      batch_loss = batch_loss+LA.norm(values[i]-V0)

  return batch_loss

def make_dataset(act_file, session_count):
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  temp_label_old= []
  session_number=0
  Lines = act_file.readlines()
  for line in Lines:
    acts, temp_label, label = line.split(';')
    temp_label_old.append(temp_label)
    session_label_old.append(label.strip('\n'))
    sequence_number=0
    for act in acts.split(','):
      if sequence_number<seq_length:
        x=word2Vec_model.wv.get_vector(act.lower())
        for i in range(features):
          dataset[session_number][sequence_number][i]=x[i]
        sequence_number=sequence_number+1
    session_number=session_number+1

  temp_label=np.array(temp_label_old)
  temp_label= temp_label.astype(np.float32)
  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, temp_label, session_label

def train_model(dataloader,net,optimizer):
  for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()
    values, _ = net(data)
    loss = contrastive_loss(values, targets)
    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    values, _ = net(data)
    loss = DeepSVDD_loss(values, targets)
    loss.backward()
    optimizer.step()
    
  return loss

def train_dataset(seq_length,features, model,optimizer,batch_size):
  session_count=batch_size
  for epoch in range(num_epochs):
    for root, dirs, files in os.walk("training_set"):
      for filename in files:
        with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
          data_vector, temp_label, label = make_dataset(act_file, session_count)
          temp_label=torch.from_numpy(temp_label) 
          dataset=Seq_Dataset(data_vector,label)
          dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
          loss = train_model(dataloader,model,optimizer)    
  
  return loss

loss = train_dataset(seq_length, features, net, optimizer,batch_size)
save_path = './encoder.pth'
torch.save(net.state_dict(), save_path)
word2Vec_model.save("word2vec.model")

