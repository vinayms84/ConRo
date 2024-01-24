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





features=50
seq_length=25
input_size = features
hidden_size = features
num_layers = 2
num_classes = features
sequence_length = seq_length
learning_rate = 0.005
batch_size = 120
num_epochs = 10


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

save_path = './encoder.pth'
word2Vec_model = gensim.models.Word2Vec.load("word2vec.model")
net = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
net.load_state_dict(torch.load(save_path))
net.eval()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

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

def set_A_stage2(data, target, index):
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

def set_B_stage2(A, Alabel, label):
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

def build_auxillary_malicious_batch(data, target, temp_label):
  count = 0 
  for i in range(data.shape[0]):
    if target[i] == 1 and temp_label[i] == 3:
      count = count+1

  B = np.zeros([count, data.shape[1]], dtype=np.float32)

  count = 0
  
  for i in range(data.shape[0]):
    if target[i] == 1 and temp_label[i] == 3:
      B[count] = data[i]
      count = count + 1

  return B

def get_positive_sample(X, batch_size):
  k = int(batch_size*np.random.randint(1))
  return X[k]

def set_B_hat(query, values, target, temp_label, M=20, beta_1=0.92):
  query = query.detach().cpu().numpy()
  values = values.detach().cpu().numpy()
  target = target.detach().cpu().numpy()
  temp_label = temp_label.detach().cpu().numpy()
  B_hat = np.zeros([M, values.shape[1]], dtype=np.float32)
  B = build_auxillary_malicious_batch(values, target, temp_label)

  for i in range(B_hat.shape[0]):
    alpha_1 = np.random.uniform(beta_1,1)
    B_hat[i] = (alpha_1*query) + ((1-alpha_1)*get_positive_sample(B, B.shape[0]))

  return torch.from_numpy(B_hat)

def normal_distribution_center_stage2(data):
  data=data.detach().cpu().numpy()
  V0=np.mean(data, axis=0)
  V0=torch.from_numpy(V0)
  return V0

def calculate_radius(values, V0, nu = 0.1):
  dist = torch.zeros(values.shape[0])
  for i in range(values.shape[0]):
    dist[i] = LA.norm(values[i]-V0)
  
  dist = dist.detach().cpu().numpy()
  radius = np.quantile(dist, 1 - nu)
  radius = torch.from_numpy(np.asarray(radius))
  return radius

def false_positive(sample, V0, radius):
  sample = torch.from_numpy(sample)
  if LA.norm(sample-V0)<=radius:
    return True
  else:
    return False

def set_B_tilde(query, values, target, temp_label, V0, radius, M=200, beta_2=4, mul = 10):
  query = query.detach().cpu().numpy()
  values = values.detach().cpu().numpy()
  target = target.detach().cpu().numpy()
  temp_label = temp_label.detach().cpu().numpy()
  flag = True
  B_tilde = np.zeros([M, values.shape[1]], dtype=np.float32)
  B = build_auxillary_malicious_batch(values, target, temp_label)
  count = 0
  i = 0
  while i<B_tilde.shape[0] and count < mul*M:
    alpha_2 = np.random.uniform(-beta_2,beta_2)
    X = (alpha_2*query) + ((1-alpha_2)*get_positive_sample(B, B.shape[0]))
    if not false_positive(X, V0, radius):
      B_tilde[i] = X
      i = i+1
    else:
      count = count+1 
  B_tilde = B_tilde[0:i]
  if i>0:
    flag = False

  return torch.from_numpy(B_tilde), flag

def loss_pair(query, A, B, index, alpha=1):
  cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  numerator=torch.exp((cos(query.view(1,features),B[index,:].view(1,features)))/alpha)
  denom=0
  for k in range(A.shape[0]):
    denom=denom+torch.exp((cos(query.view(1,features),A[k,:].view(1,features)))/alpha)
    
  score=-torch.log(torch.div(numerator,denom))

  return score

def contrastive_loss_stage2(values, target, temp_label, normal_values, V0, radius):
  batch_loss = 0
  zero_flag =True 
  for i in range(values.shape[0]):
    flag = target[i]+temp_label[i]
    if flag!= 4 and target[i] == 1:
      query=values[i]
      A, Alabel = set_A_stage2(values, target, i)
      B, Blabel = set_B_stage2(A, Alabel, target[i])
      B_hat = set_B_hat(query, values, target, temp_label)
      B_tilde, null_flag = set_B_tilde(query, values, target, temp_label, V0, radius)
      if null_flag:
        A = torch.cat((A, B_hat), 0)
        B = torch.cat((B, B_hat), 0)
      else:
        A = torch.cat((A, B_hat, B_tilde), 0)
        B = torch.cat((B, B_hat, B_tilde), 0)
      loss=0
      for j in range(B.shape[0]):
        loss=loss+loss_pair(query, A, B, j)
      if B.shape[0] > 0:
        loss=loss/B.shape[0]
      batch_loss=batch_loss+loss
      zero_flag = False
    
  return batch_loss, zero_flag

def make_normal_dataset(seq_length, features):
  for root, dirs, files in os.walk("normal"):
    for filename in files:
      with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
        Lines = act_file.readlines()
  
  
  session_count = len(Lines)
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old = []
  session_number=0
  for line in Lines:
    acts, _, label = line.split(';')
    session_label_old.append(label.strip('\n'))
    sequence_number=0
    for act in acts.split(','):
      if sequence_number<seq_length:
        x=word2Vec_model.wv.get_vector(act.lower())
        for i in range(features):
          dataset[session_number][sequence_number][i]=x[i]
        sequence_number=sequence_number+1
    session_number=session_number+1

  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, session_label

def get_normal_batch(dataloader,net):
  for batch_idx, (data, targets) in enumerate(dataloader):
    values, _ = net(data)
  return values

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

def train_model(dataloader,net,optimizer, temp_label, normal_dataloader):
  for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()
    values, _ = net(data)
    normal_values =  get_normal_batch(normal_dataloader, net)
    V0 = normal_distribution_center_stage2(normal_values)
    radius = calculate_radius(normal_values, V0)
    loss, zero_flag = contrastive_loss_stage2(values, targets, temp_label, normal_values, V0, radius)
    if not zero_flag:
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
          normal_data, normal_label = make_normal_dataset(seq_length, features)
          normal_dataset = Seq_Dataset(normal_data,normal_label)
          normal_dataloader = DataLoader(dataset=normal_dataset, shuffle=False, batch_size=len(normal_dataset))
          loss = train_model(dataloader,model,optimizer, temp_label, normal_dataloader)    
  
  return loss

loss = train_dataset(seq_length, features, net, optimizer,batch_size)
save_path = './encoder.pth'
torch.save(net.state_dict(), save_path)
word2Vec_model.save("word2vec.model")

