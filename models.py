from torch import nn
from tcn import TemporalConvNet
from tcn import TcnnAutoEncoder
import torch.nn.functional as F
import torch

import numpy as np
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, temp, dropout, win_size,mean_emb):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.batch_norm = nn.BatchNorm1d(win_size)

        self.temp = temp
        self.mean_emb = mean_emb
        if self.mean_emb == 0:
            self.sig = nn.Softmax(dim=2)  # needs to b2 when not average. 1 when averaged
        else:
            self.sig = nn.Softmax(dim=1)  # needs to b2 when not average. 1 when averaged


    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        #output = self.batch_norm(output)

        if self.mean_emb == 1:
            output = torch.mean(output,1)

            output = self.linear(output)
            output = F.normalize(output,p=2,dim=1); #dim needs to be 2 when not averaged, 1 when averaged
        else:
            output = self.linear(output)

            output = F.normalize(output, p=2, dim=2);  # dim needs to be 2 when not averaged, 1 when averaged
        output = self.sig(self.temp*output)
        return   self.temp*output#self.sig(output)

class Feedforward(nn.Module):
    def __init__(self, input_size,  out_size,hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, hidden_size)
        self.fc5 = nn.Linear(self.hidden_size, hidden_size)
        self.fc6 = nn.Linear(self.hidden_size, out_size)

        self.sigmoid = nn.Sigmoid()
        self.sig = nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden2 = self.fc2(relu)
        relu = self.relu(hidden2)
        hidden3 = self.fc3(relu)
        #relu = self.relu(hidden3)
       # hidden4 = self.fc4(relu)
       # relu = self.relu(hidden4)
        #hidden5 = self.fc5(relu)
       # relu = self.relu(hidden5)
       # hidden6 = self.fc5(relu)
       # relu = self.relu(hidden6)
        #output = self.fc6(relu)
        output = self.sig(hidden3)
        return output




class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(Autoencoder,self).__init__()
        self.AE = TcnnAutoEncoder(input_size, output_size,num_channels, kernel_size, dropout=dropout)
    def forward(self, x):
        recon,encode =  self.AE(x.transpose(1, 2))

        recon = recon.transpose(1,2)
        encode = encode.transpose(1,2)
        return recon, encode

class NN_classification(nn.Module):
    def __init__(self, input_size, out_size):
        super(NN_classification, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.fc1 = nn.Linear(self.input_size, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, self.out_size)
        self.sig = nn.Softmax(dim=1)

    def forward(self,X):
        #X = F.normalize(X, p=2, dim=1)
        hidden = self.fc1(X)

        hidden = self.relu(hidden)
        #hidden= F.normalize(hidden, p=2, dim=1)
        hidden = self.fc2(hidden)
        hidden = self.relu(hidden)
        hidden = self.fc3(hidden)

        #hidden = F.normalize(hidden, p=2, dim=1)
        output = self.sig(hidden)

        return  output


class LadderNetwork(nn.Module):
    def __init__(self, input_size, out_size, kernel_size):
        super(LadderNetwork, self).__init__()
        self.enccov1d1 = nn.Conv1d(input_size,150,kernel_size)
        self.enccov1d2 = nn.Conv1d(150, 100, kernel_size)
        self.enccov1d3 = nn.Conv1d(100, 30, kernel_size)
        self.maxpool1d = nn.MaxPool1d(2,return_indices=True)
        self.erelu1 = nn.ReLU()
        self.erelu2 = nn.ReLU()
        self.erelu3 = nn.ReLU()
        self.drelu1 = nn.ReLU()
        self.drelu2 = nn.ReLU()
        self.drelu3 = nn.ReLU()
        self.softmax = nn.Softmax()
        self.encodefc1 = nn.Linear(420,out_size)
        self.ebatch1 = nn.BatchNorm1d(150)
        self.ebatch2 = nn.BatchNorm1d(100)
        self.ebatch3 = nn.BatchNorm1d(30)

        self.dbatch1 = nn.BatchNorm1d(input_size)
        self.dbatch2 = nn.BatchNorm1d(150)
        self.dbatch3 = nn.BatchNorm1d(100)
        self.decodefc1 = nn.Linear(out_size,420)

        self.decodeuppool = nn.MaxUnpool1d(2)
        self.deccov1d3 = nn.ConvTranspose1d(30,100,kernel_size)
        self.deccov1d2 = nn.ConvTranspose1d( 100, 150,kernel_size)
        self.deccov1d1 = nn.ConvTranspose1d( 150, input_size, kernel_size)

    def encoder(self,X):
        hidden1 =self.enccov1d1(X.transpose(1,2))
        hidden1 = self.erelu1(hidden1)
        hidden2 = self.enccov1d2(hidden1)
        hidden2 =self.erelu2(hidden2)
        hidden3 = self.enccov1d3(hidden2)
        self.hidden3 = self.erelu1(hidden3)
        hidden3,id = self.maxpool1d(hidden3)
        s = hidden3.shape
        hidden3= hidden3.reshape(s[0],s[1]*s[2])
        encoded = self.encodefc1(hidden3)
        return encoded,id,s

    def decoder(self,X,mpool_i,s):
        decode1 = self.decodefc1(X)
        decode1 = decode1.reshape(s[0],s[1],s[2])
        decode1 = (self.decodeuppool(decode1,mpool_i))
        decode1 = self.drelu1(self.deccov1d3(decode1))
        decode1 =  self.drelu2(self.deccov1d2(decode1))
        recon = self.deccov1d1(decode1)
        return recon.transpose(1,2)

    def forward(self,X):
        encoded,mpool_i,s = self.encoder(X)
        recon = self.decoder(encoded,mpool_i,s)
        return recon,encoded

