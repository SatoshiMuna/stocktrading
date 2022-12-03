import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class StockSeriesRNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, input_length, output_size, fcst_period):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    #self.fc = nn.Linear(in_features=input_length * hidden_size, out_features=output_size * fcst_period)
    self.fc_last = nn.Linear(in_features=hidden_size, out_features=output_size * fcst_period)

  def forward(self, x):
    h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # h0:(D*nl,N,Hi)
    out, h_n = self.rnn(x, h_0)  # out:(N,L,D*Ho)
    #out = out.reshape(out.shape[0],-1) # In case of using all hidden layers output 
    #out = self.fc(out)
    out = self.fc_last(out[:,-1,:])    # In case of using only the last hidden layer output
    return out

class StockSeriesLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, input_length, output_size, fcst_period):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(in_features=hidden_size, out_features=output_size * fcst_period)

  def forward(self, x):
    h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)#to.(dtype=torch.float64, device=device)  # h0:(D*nl,N,Hi)
    c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)#to.(dtype=torch.float64, device=device)  # h0:(D*nl,N,Hc)
    out, (h_n, c_n) = self.lstm(x, (h_0, c_0))                                                              # out:(N,L,D*Ho)
    out = self.fc(out[:,-1,:]) 
    return out