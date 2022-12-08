import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SeriesRNN(nn.Module):
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

class SeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_length, bidirectional, output_size, fcst_period):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size * fcst_period)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)#to.(dtype=torch.float64, device=device)  # h0:(D*nl,N,Hi)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)#to.(dtype=torch.float64, device=device)  # h0:(D*nl,N,Hc)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))  # out:(N,L,D*Ho)
        out = self.fc(out[:,-1,:]) 
        return out

class SelfAttention(nn.Module):
    '''Zhouhan Lin el al., "A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING", ICLR2017.
       a self-attention method for sentiment classification
    '''
    def __init__(self, dim_h_out, dim_w1, dim_w2):
        super().__init__()
        self.W1 = nn.Linear(dim_h_out, dim_w1, bias=False)
        self.W2 = nn.Linear(dim_w1, dim_w2, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, rnn_out):  # rnn_out:(N, L, D*Ho)              
        scores = self.W2(self.tanh(self.W1(rnn_out)))   # scores:(N, L, r)
        attn_weights = F.softmax(scores, dim=2)
        out = torch.matmul(torch.transpose(attn_weights, 1, 2), rnn_out) # out:(N, r, L)*(N, L, D*Ho)→(N, r, D*Ho)
        return out, attn_weights

class SelfAttnSeriesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_length, bidirectional, dim_w1, dim_w2, output_size, fcst_period):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_length = input_length
        self.bidirectional = bidirectional
        self.dim_h_out = hidden_size * 2 if bidirectional else hidden_size
        self.da = dim_w1
        self.r = dim_w2
        self.attn_weights = None
        self.D = 2 if bidirectional else 1
        
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.attn = SelfAttention(dim_h_out=self.dim_h_out, dim_w1=self.da, dim_w2=self.r)
        self.fc = nn.Linear(self.r*self.dim_h_out, output_size*fcst_period)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size).to(device)
        out, h_n = self.rnn(x, h_0)              # out:(N, L, D*Ho)
        out, self.attn_weights = self.attn(out)  # out:(N, r, D*Ho), attn:(N, L, r)
        out = self.fc(out.view(batch_size, -1))  # out:(N, 1)
        return out

class SelfAttnSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_length, bidirectional, dim_w1, dim_w2, output_size, fcst_period):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.D = 2 if bidirectional else 1
        self.dim_h_out = 2 * hidden_size if bidirectional else hidden_size
        self.da = dim_w1
        self.r = dim_w2
        self.attn_weights = None

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.attn = SelfAttention(dim_h_out=self.dim_h_out, dim_w1=self.da, dim_w2=self.r)
        self.fc = nn.Linear(self.r*self.dim_h_out, output_size*fcst_period)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))  # out:(N, L, D*Ho)
        out, self.attn_weights = self.attn(out)     # out:(N, r, D*Ho), attn:(N, L, r)
        out = self.fc(out.view(batch_size, -1))     # out:(N, 1)
        return out
