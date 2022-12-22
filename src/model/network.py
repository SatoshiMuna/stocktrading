import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set device & type
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64 if torch.cuda.is_available() else torch.float32

class SeriesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.D = 2 if bidirectional else 1

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        #self.fc = nn.Linear(in_features=input_length * hidden_size, out_features=output_size)
        self.fc_last = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype)  # h0:(D*nl,N,Hi)
        out, h_n = self.rnn(x, h_0)  # out:(N,L,D*Ho)
        #out = out.reshape(out.shape[0],-1) # In case of using all hidden layers output 
        #out = self.fc(out)
        out = self.fc_last(out[:,-1,:])    # In case of using only the last hidden layer output
        return out

class SeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.D = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size*self.D, out_features=output_size)
        
    def forward(self, x):
        h_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype) # h0:(D*nl,N,Hi)
        c_0 = torch.zeros(self.D*self.num_layers, x.size(0), self.hidden_size).to(device, dtype) # h0:(D*nl,N,Hc)
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
        attn_weights = F.softmax(scores, dim=1)
        out = torch.matmul(torch.transpose(attn_weights, 1, 2), rnn_out) # out:(N, r, L)*(N, L, D*Ho)→(N, r, D*Ho)
        return out, attn_weights

class SelfAttnSeriesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dim_w1, dim_w2, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dim_h_out = hidden_size * 2 if bidirectional else hidden_size
        self.da = dim_w1
        self.r = dim_w2
        self.attn_weights = None
        self.D = 2 if bidirectional else 1
        
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.attn = SelfAttention(dim_h_out=self.dim_h_out, dim_w1=self.da, dim_w2=self.r)
        self.fc = nn.Linear(self.r*self.dim_h_out, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size).to(device, dtype)
        out, h_n = self.rnn(x, h_0)              # out:(N, L, D*Ho)
        out, self.attn_weights = self.attn(out)  # out:(N, r, D*Ho), attn:(N, L, r)
        out = self.fc(out.view(batch_size, -1))  # out:(N, 1)
        return out

class SelfAttnSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dim_w1, dim_w2, output_size):
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
        self.fc = nn.Linear(self.r*self.dim_h_out, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size).to(device, dtype)
        c_0 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size).to(device, dtype)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))  # out:(N, L, D*Ho)
        out, self.attn_weights = self.attn(out)     # out:(N, r, D*Ho), attn:(N, L, r)
        out = self.fc(out.view(batch_size, -1))     # out:(N, 1)
        return out

class DilatedConvLayer(nn.Module):
    def __init__(self, num_channels, width_dilation):
        super().__init__()
        self.conv_t = nn.Conv1d(num_channels, num_channels, kernel_size=2, stride=1, dilation=width_dilation, bias=False)  # (N,C,L)→(N,C,L-d) 
        # self.conv_s = nn.Conv1d(num_channels, num_channels, kernel_size=2, stride=1, dilation=width_dilation, bias=False) 
        # self.tanh = nn.Tanh()
        # self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # tanh_out = self.tanh(self.conv_t(x))
        # sigm_out = self.sigm(self.conv_s(x))
        # z = torch.mul(tanh_out, sigm_out)
        z = self.relu(self.conv_t(x))
        return z

class ResidualBlock(nn.Module):
    def __init__(self, num_channels, window_size):
        super().__init__()
        conv_layers = []
        num_stacks = math.floor(math.log2(window_size))
        out_length = window_size - 2**num_stacks + 1
        for i in range(num_stacks):
            width_dilation = 2**i  # dilation for width direction
            conv_layers.append(DilatedConvLayer(num_channels, width_dilation)) 
        self.dilated_conv = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(out_length, window_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.dilated_conv(x)  # (N,C,L)→(N,C,out_len)
        x = self.relu(self.fc(x)) # (N,C,out_len)→(N,C,L)
        z = x + identity
        return z

class DilatedConvResNet(nn.Module):
    def __init__(self, in_channels, block_channels, out_channels, window_size, num_blocks, output_size):
        super().__init__()
        res_layers = []
        self.conv_st = nn.Conv1d(in_channels, block_channels, kernel_size=1, stride=1, bias=False) # (N,C=4,L)→(N,C',L)
        for i in range(num_blocks):
             res_layers.append(ResidualBlock(block_channels, window_size))
        self.res_blocks = nn.Sequential(*res_layers)
        self.conv_ed = nn.Conv1d(block_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.avgpool = nn.AvgPool1d(kernel_size=window_size, stride=1)
        # self.conv_fn = nn.Conv1d(out_channels, 1, kernel_size=1, stride=1, bias=True)
        self.fc = nn.Linear(out_channels, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))  # (N, L, C)→(N, C, L)
        x = self.relu(self.conv_st(x))
        x = self.res_blocks(x)
        x = self.relu(self.conv_ed(x))
        x = self.avgpool(x)
        # x = self.relu(self.conv_fn(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
