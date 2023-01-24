import sys
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.linalg as la
from torch.utils.data import DataLoader
from torchinfo import summary

from data.dataset import StockSeriesDataSet
from model.network import device, dtype, SeriesRNN, SeriesLSTM, SelfAttnSeriesLSTM, SelfAttnSeriesRNN, DilatedConvResNet

class NetworkTrainer:
    def __init__(self, stock_data, insample_end_idx, input_size=4, in_channels=4, block_channels=32,
                 hidden_size=128, num_layers=1, num_blocks=10, window_size=32, out_channels=64, output_size=1,
                 bidirectional=False, da=64, r=1, pc=0.01, fcst_period=1, from_open=False, prob_target=False,
                 seed=1):
        self.stock_data = stock_data
        self.insample_end_idx = insample_end_idx
        self.input_size = input_size
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.out_channels = out_channels
        self.output_size = output_size
        self.fcst_period = fcst_period
        self.from_open = from_open
        self.prob_target = prob_target
        self.penalty_coef = pc
        # 0:Open, 1:High, 2:Low, 3:Close, 4:Volume
        if input_size == 4:   
            self.columns = [0, 4]
        elif input_size == 5: # In case of using Open price for forecasting
            self.columns = [0, 4]
        else:
            self.columns = [3, 4]  # 3:Close
        
        # Normalization factor is a max value of 4 stock prices in training data
        self.normalization = stock_data.iloc[:insample_end_idx+1, 0:4].max().max() 

        # Initialize network
        torch.manual_seed(seed)
        # self.net = SeriesRNN(input_size, hidden_size, num_layers, bidirectional, output_size)
        #self.net = SeriesLSTM(input_size, hidden_size, num_layers, bidirectional, output_size)
        # self.net = SelfAttnSeriesRNN(input_size, hidden_size, num_layers, bidirectional, da, r, output_size)
        self.net = SelfAttnSeriesLSTM(input_size, hidden_size, num_layers, bidirectional, da, r, output_size)
        #self.net = DilatedConvResNet(in_channels, block_channels, out_channels, window_size, num_blocks, output_size)
        #summary(self.net, input_size=(64, window_size, input_size))
        
    def do_train(self, learning_rate=0.01, batch_size=32, epoch=5):
        # Training data
        train_dataset = StockSeriesDataSet(True, self.stock_data, self.window_size, self.columns, self.normalization,
                                           self.insample_end_idx, self.from_open, self.prob_target) 
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(params=self.net.parameters(), lr=learning_rate)

        # Train network
        logging.info('Start Training - size:%s, epoch:%s, batch:%s, model:%s', len(train_dataset), epoch, batch_size, type(self.net))
        self.net.to(device, dtype) 
        self.net.train()
        train_loss = []
        loss = None
        pre_loss = sys.float_info.max
        for e in range(epoch):
            for idx, (x, y) in enumerate(train_loader):    
                # Transfer data to CUDA if possible
                x = x.to(device, dtype)
                y = y.to(device, dtype) 
                # Forward
                y_pred = self.net(x)
                #print(self.net.state_dict())
                if (isinstance(self.net, SelfAttnSeriesRNN) or isinstance(self.net, SelfAttnSeriesLSTM)) and self.net.r > 1:
                    p = torch.matmul(torch.transpose(self.net.attn_weights, 2, 1), self.net.attn_weights) - torch.eye(self.net.r)
                    loss = criterion(y, y_pred) + self.penalty_coef * la.norm(p)  
                else:
                    loss = criterion(y, y_pred)    
                # Backward and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            num_loss = loss.item()
            print(f"epoch:{e}, loss:{num_loss}")
            train_loss.append(num_loss)
            logging.info('epoch:%s, loss:%s, model-save:%s', e, num_loss, num_loss<pre_loss)
            if num_loss < pre_loss:
                torch.save(self.net.state_dict(), 'learned_model.pth')
                pre_loss = num_loss

        return train_loss

    def do_test(self, fcst_col_idx=3):
        self.net.load_state_dict(torch.load('learned_model.pth'))
        self.net.to(device, dtype)  
        self.net.eval()
        # Test data
        test_dataset = StockSeriesDataSet(False, self.stock_data, self.window_size, self.columns, self.normalization,
                                          None, self.from_open, self.prob_target)
        logging.info('Start Out-of-sample Test - size:%s, model:%s', len(test_dataset), type(self.net))

        mape = []
        rmse = []
        close_fcsts = test_dataset[0][0][:,fcst_col_idx].numpy().copy() * self.normalization
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataset):
                x = x.unsqueeze(0).to(device, dtype)  # (1, window_size, input_size)
                y = y.to(device, dtype)
                y_pred = self.net(x)
                y_pred = y_pred.squeeze(0)
                if self.prob_target:
                    y2 = y[1]
                    y_pred2 = y_pred[1]
                    y = y[0]
                    y_pred = y_pred[0]
                    print(y2.to('cpu').numpy().copy(), y_pred2.to('cpu').numpy().copy())
                if idx > self.insample_end_idx - self.window_size:  # when true, y and y_pred are in the forecast period
                    mape.append(torch.abs((y-y_pred)/y).to('cpu').numpy().copy())
                    rmse.append(torch.pow((y-y_pred)*self.normalization, 2).to('cpu').numpy().copy())
                close_fcsts = np.append(close_fcsts, y_pred.to('cpu').numpy().copy() * self.normalization)

        mape = np.mean(mape)
        rmse = np.sqrt(np.mean(rmse))
        logging.info('Out-of-sample Test Result - mape:%s, rmse:%s', mape, rmse)
        df = self.stock_data.assign(Forecast=close_fcsts)
        return df, mape, rmse

