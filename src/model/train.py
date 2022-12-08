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
from model.network import device, SeriesRNN, SeriesLSTM, SelfAttnSeriesLSTM, SelfAttnSeriesRNN

class NetworkTrainer:
    def __init__(self, stock_data, insample_end_idx, input_size=4, hidden_size=128, 
                 num_layers=1, window_size=14, bidirectional=False, da=100, r=1, 
                 output_size=1, fcst_period=1, seed=1):
        self.stock_data = stock_data
        self.insample_end_idx = insample_end_idx
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.output_size = output_size
        self.fcst_period = fcst_period

        if input_size == 4:   # 0:Open, 1:High, 2:Low, 3:Close
            self.columns = [0, 4]
        elif input_size == 1: # 3:Close
            self.columns = [3, 4]
        
        # Normalization factor is a max value of training data
        self.normalization = stock_data[:insample_end_idx+1].max().max() 

        # Initialize network
        torch.manual_seed(seed)
        #self.net = SeriesRNN(input_size, hidden_size, num_layers, window_size, output_size, fcst_period)
        #self.net = SeriesLSTM(input_size, hidden_size, num_layers, window_size, bidirectional, output_size, fcst_period)
        #self.net = SelfAttnSeriesRNN(input_size, hidden_size, num_layers, window_size, bidirectional, da, r, output_size, fcst_period)
        self.net = SelfAttnSeriesLSTM(input_size, hidden_size, num_layers, window_size, bidirectional, da, r, output_size, fcst_period)
        #summary(self.net, input_size=(64, window_size, input_size))
        
    def do_train(self, learning_rate=0.01, batch_size=64, epoch=5):
        # Training data
        train_dataset = StockSeriesDataSet(True, self.stock_data, self.window_size, self.fcst_period, self.columns,
                                           self.normalization, self.insample_end_idx) 
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(params=self.net.parameters(), lr=learning_rate)

        # Train network
        logging.info('Start Training - epoch:%s, model:%s', epoch, type(self.net))
        self.net.to(device)  #.to(dtype=torch.float64, device=device)
        self.net.train()
        train_loss = []
        loss = None
        pre_loss = sys.float_info.max
        for e in range(epoch):
            for batch_idx, (x, y) in enumerate(train_loader):    
                if not torch.cuda.is_available():
                    x = x.float()
                    y = y.float()
                # Transfer data to CUDA if possible
                x = x.to(device)
                y = y.to(device)  

                # Forward
                y_pred = self.net(x)
                #print(self.net.state_dict())
                if (isinstance(self.net, SelfAttnSeriesRNN) or isinstance(self.net, SelfAttnSeriesLSTM)) and self.net.r > 1:
                    p = torch.matmul(torch.transpose(self.net.attn_weights, 2, 1), self.net.attn_weights) - torch.eye(self.net.r)
                    loss = criterion(y, y_pred) + 0.01 * la.norm(p)  
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
        logging.info('Start Out-of-sample Test - model:%s', type(self.net))
        self.net.load_state_dict(torch.load('learned_model.pth'))
        self.net.to(device)  #.to(dtype=torch.float64, device=device)
        self.net.eval()
        # Test data
        test_dataset = StockSeriesDataSet(False, self.stock_data, self.window_size, self.fcst_period, self.columns, self.normalization)

        mape = []
        rmse = []
        #close_rslts = test_dataset[0][0][:,fcst_col_idx].numpy().copy() * self.denormalize_factor
        close_fcsts = test_dataset[0][0][:,fcst_col_idx].numpy().copy() * self.normalization
        with torch.no_grad():
            for itr, (x, y) in enumerate(test_dataset):
                if not torch.cuda.is_available():
                    x = x.float()
                    y = y.float()
                x = x.unsqueeze(0).to(device=device)  # (1, window_size, input_size)
                y = y.to(device=device)
                y_pred = self.net(x)
                
                if itr > self.insample_end_idx:
                    mape.append(torch.abs((y-y_pred[0])/y).to('cpu').numpy().copy())
                    rmse.append(torch.pow((y-y_pred[0])*self.normalization, 2).to('cpu').numpy().copy())

                #close_rslts = np.append(close_rslts, y.to('cpu').numpy().copy() * self.denormalize_factor)
                close_fcsts = np.append(close_fcsts, y_pred.to('cpu').numpy().copy() * self.normalization)

        mape = np.mean(mape)
        rmse = np.sqrt(np.mean(rmse))
        logging.info('Out-of-sample Test Result - mape:%s, rmse:%s', mape, rmse)
        df = self.stock_data.assign(Forecast=close_fcsts)
        return df, mape, rmse

