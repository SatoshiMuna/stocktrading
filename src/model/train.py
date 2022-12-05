import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader

from data.dataset import StockSeriesDataSet
from model.network import device, StockSeriesRNN, StockSeriesLSTM

class NetworkTrainer:
    def __init__(self, stock_data, insample_end_idx, input_size=4, hidden_size=128, 
                 num_layers=1, window_size=14, output_size=1, fcst_period=1, seed=1):
        self.stock_data = stock_data
        self.insample_end_idx = insample_end_idx
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.output_size = output_size
        self.fcst_period = fcst_period

        if input_size == 4:   # 0:Open, 1:High, 2:Low, 3:Close
            self.col_startidx = 0
            self.col_endidx = 4  
        elif input_size == 1: # 3:Close
            self.col_startidx = 3
            self.col_endidx = 4

        self.denormalize_factor = 1  # Value obtained by normalizing traing data 

        # Initialize network
        torch.manual_seed(seed)
        #self.net = StockSeriesRNN(input_size, hidden_size, num_layers, window_size, output_size, fcst_period)
        self.net = StockSeriesLSTM(input_size, hidden_size, num_layers, window_size, output_size, fcst_period)
        #summary(self.net, input_size=(64, window_size, input_size))
        
    def do_train(self, learning_rate=0.01, batch_size=64, epoch=5):
        # Training data
        train_dataset = StockSeriesDataSet(self.stock_data, self.window_size, self.fcst_period, self.col_startidx, self.col_endidx, is_train=True, insample_end_idx=self.insample_end_idx) 
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        self.denormalize_factor = train_dataset.denormalize

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(params=self.net.parameters(), lr=learning_rate)

        # Train network
        self.net.to(device)  #.to(dtype=torch.float64, device=device)
        self.net.train()
        train_loss = []
        loss = None
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
                loss = criterion(y, y_pred)    

                # Backward and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"epoch:{e}, loss:{loss.item()}")
            train_loss.append(loss.item())

        torch.save(self.net.state_dict(), 'learned_model.pth')
        return train_loss

    def do_test(self, fcst_col_idx=3):
        self.net.eval()
        # Test data
        test_dataset = StockSeriesDataSet(self.stock_data, self.window_size, self.fcst_period, self.col_startidx,
                                          self.col_endidx, is_train=False, denormalize=self.denormalize_factor)

        mape = []
        rmse = []
        close_rslts = test_dataset[0][0][:,fcst_col_idx].numpy().copy() * self.denormalize_factor
        close_fcsts = test_dataset[0][0][:,fcst_col_idx].numpy().copy() * self.denormalize_factor
        with torch.no_grad():
            for itr, (x, y) in enumerate(test_dataset):
                if not torch.cuda.is_available():
                    x = x.float()
                    y = y.float()
                x = x.unsqueeze(0).to(device=device)  # (1, window_size, input_size)
                y = y.to(device=device)
                y_pred = self.net(x)
                #print(y,y_pred)
                if itr > self.insample_end_idx:
                    mape.append(torch.abs((y-y_pred[0])/y).to('cpu').numpy().copy())
                    rmse.append(torch.pow((y-y_pred[0])*self.denormalize_factor, 2).to('cpu').numpy().copy())

                close_rslts = np.append(close_rslts, y.to('cpu').numpy().copy() * self.denormalize_factor)
                close_fcsts = np.append(close_fcsts, y_pred.to('cpu').numpy().copy() * self.denormalize_factor)

        mape = np.mean(mape)
        rmse = np.sqrt(np.mean(rmse))

        return close_rslts, close_fcsts, mape, rmse

