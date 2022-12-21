import logging
import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset
from data.datautils import visualize_pca_of_stockprice


def get_stock_data(stock_code, start_date, end_date, use_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
    code_info = yf.Ticker(stock_code) 
    start = start_date
    end = end_date
    # Return pandas.DataFrame(date, stock_values)
    stock_data = code_info.history(start=start, end=end)
    logging.info('Get stock data - code:%s, data start:%s, data end:%s', stock_code, start_date, end_date)
    return stock_data[use_cols]
    
class StockSeriesDataSet(Dataset):
    def __init__(self, is_train, data, window_size, columns, normalization, insample_end_idx=None, from_open=False, prob_target=False):
        super().__init__()
        # Select columns of time series to use
        df = data.iloc[:,columns[0]:columns[1]]
        if is_train:
            # Extract data during training period
            insample_data = df[:insample_end_idx + 1] / normalization
            # Get training data as torch.tensor type from a raw stock series
            self.x, self.y = self._create_input_target_data(insample_data, window_size, from_open, prob_target)
        else :
            outsample_data = df / normalization
            self.x, self.y = self._create_input_target_data(outsample_data, window_size, from_open, prob_target)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def _create_input_target_data(self, data_frame, window_size, from_open, prob_target):   
        # Get input and target data from time series
        inputs = []
        targets = []
        fcst_period = 1
        for i in range(data_frame.shape[0] - window_size + 1 - fcst_period):
            if from_open:
                # ith_prices and (i+1)th_open are used for (i+1)th_close forecasting
                row_data = []
                for j in range(window_size):
                    ith_prices = data_frame[i+j:i+j+1].values.tolist()
                    next_open = data_frame[i+j+1:i+j+2]['Open'].values
                    ith_prices[0].append(next_open[0])
                    row_data.append(ith_prices[0])
                inputs.append(torch.tensor(np.array(row_data)))
            else:
                inputs.append(torch.tensor(data_frame[i:i+window_size].to_numpy()))    

            if prob_target:
                # Give labels of rise/drop in price 
                previous_price = data_frame[i+window_size-1:i+window_size]['Close'].to_numpy()
                todays_price = data_frame[i+window_size:i+window_size+fcst_period]['Close'].to_numpy()
                vector = self._get_label(previous_price[0], todays_price[0])
                targets.append(torch.tensor(vector))
            else:
                targets.append(torch.tensor(data_frame[i+window_size:i+window_size+fcst_period]['Close'].to_numpy()))

        print(f'sample size is {len(inputs)}')
        # if prob_target: 
        #     visualize_pca_of_stockprice(inputs, targets)
        assert len(inputs)==len(targets), "Input data size and target one are different."
        return inputs, targets
    
    def _get_label(self, previous, today):
        category = [0] * 2
        category[0] = today
        # category[1] = (today - previous) / previous
        category[1] = 1.0 if today >= previous else -1.0
        return category
