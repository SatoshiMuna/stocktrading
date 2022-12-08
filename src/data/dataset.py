import logging
import yfinance as yf
import torch
from torch.utils.data import Dataset

def get_stock_data(stock_code, start_date, end_date, use_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
    code_info = yf.Ticker(stock_code) 
    start = start_date
    end = end_date
    # Return pandas.DataFrame(date, stock_values)
    stock_data = code_info.history(start=start, end=end)
    logging.info('Get stock data - code:%s, company name:%s, data start:%s, data end:%s', stock_code, code_info.info['shortName'], start_date, end_date)
    return stock_data[use_cols]
    
class StockSeriesDataSet(Dataset):
    def __init__(self, is_train, data, window_size, fcst_period, columns, normalization, insample_end_idx=None):
        super().__init__()
        # Select columns of time series to use
        df = data.iloc[:,columns[0]:columns[1]]
        if is_train:
            # Extract data during training period
            insample_data = df[:insample_end_idx + 1] / normalization
            # Get training data as torch.tensor type from a raw stock series
            self.x, self.y = self._create_input_target_data(insample_data, window_size, fcst_period)
        else :
            outsample_data = df / normalization
            self.x, self.y = self._create_input_target_data(outsample_data, window_size, fcst_period)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def _create_input_target_data(self, data_frame, window_size, fcst_period):   
        # Get input and target data from time series
        inputs = []
        targets = []
        for i in range(data_frame.shape[0] - window_size + 1 - fcst_period):
            inputs.append(torch.tensor(data_frame[i:i+window_size].to_numpy()))
            targets.append(torch.tensor(data_frame[i+window_size:i+window_size+fcst_period]['Close'].to_numpy()))

        #print(inputs[0], targets[0])
        assert len(inputs)==len(targets), "Input data size and target one are different."
        return inputs, targets