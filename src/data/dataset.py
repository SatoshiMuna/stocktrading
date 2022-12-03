import logging
import yfinance as yf
import torch
from torch.utils.data import Dataset

def get_stock_info(sotck_code, start_date, end_date):
    code_info = yf.Ticker(sotck_code)  # Tokyo Electron
    start = start_date
    end = end_date
    # Return pandas.DataFrame(date, stock_values)
    stock_data = code_info.history(start=start, end=end)
    logging.info('company name:%s, data start:%s, data end:%s', code_info.info['shortName'], start_date, end_date)
    return stock_data
class StockSeriesDataSet(Dataset):
    def __init__(self, data, window_size, fcst_period, col_start, col_end, is_train, insample_end_date=None, denormalize=None):
        super().__init__()
        # Select columns of time series to use
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = data[cols[col_start:col_end]]
        self.insample_end_index = df.index.get_loc(insample_end_date) if insample_end_date is not None else None
        if is_train:
            # Extract data during training period
            insample_data = df[:self.insample_end_index + 1]
            # Get training data as torch.tensor type from a raw stock series
            self.x, self.y, self.denormalize = self._create_input_target_data(insample_data, window_size, fcst_period)
        else :
            self.x, self.y, _ = self._create_input_target_data(df, window_size, fcst_period, denormalize)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def _create_input_target_data(self, data_frame, window_size, fcst_period, normalize_factor=None):
        # Normalize data
        if normalize_factor is None:
            normalize = data_frame.max().max()
            df = data_frame / normalize  
        else:
            normalize = None
            df = data_frame / normalize_factor    
        # Get input and target data from time series
        inputs = []
        targets = []
        for i in range(df.shape[0] - window_size + 1 - fcst_period):
            inputs.append(torch.tensor(df[i:i+window_size].to_numpy()))
            targets.append(torch.tensor(df[i+window_size:i+window_size+fcst_period]['Close'].to_numpy()))

        #print(inputs[0], targets[0])
        assert len(inputs)==len(targets), "Input data size and target one are different."
        return inputs, targets, normalize