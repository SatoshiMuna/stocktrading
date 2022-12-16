import logging
import math
import yfinance as yf
import torch
from torch.utils.data import Dataset
from data.datautils import visualize_pca_of_stockprice


def get_stock_data(stock_code, start_date, end_date, use_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
    code_info = yf.Ticker(stock_code) 
    start = start_date
    end = end_date
    # Return pandas.DataFrame(date, stock_values)
    stock_data = code_info.history(start=start, end=end)
    logging.info('Get stock data - code:%s, company name:%s, data start:%s, data end:%s', stock_code, code_info.info['shortName'], start_date, end_date)
    return stock_data[use_cols]
    
class StockSeriesDataSet(Dataset):
    def __init__(self, is_train, data, window_size, fcst_period, columns, normalization, insample_end_idx=None, prob_target=False):
        super().__init__()
        # Select columns of time series to use
        df = data.iloc[:,columns[0]:columns[1]]
        if is_train:
            # Extract data during training period
            insample_data = df[:insample_end_idx + 1] / normalization
            # Get training data as torch.tensor type from a raw stock series
            self.x, self.y = self._create_input_target_data(insample_data, window_size, fcst_period, prob_target)
        else :
            outsample_data = df / normalization
            self.x, self.y = self._create_input_target_data(outsample_data, window_size, fcst_period, prob_target)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def _create_input_target_data(self, data_frame, window_size, fcst_period, prob_target):   
        # Get input and target data from time series
        inputs = []
        targets = []
        for i in range(data_frame.shape[0] - window_size + 1 - fcst_period):
            inputs.append(torch.tensor(data_frame[i:i+window_size].to_numpy()))
            if prob_target:
                # Price distribution by 1% unit bins
                previous_price = data_frame[i+window_size-1:i+window_size]['Close'].to_numpy()
                todays_price = data_frame[i+window_size:i+window_size+fcst_period]['Close'].to_numpy()
                fluctuation_ratio = (todays_price - previous_price) * 100.0 / previous_price
                onehot = self._get_class_label(fluctuation_ratio)
                targets.append(torch.tensor(onehot))
            else:
                targets.append(torch.tensor(data_frame[i+window_size:i+window_size+fcst_period]['Close'].to_numpy()))

        #print(inputs[0], targets[0])
        print(f'sample size is {len(inputs)}')
        if prob_target: 
            visualize_pca_of_stockprice(inputs, targets)
        assert len(inputs)==len(targets), "Input data size and target one are different."
        return inputs, targets
    
    def _get_class_label(self, fluctuation_ratio):
        category = [0] * 2
        size = len(category)  # size=2
        position = math.ceil(fluctuation_ratio)
        # when fluctuation is more than plus/minus 30%, it is counted in '30%' category due to limit high/low 
        # if position >= 30:
        #     category[size-1] = 1
        #     return category
        
        # if position < -30:
        #     category[0] = 1
        #     return category
        
        # index = position + 30 - 1
        # category[index] = 1
        # return category

        if position >= 0:
            category[1] = 1.0
        # elif position >= 0 and position < 3:
        #     category[2] = 1
        # elif position >= -3 and position < 0:
        #     category[1] = 1
        else:
            category[0] = 1.0
        return category
