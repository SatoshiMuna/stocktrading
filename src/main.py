import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from model import train
from data.dataset import get_stock_data

def _make_strategy(forecast, result, position):
    if forecast >= result:
        return 1 # buy
    else:
        return 0 # sell      

def set_strategy_col(data_frame, insample_end_idx, from_open):
    strategy = ['nan'] * len(data_frame)
    position = 0
    for i, (idx, row) in enumerate(data_frame.iterrows()):
        if from_open == False and (i >= insample_end_idx and i < len(data_frame) - 1):
            strategy[i] = _make_strategy(data_frame.iloc[i+1]['Forecast'], data_frame.iloc[i]['Close'], position)
            position = strategy[i]
        if from_open and i > insample_end_idx:
            strategy[i] = _make_strategy(data_frame.iloc[i]['Forecast'], data_frame.iloc[i]['Open'], position)
            position = strategy[i]
    df_out = data_frame.assign(Strategy=strategy)
    return df_out

def main(stock_code, data_start, data_end, insample_end, exec_train):
    logging.basicConfig(filename='stocktrading.log', level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    sns.set_theme(context='paper', style='darkgrid', palette='Set2', font_scale=1.5, rc={'figure.figsize': (14, 5), 'grid.linestyle': 'dashdot'})
    
    # Get stock info from yfinance as type of pandas.DataFrame
    stock_data = get_stock_data(stock_code, data_start, data_end, use_cols=['Open', 'High', 'Low', 'Close', 'Volume'])
    insample_end_idx = stock_data.index.get_loc(insample_end)
    open2close = False # if true, input_size/in_channels are 5

    # Training network
    trainer = train.NetworkTrainer(stock_data, insample_end_idx, input_size=4, in_channels=4, block_channels=32, window_size=32, 
                                   out_channels=64, bidirectional=True, r=2, output_size=1, from_open=open2close, prob_target=False)
    if exec_train == 'y':
        train_loss = trainer.do_train(learning_rate=0.01, batch_size=64, epoch=30)
        # Plot train loss 
        pd.Series(train_loss).plot(title='training-loss')
        plt.show()
    
    # Out-of-sample test    
    df_fcsts, mape, rmse = trainer.do_test()
    print(f'mape:{mape}, rmse:{rmse}')

    # Make buy/sell strategy and csv output
    df_out = set_strategy_col(df_fcsts, insample_end_idx, from_open=open2close)
    df_out.to_csv('forecast.csv')

    # Visualize
    diffs = df_out['Forecast'] - df_out['Close']
    df = df_out.assign(Difference=diffs)

    fig = plt.figure()    
    axes = fig.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    a1 = df[['Close', 'Forecast']].plot(ax=axes[0], title='close price forecasts')
    # ax = df[['Close', 'Forecast', 'Difference']].plot(title='close-forecasts')
    a1.axvline(insample_end, color='black')
    a2 = df[['Difference']].plot(ax=axes[1], color='blue', title='forecast errors')
    a2.axvline(insample_end, color='black')
    plt.show()

if __name__ == '__main__':
    args = input('証券コード,データ開始日,データ終了日,学習終了日,学習あり(y/n):').split(',')
    main(args[0], args[1], args[2], args[3], args[4])


