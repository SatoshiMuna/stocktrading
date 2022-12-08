import logging
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from model import train
from data.dataset import get_stock_data

matplotlib.use('TkAgg')

def _make_strategy(forecast_close, today_close):
    if forecast_close > today_close:
        return 1  # buy
    else:
        return 0  # sell

def set_strategy_col(data_frame, insample_end_idx):
    strategy = ['nan'] * len(data_frame)
    for i, (idx, row) in enumerate(data_frame.iterrows()):
        if i > insample_end_idx and i < len(data_frame):
            strategy[i-1] = _make_strategy(data_frame.iloc[i]['Forecast'], data_frame.iloc[i-1]['Close'])
    df_out = data_frame.assign(Strategy=strategy)
    return df_out

def main(stock_code, data_start, data_end, insample_end, exec_train):
    logging.basicConfig(filename='stocktrading.log', level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    sns.set_theme(context='paper', style='darkgrid', palette='Set2', font_scale=1.5, rc={'figure.figsize': (16, 6), 'grid.linestyle': 'dashdot'})
    
    # Get stock info from yfinance as type of pandas.DataFrame
    stock_data = get_stock_data(stock_code, data_start, data_end, use_cols=['Open', 'High', 'Low', 'Close'])
    insample_end_idx = stock_data.index.get_loc(insample_end)

    # Training network
    trainer = train.NetworkTrainer(stock_data, insample_end_idx, bidirectional=True, r=1)
    if exec_train == 'y':
        train_loss = trainer.do_train(epoch=30)
        # Plot train loss 
        tl = pd.Series(train_loss)
        tl.plot(title='training-loss')
        plt.show()
    
    # Out-of-sample test    
    df_fcsts, mape, rmse = trainer.do_test()
    print(f'mape:{mape}, rmse:{rmse}')

    # Make buy/sell strategy and csv output
    df_out = set_strategy_col(df_fcsts, insample_end_idx)
    df_out.to_csv('forecast.csv')
    # Visualize
    diffs = df_out['Forecast'] - df_out['Close']
    df = df_out.assign(Difference=diffs)
    ax = df[['Close', 'Forecast', 'Difference']].plot(title='close-forecasts')
    ax.axvline(insample_end, color='black')
    plt.show()

if __name__ == '__main__':
    args = input('証券コード,データ開始日,データ終了日,学習終了日,学習あり(y/n):').split(',')
    main(args[0], args[1], args[2], args[3], args[4])


