import logging
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from model import train
from data.dataset import get_stock_data

matplotlib.use('TkAgg')

def main(stock_code, data_start, data_end, insample_end):
    logging.basicConfig(filename='stocktrading.log', level=logging.INFO)
    sns.set_theme(context='paper', style='darkgrid', palette='Set2', font_scale=1.5, rc={'figure.figsize': (16, 6), 'grid.linestyle': 'dashdot'})
    # Get stock info from yfinance as type of pandas.DataFrame
    stock_data = get_stock_data(stock_code, data_start, data_end)
    insample_end_idx = stock_data.index.get_loc(insample_end)

    # Training network
    trainer = train.NetworkTrainer(stock_data, insample_end_idx)
    train_loss = trainer.do_train()
    # Out-of-sample test    
    rslts, fcsts, mape, rmse = trainer.do_test()

    # Plot train loss 
    tl = pd.Series(train_loss)
    tl.plot(title='training-loss')
    plt.show()

    rf = pd.DataFrame({'forecast':fcsts, 'result':rslts, 'difference':fcsts-rslts})
    print(rf[insample_end_idx:])
    print(f'mape:{mape}, rmse:{rmse}, forecast_index:{insample_end_idx+1}')
    ax = rf.plot()
    ax.vlines(insample_end_idx,0,rslts.max())
    plt.show()

if __name__ == '__main__':
    args = input('証券コード,データ開始日,データ終了日,学習終了日:').split(',')
    main(args[0], args[1], args[2], args[3])


