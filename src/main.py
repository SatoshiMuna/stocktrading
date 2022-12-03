import logging
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from model import train

matplotlib.use('TkAgg')

def main(stock_code, data_start, data_end, insample_end):
    logging.basicConfig(filename='stocktrading.log', level=logging.INFO)
    sns.set_theme(context='paper', style='darkgrid', palette='Set2', font_scale=1.5, rc={'figure.figsize': (16, 6), 'grid.linestyle': 'dashdot'})
    train_loss = train.do_train(stock_code, data_start, data_end, insample_end)

    # Plot train loss 
    tl = pd.Series(train_loss)
    tl.plot(title='train-loss')
    plt.show()

if __name__ == '__main__':
    args = input('証券コード,データ開始日,データ終了日,学習終了日:').split(',')
    main(args[0], args[1], args[2], args[3])


