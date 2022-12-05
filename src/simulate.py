from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from backtesting import Backtest
import talib as ta
import pandas as pd
import numpy as np

from data.dataset import get_stock_data

class SMACrossover(Strategy):
    '''単純移動平均クロスオーバー戦略
    '''
    n1 = 5
    n2 = 75

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
        self.pre_diff = 0

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            #self.position.close()
            self.sell()
        else:
            if self.position.is_long and self.sma1[-1] - self.sma2[-1] < self.pre_diff:
                self.position.close()
            if self.position.is_short and self.sma2[-1] - self.sma1[-1] < -self.pre_diff:
                self.position.close()

        self.pre_diff = self.sma1 - self.sma2  

class MACDCrossover(Strategy):
    '''MCAD/シグナルクロスオーバー戦略
    '''
    n1 = 17  # short EMA term
    n2 = 31  # long EMA term
    ns = 9   # signal term

    def init(self):
        self.macd, self.macdsignal, _ = self.I(self._macd, self.data.Close, self.n1, self.n2, self.ns)
        self.pre_diff = 0

    def next(self):
        if crossover(self.macd, self.macdsignal):
            self.buy()

        #elif crossover(self.macdsignal, self.macd):
        #self.position.close()
        #self.sell()

        else:
            if self.position.is_long and self.macd[-1] - self.macdsignal[-1] < self.pre_diff:
                self.position.close()

        self.pre_diff = self.macd - self.macdsignal 

    def _macd(self, series, short, long, signal):
        macd, macdsignal, macdhist = ta.MACD(series, fastperiod=short, slowperiod=long, signalperiod=signal)
        return macd, macdsignal, macdhist

class AIStrategy(Strategy):
    '''AIによる戦略
    '''
    def init(self):
        self.strategy = np.array(self.data.Strategy)

    def next(self):
        if self.strategy == 1 :
            self.buy()
        elif self.strategy == 0 :
            self.position.close()


def backtest(csv_file_path):
    # Get stock data through yfinance api as a type of pandas.DataFrame 
    stock_data = pd.read_csv(csv_file_path)#, index_col='Date', parse_dates=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    bt = Backtest(data=stock_data, strategy=AIStrategy, cash=100000, trade_on_close=True,  exclusive_orders=False)
    stats = bt.run()
    #stats=bt.optimize(n1=range(10, 50, 5), n2=range(50, 100, 5), maximize='Equity Final [$]')
    #stats=bt.optimize(n1=range(5, 20, 1), n2=range(20, 32, 1), maximize='Equity Final [$]')
    print(stats)
    print(stats['_trades'])
    bt.plot()

if __name__ == '__main__':
    arg = input('CSVファイルパス:')
    backtest(arg)