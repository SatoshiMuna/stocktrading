from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from backtesting import Backtest
import talib as ta

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

def backtest(stock_code, data_start, data_end, insample_end):
    # Get stock data through yfinance api as a type of pandas.DataFrame 
    stock_data = get_stock_data(stock_code, data_start, data_end)
    insample_end_idx = stock_data.index.get_loc(insample_end)

    bt = Backtest(data=stock_data, strategy=MACDCrossover, cash=100000, trade_on_close=True,  exclusive_orders=False)
    stats = bt.run()
    #stats=bt.optimize(n1=range(10, 50, 5), n2=range(50, 100, 5), maximize='Equity Final [$]')
    #stats=bt.optimize(n1=range(5, 20, 1), n2=range(20, 32, 1), maximize='Equity Final [$]')
    print(stats)
    print(stats['_trades'])
    bt.plot()

if __name__ == '__main__':
    args = input('証券コード,データ開始日,データ終了日,学習終了日:').split(',')
    backtest(args[0], args[1], args[2], args[3])