### STOCK TRADING
This is a sample program for daily stock price forecasting with using AI and evaluation of trading strategies based on its forecasts.
<br>
The purpose of this site is to open personal learning outcomes of algorithmic trading.


**Warning**

This site is NOT intended to solicit investment in or recommend purchase or sale of specific products. 
<br>
The program does NOT ensure forecasting accuracy and trading performance!
<br>
このサイトでは株価予測のプログラムを公開しておりますが、
投資や特定の金融商品の購入を推奨するものではありません。<br>
プログラムの予測精度や予測を用いた投資収益は保証しません。

### FEATURES
* Main four types of time series - Open, High, Low, Close prices are utilized for model input.
* One-step ahead prediction of close price will be output.
* Trading strategy is tested by backtesting.py

### REQUIREMENTS
* PyTorch
* yfinance
* backtesting
* numpy
* pandas

### SIMPLE EXPERIMENTS
#### 1.Training Settings
* Train each model with using over 10 years daily stock prices including Open, High, Low and Close.
* Input window size is 32 and target length is 1 because it is the close price of tomorrow or today.
* Inputs and targets are extracted sequencially from a stock time series, then the supervised data size is 2,455.
* Batch size is 64, the number of epoch is 30, loss function is MSE, and optimizer is Adam.
#### 2.Out-of-sample Tests
* Firstly, predict a tomorrow's close price everyday when we have got the today's one during the forecast horizon (for 10 months).
* Secondaly, predict a today's close price everyday when we have got the today's open price during the forecast horizon (for 10 months).  
* Forecasting performance is evaluated by MAPE and RMSE.
#### 3.Position Strategy
* Take a long position (buy stocks) and keep it when the predicted price is larger than the today's one (1st oos test).
* Close the position when the predicted price is smaller than the today's one (1st oos test).
* Take a long/short position based on the predition at the time to open and close it at the end of daily trade (2nd oos test).  

#### 4.Results (Stock code: 6501 TSE)
・1st oos test (predict the tomorrow's close price)
|  model               |  MAPE  |  RMSE    |  PROFIT   |
|  :----:              | :----: | :----:   |   ----:   |
|  LSTM                | 0.0154 | 128.9547 |   1,594   |
|  SelfAttnLSTM(r=1)*  | 0.0194 | 154.6140 |   4,318   |
|  SelfAttnLSTM(r=2)   | 0.0158 | 132.6468 |   13,870  |
|  SelfAttnLSTM(r=5)   | 0.0167 | 136.3996 |   16,645  |
|  DilatedConvResNet** | 0.0176 | 139.2039 |   8,675   |

\* LSTM with self-attention layer. Parameter 'r' means the number of attention vectors.
<br>
** ResNet with dilated convolutional layers as a residual block. 

With regard to the model of SelfAttnLSTM(r=2), the graphs of training loss, forecasts and backtesting output (trading performance) are shown below.
![training loss of SelfAttnLSTM(r=2)](https://github.com/SatoshiMuna/stocktrading/blob/main/SelfAttnLSTM(r%3D2)_loss.png)
![forecast result of SelfAttnLSTM(r=2)](https://github.com/SatoshiMuna/stocktrading/blob/main/SelfAttnLSTM(r%3D2)_forecasts.png)
![backtesting output of SelfAttnLSTM(r=2)](https://github.com/SatoshiMuna/stocktrading/blob/main/SelfAttnLSTM(r%3D2).png)


・2nd oos test  (predict today's close price)
|  model               |  MAPE  |  RMSE    |  PROFIT***|
|  :----:              | :----: | :----:   |   ----:   |
|  LSTM                | 0.0137 | 112.2655 |   1,193   |
|  SelfAttnLSTM(r=1)   | 0.0156 | 123.5559 |     318   |
|  SelfAttnLSTM(r=2)   | 0.0152 | 122.5174 |    -191   |
|  SelfAttnLSTM(r=5)   | 0.0168 | 134.8579 |  -1,108   |
|  DilatedConvResNet   | 0.0147 | 115.2771 |  -2,322   |

*** The way to calculate the PROFIT is different from one of the 1st test. 

#### Results (Stock code: 8035 TSE)
・1st oos test (predict the tomorrow's close price)
|  model               |  MAPE  |  RMSE     |  PROFIT   |
|  :----:              | :----: |  :----:   |   ----:   |
|  LSTM                | 0.0218 | 1321.3167 |  -16,707  |
|  SelfAttnLSTM(r=1)   | 0.0228 | 1404.6824 |    1,067  |
|  SelfAttnLSTM(r=2)   | 0.0214 | 1320.1464 |    2,023  |
|  SelfAttnLSTM(r=5)   | 0.0386 | 2221.0750 |   -9,367  |
|  DilatedConvResNet   | 0.0266 | 1598.9781 |      741  |

・2nd oos test  (predict today's close price)
|  model               |  MAPE  |  RMSE     |  PROFIT   |
|  :----:              | :----: |  :----:   |   ----:   |
|  LSTM                | 0.0192 | 1185.0369 |   3,515   |
|  SelfAttnLSTM(r=1)   | 0.0295 | 1701.9565 |     905   |
|  SelfAttnLSTM(r=2)   | 0.0198 | 1188.1904 |  19,089   |
|  SelfAttnLSTM(r=5)   | 0.0237 | 1477.9225 |  12,425   |
|  DilatedConvResNet   | 0.0351 | 1951.5614 |  10,681   |

#### 5. Additive Results
* Price forecasting and trading performance for 20 companies' stocks in various industries of TSE(Prime).
* The SelfAttnLSTM(r=2) model is utilized.

・1st oos test (predict the tomorrow's close price)
|  stock code  |  industry | MAPE  |   RMSE   |   PROFIT  |
|  :----:      |  :----:   | :----:|  :---:   |   ----:   |
|1925|construction|0.022390258|79.371544|-3,701|
|2502|food|0.016816227|103.79857|-340|
|2802|food|0.012737455|60.882423|18,957|
|3382|retail|0.016506597|116.24326|15,581|
|3402|fiber|0.07663136|58.872154|19,240|
|3861|pulp/paper|0.018936943|12.592528|-7,818|
|4507|pharmaceuticals|0.016800655|172.45436|-5,044|
|4751|service|0.021781694|37.863106|3,094|
|4901|chemistry|0.014572835|135.94965|754|
|4911|chemistry|0.021211086|153.65544|-12,255|
|6272|machinery|0.022701437|33.118965|146|
|6471|machinery|0.017931495|16.313107|2,385|
|6501|electromechanical|0.01582813|132.64684|13,870|
|7203|automobile|0.021113189|54.520725|-2,527|
|7741|precision machinery|0.019753909|351.32437|6,926|
|8001|trading company|0.015807664|77.08195|6,800|
|8035|electromechanical|0.021385979|1320.1464|2,023|
|8316|bank|0.02017012|96.63412|-5,194|
|9020|railroad|0.011764813|112.12646|11,618|
|9433|communication|0.012926099|67.12559|581|
|TOTAL||||63,341|
|AVERAGE||||3167.05|

・2nd oos test (predict today's close price)
|  stock code    |  industry  |  MAPE  |   RMSE   |   PROFIT  |
|  :----:        |  :----:    | :----: |  :----:  |   ----:   |
|1925|construction|0.010447112|41.158936|249|
|2502|food|0.01183756|75.209496|751|
|2802|food|0.011183262|53.97314|223|
|3382|retail|0.012955708|92.99024|84|
|3402|fiber|0.016609129|14.541413|6|
|3861|pulp/paper|0.010005164|7.529804|-78|
|4507|pharmaceuticals|0.016676743|163.84259|957|
|4751|service|0.022607347|37.07497|56|
|4901|chemistry|0.014246126|127.49055|-625|
|4911|chemistry|0.016560735|123.72586|-828|
|6272|machinery|0.018976301|28.0424|-118|
|6471|machinery|0.017675254|16.015644|144|
|6501|electromechanical|0.015159226|122.51744|-191|
|7203|automobile|0.012190495|33.517616|103|
|7741|precision machinery|0.0167388|298.49338|365|
|8001|trading company|0.012706843|63.072407|96|
|8035|electromechanical|0.019787943|1188.1904|19,089|
|8316|bank|0.009523883|51.729343|505|
|9020|railroad|0.013989773|128.0516|-103|
|9433|communication|0.014662711|72.46877|-862|
|TOTAL||||19,823|
|AVERAGE||||991.15|