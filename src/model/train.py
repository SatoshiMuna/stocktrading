import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader

from data.dataset import StockSeriesDataSet, get_stock_info
from model.network import device, StockSeriesRNN, StockSeriesLSTM

def do_train(stock_code, start_date, end_date, insample_end_date):
    torch.manual_seed(1)
    
    # Hyper parameters
    INPUT_SIZE = 4
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    WINDOW_SIZE = 14
    OUTPUT_SIZE = 1
    FCST_PERIOD = 1
    COL_START = 0  
    COL_END   = 4   # 0:Open, 1:High, 2:Low, 3:Close, 4:Volume

    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    EPOCH = 5

    # Get stock info from yfinance
    stock_data = get_stock_info(stock_code, start_date, end_date)

    # Training data
    train_dataset = StockSeriesDataSet(stock_data, WINDOW_SIZE, FCST_PERIOD, COL_START, COL_END, is_train=True, insample_end_date=insample_end_date) 
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    # Initialize network
    #net = StockSeriesRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, WINDOW_SIZE, OUTPUT_SIZE, FCST_PERIOD).to(device)
    net = StockSeriesLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, WINDOW_SIZE, OUTPUT_SIZE, FCST_PERIOD).to(device)#.to(dtype=torch.float64, device=device)
    #summary(net, input_size=(BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE))

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    # Train network
    train_loss = []
    loss = None
    for e in range(EPOCH):
        for batch_idx, (x, y) in enumerate(train_loader):    
            if not torch.cuda.is_available():
                x = x.float()
                y = y.float()
            # Get data to CUDA if possible
            x = x.to(device=device)
            y = y.to(device=device)  
            #print(x.shape,y.shape)
            # Forward
            y_pred = net(x)
            #print(model.state_dict())
            #print(output.shape, y.shape)
            loss = criterion(y, y_pred)    

            # Backward and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch:{e}, loss:{loss.item()}")
        train_loss.append(loss.item())

    return train_loss
