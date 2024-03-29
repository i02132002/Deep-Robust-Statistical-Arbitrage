import torch
from pairs_trading import Net_independent, load_data, stat_arb_success, STOCK_TICKERS
from time import time
from time import sleep
import random
import numpy as np
from datetime import datetime

def main():
    print("Loading model...")
    model = Net_independent(10,2).cuda()
    model.load_state_dict(torch.load('pairs_model.pth'))
    model.eval()
    print("Model loaded!")
    asset_list = load_data()
    start = 0
    stop = 3000
    sleep_time = random.uniform(0, 0.3)
    for i in range(start, stop, 100):
        test_begin_day = 4500 + i
        test_end_day = 5000 + i
        stock_test = [asset[test_begin_day:test_end_day] for asset in asset_list]
        sleep(sleep_time)
        current_datetime = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_datetime}] ***New data from exchange received***")
        start_time = time()
        record, total_gain, total_cost, stock_profits, stock_costs = stat_arb_success(stock_test, model)
        end_time = time()
        random_value = random.choices([0, 1], weights=[0.3, 0.7])[0]
        if (total_gain > total_cost) and random_value:
            buy_stock = np.argmax(stock_profits)
            sell_stock = np.argmin(stock_profits)
            current_datetime = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_datetime}] ({int((end_time - start_time)*100)} ms) BUY {STOCK_TICKERS[buy_stock]} at {stock_costs[buy_stock]:.3f} SELL {STOCK_TICKERS[sell_stock]} at {stock_costs[sell_stock]:.3f}, projected profit: {total_gain - total_cost:.3f}")

if __name__ == '__main__':
    main()
