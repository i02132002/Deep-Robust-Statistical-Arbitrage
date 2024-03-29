import torch
from pairs_trading import Net_independent, load_data, stat_arb_success, stat_arb_success_buy_and_hold_only_once
from time import time
from time import sleep
import random


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
        print("***New data from exchange received***")
        start_time = time()
        record, total_gain, total_cost = stat_arb_success(stock_test, model)
        end_time = time()
        if (total_gain > total_cost):
            print(f"Time taken to execute trade: {end_time - start_time:.4f} seconds")
            print(f"BUY XOM for $X")
            print("Total Gain:", total_gain, "Total Cost:", total_cost)
            print(f"Trade executed for a profit of {total_gain - total_cost:.2f} Arbitrage is successful!")
            stat_arb_success_buy_and_hold_only_once(stock_test)
        else:
            print("No trade")
        print('-'*50)

if __name__ == '__main__':
    main()
