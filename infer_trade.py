import torch
from pairs_trading import Net_independent, load_data, stat_arb_success, stat_arb_success_buy_and_hold_only_once

def main():
    print("Loading model...")
    model = Net_independent(10,2).cuda()
    model.load_state_dict(torch.load('pairs_model.pth'))
    model.eval()
    print("Model loaded!")
    asset_list = load_data()
    start = 0
    stop = 6000
    for i in range(start, stop, 100):
        print("Pulling latest data from exchange...")
        test_begin_day = 4500 + i
        test_end_day = 5000 + i
        stock_test = [asset[test_begin_day:test_end_day] for asset in asset_list]
        record, total_gain, total_cost = stat_arb_success(stock_test, model)
        if (total_gain > total_cost):
            print(f"Trade executed for a profit of {total_gain - total_cost} Arbitrage is successful!")
        stat_arb_success_buy_and_hold_only_once(stock_test)

if __name__ == '__main__':
    main()
