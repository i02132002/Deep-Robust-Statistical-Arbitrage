import torch
from pairs_trading import Net_independent, load_data, stat_arb_success, stat_arb_success_buy_and_hold_only_once

def main():
    model = Net_independent(10,2).cuda()
    model.load_state_dict(torch.load('pairs_model.pth'))
    model.eval()
    asset_list = load_data()
    test_begin_day = 4500
    test_end_day = 6500
    stock_test = [asset[test_begin_day:test_end_day] for asset in asset_list]
    record, total_gain, total_cost = stat_arb_success(stock_test, model)
    if (total_gain > total_cost):
        print(f"Trade executed for a profit of {total_gain - total_cost} Arbitrage is successful!")
    print(stat_arb_success_buy_and_hold_only_once(stock_test))

if __name__ == '__main__':
    main()
