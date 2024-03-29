import torch
from pairs_trading import Net_independent, load_data, stat_arb_success

def main():
    model = Net_independent(10,2)
    model.load_state_dict(torch.load('pairs_model.pth'))
    model.eval()
    asset_list = load_data()
    test_begin_day = 4500
    test_end_day = 6500
    stock_test = [asset[test_begin_day:test_end_day] for asset in asset_list]
    record = stat_arb_success(stock_test, model)
    print(record)

if __name__ == '__main__':
    main()
