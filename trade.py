'''
模拟买卖、计算费用、收益率及回撤等
'''
import os
import pandas as pd
from utils import trade_charge

if not os.path.exists('./trade_data'):
    os.mkdir('./trade_data')

holding_stock_path = './trade_data/holding_info.csv'
trade_detail_path = './trade_data/trade_detail_info.csv'


def buy_stock(code, num, price, trade_time):
    '''
    买入股票
    :param stock_code:  股票代码
    :param num:         股数
    :param price:       成交价
    :param trade_time:  交易时间
    :return:
    '''
    if os.path.exists(holding_stock_path):
        holding_df = pd.read_csv(holding_stock_path, index_col='code')
        exist_code_li = holding_df.index.tolist()
        if code in exist_code_li:  # 已建仓
            # 计算平均成本
            cost = trade_charge(code, 'b', num, price)
            avg_price = price + cost / num
            avg_price = format((avg_price * num + holding_df.loc[code, 'cost'] * holding_df.loc[code, 'num']) / (
                    num + holding_df.loc[code, 'num']), '.3f')

            # 修改数据
            holding_df.loc[code, 'num'] = num + holding_df.loc[code, 'num']  # 持仓数量
            holding_df.loc[code, 'cost'] = avg_price  # 持仓成本

            # 计算可用数量
            detail_df = pd.read_csv(trade_detail_path, index_col='idx')
            detail_df = detail_df[detail_df['code'] == code]
            sell_num = detail_df[detail_df['model'] == 's']['num'].sum()
            buy_df = detail_df[detail_df['model'] == 'b']
            buy_df = buy_df.astype({'time': 'datetime64'})
            buy_df = buy_df[buy_df['time'] < trade_time]
            buy_num = buy_df['num'].sum()
            available = buy_num - sell_num

            holding_df.loc[code, 'available'] = available

            holding_df.to_csv(holding_stock_path, index=True)
        else:
            # 建仓
            cost = trade_charge(code, 'b', num, price)
            avg_price = format(price + cost / num, '.3f')
            data = {
                'code'     : code,
                'num'      : num,
                'available': 0,
                'cost'     : avg_price,
                'time'     : trade_time
            }
            df = pd.DataFrame(data=data, columns=data.keys(), index=[0])
            df.set_index('code', inplace=True)
            holding_df = pd.concat([holding_df, df], ignore_index=False, sort=True)
            holding_df.to_csv(holding_stock_path, index=True)
    else:
        cost = trade_charge(code, 'b', num, price)
        avg_price = format(price + cost / num, '.3f')
        data = {
            'code'     : code,
            'num'      : num,
            'available': 0,
            'cost'     : avg_price,
            'time'     : trade_time
        }
        df = pd.DataFrame(data=data, columns=data.keys(), index=['code'])
        df.to_csv(holding_stock_path, index=False)

    # 记录交易明细
    record_detail_info(code, num, price, trade_time, 'b')


def record_detail_info(code, num, price, trade_time, model):
    '''
    记录交易历史信息
    :param code:        股票代码
    :param num:         成交数量
    :param price:       成交价格
    :param trade_time:  交易时间
    :param model:       s-sell, b-buy
    :return:
    '''
    taxes = trade_charge(code, model, num, price)
    data = {
        'code' : code,
        'num'  : num,
        'price': price,
        'taxes': taxes,
        'time' : trade_time,
        'model': model
    }
    if os.path.exists(trade_detail_path):
        df = pd.read_csv(trade_detail_path, index_col='idx')
        idx = len(df.index.tolist())
        cur_df = pd.DataFrame(data, columns=data.keys(), index=[idx])
        df = pd.concat([df, cur_df])
    else:
        df = pd.DataFrame(data, columns=data.keys(), index=[0])
    df.rename_axis('idx', inplace=True)
    df.to_csv(trade_detail_path, index=True)


def sell_stock(code, num, price, time):
    holding_df = pd.read_csv(holding_stock_path, index_col='code')

    if holding_df.loc[code, 'available'] < num:
        raise ValueError('卖出:{},可用数量不足！可用:{}, 卖出数量:{}'.format(code, holding_df.loc[code, 'available'], num))

    # 计算税费
    taxes = trade_charge(code, 's', num, price)

    # ===============修改持仓信息===============
    if holding_df.loc[code, 'num'] == num:  # 清仓
        holding_df.drop(code, inplace=True)
    else:
        # 数量
        rest_num = holding_df.loc[code, 'num'] - num
        holding_df.loc[code, 'num'] = rest_num
        # 可用
        holding_df.loc[code, 'available'] = holding_df.loc[code, 'available'] - num
        # 时间
        holding_df.loc[code, 'time'] = time
        # 成本
        cost = holding_df.loc[code, 'cost']
        gain = (price - cost) * num - taxes
        cost = format((cost * rest_num - gain) / rest_num, '.3f')
        holding_df.loc[code, 'cost'] = cost

    holding_df.to_csv(holding_stock_path, index=True)

    # 记录交易明细
    record_detail_info(code, num, price, time, 's')


def maximum_drawdown(price_li):
    '''
    计算最大回撤率
    :param price_li: 价格序列
    :return:         回撤率
    '''
    drawdown = []
    for idx, price in enumerate(price_li):
        if idx < len(price_li) - 2:
            min_price = min(price_li[idx + 1:])
            if min_price >= price:
                continue
            else:
                drawdown.append('%.2f%%' % ((price - min_price) * 100 / price))
    return max(drawdown)


def conservative_earning_ratio():
    '''

    :return:
    '''
    pass


if __name__ == '__main__':
    price_li = [30, 20, 25, 10, 20, 15, 30, 20, 40, 30]
    price_li = [10, 20, 15, 30, 20, 40, 20]
    rst = maximum_drawdown(price_li)
    print(rst)
    # buy_stock('000001.SZ', 500, 20.0, '2021-07-28 14:30:00')
    # buy_stock('601919.SH', 1000, 10.0, '2021-07-29 14:00:00')
    # buy_stock('000001.SZ', 500, 18.0, '2021-07-29 14:35:00')
    #
    # sell_stock('000001.SZ', 400, 22, '2021-07-30 10:00:00')
