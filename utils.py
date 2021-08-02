import datetime
import logging
import time

import numpy as np
import pandas as pd
from chinese_calendar import is_holiday, is_workday


def get_last_trade_date(cur_date=None, interval=1, strformat='%Y%m%d'):
    '''
    获取上一个交易日期
    cur_date: 默认为当天
    interval: 时间间隔，默认为上一个交易日
    '''
    if cur_date is None:
        cur_date = datetime.datetime.today()
    elif isinstance(cur_date, str):
        cur_date = datetime.datetime.strptime(cur_date, strformat).date()
    elif isinstance(cur_date, time.struct_time):
        cur_date = datetime.datetime(*cur_date[:6])

    last_date = cur_date - datetime.timedelta(days=interval)

    if is_holiday(last_date) or (not is_workday(last_date)):
        last_date = last_date - datetime.timedelta(days=interval)

    return last_date.strftime(strformat)


def trade_charge(stock_code, model, num, price):
    '''
    总的交易费用
    stock_code: 股票代码
    model:      b-买入, s-卖出
    num:        单位: 股
    price:      成交价格
    '''
    charge1, charge2, charge3 = 0.0, 0.0, 0.0
    amount = num * price
    # 交易费
    charge1 = amount * 2.5 / 1000 if (amount * 2.5 / 1000) >= 5 else 5

    # 印花税
    if model == 's':
        charge2 = amount * 0.1 / 100 if (amount * 0.1 / 100) >= 1 else 1

    # 过户费
    if stock_code.startswith('60'):
        charge3 = 0.1 * num / 100

    return charge1 + charge2 + charge3


def level2idx_of_multi_idx(df, level1idx, col_name, col_val):
    '''
    根据值获取多重索引的第二层索引值, 要求col_val是唯一的
    :param df:
    :param level1idx: level1的名称
    :param col_name:  列名
    :param col_val:   根据col_val获取idx
    :return:          level2idx
    '''
    df_ = df.loc[level1idx, :]
    df_ = df_.astype({'date': 'str'})
    level2idx = df_[df_[col_name] == col_val].index.tolist()[0]
    return level2idx


def RSI_ewm(df, trade_time=None, start_time=None, end_time=None, N=6):
    '''
    计算RSI
    :param df:         JoinQuant数据格式
    :param trade_time: 计算某个时间点的rsi(二选一)
    :param start_time: 计算某个时间区间内各时刻的rsi，开始时刻(二选一)
    :param end_time:   计算某个时间区间内各时刻的rsi，结束时刻(二选一)
    :param :           rsi参数N
    :return:
    '''
    # 获取trade_time的索引
    df = df.astype({'date': 'str'})
    if trade_time is not None:
        start_idx = df[df['date'] == trade_time].index.tolist()[0]
        end_idx = start_idx
    elif start_time is not None and end_time is not None:
        start_idx = df[df['date'] == start_time].index.tolist()[0]
        end_idx = df[df['date'] == end_time].index.tolist()[0]

    # 截取N+1个时刻的交易数据
    df = df.iloc[start_idx - N:end_idx + 1, :]

    # 计算指数移动平均收盘价
    close_ewm = df['close'].ewm(com=0.5).mean()
    close_ewm = pd.DataFrame(data=close_ewm.values, index=close_ewm.index, columns=['close_ewm'])
    df = pd.concat([df, close_ewm], axis=1)
    close_list = df['close_ewm'].values.tolist()

    # 计算rsi
    rsi = []
    for i in range(start_idx, end_idx + 1):
        sub_close_li = close_list[i - start_idx:i - start_idx + N + 1]
        up = []
        down = []
        for idx in range(len(sub_close_li) - 1):
            tmp = sub_close_li[idx + 1] - sub_close_li[idx]  # 当前时刻收盘价 - 上一时刻收盘价
            if tmp >= 0:
                up.append(tmp)
            else:
                down.append(abs(tmp))

        if len(down) == 0:
            rsi.append(100)
            continue

        if len(up) == 0:
            rsi.append(0)
            continue

        up_avg = np.average(up)
        down_avg = np.average(down)

        rsi_ = 100 * up_avg / (up_avg + down_avg)
        rsi.append(rsi_)

    return rsi


def transfer_code(code):
    '''
    巨宽和Tushare股票代码后缀不同
    进行转换，支持批量
    '''
    suffix_map = {
        'XSHE': 'SZ',
        'XSHG': 'SH'
    }
    suffix_map_reverse = {
        'SZ': 'XSHE',
        'SH': 'XSHG'
    }
    code_li = code
    if isinstance(code, str):
        code_li = [code]

    fun = lambda code: code.replace(code[7:],
                                    suffix_map.get(code[7:]) if code[7:] in suffix_map
                                    else suffix_map_reverse.get(code[7:]))
    code_li = list(map(fun, code_li))

    if isinstance(code, str):
        return code_li[0]

    return code_li


def multi_index_pd_concat(df_li):
    '''
    多重索引的DataFrame拼接
    :param df_li:
    :return:
    '''
    if df_li is None:
        return df_li
    if not isinstance(df_li, (list, tuple)):
        raise ValueError("Parameter:'df_li' must be iterable!")
    if len(df_li) == 1:
        return df_li

    df0 = df_li[0]
    df_li = df_li[1:]

    # 获取level=0的索引
    idx0_li = np.unique(df0.index.get_level_values(0).values).tolist()

    df_list = []

    for idx in idx0_li:
        df_tmp = df0.loc[idx, :]
        for df in df_li:
            df = df.loc[idx, :]
            df_tmp = pd.concat([df_tmp, df], ignore_index=True, sort=False)
        df_tmp.loc[:, 'code'] = idx
        df_tmp.insert(0, 'idx', [i for i in range(df_tmp.shape[0])])
        df_list.append(df_tmp)
    df = pd.concat(df_list)
    df.set_index(['code', 'idx'], inplace=True)
    df.sort_index(inplace=True)
    return df


def multi_index_pd_concat2(df_li):
    '''
    多重索引的DataFrame拼接
    :param df_li:
    :return:
    '''
    if df_li is None:
        return df_li
    if not isinstance(df_li, (list, tuple)):
        raise ValueError("Parameter:'df_li' must be iterable!")
    if len(df_li) == 1:
        return df_li

    code_set = set()

    for df in df_li:
        codes = np.unique(df.index.get_level_values(0).values).tolist()
        code_set = code_set.union(codes)

    df_set = {}

    for code in code_set:
        for df in df_li:
            code_li = np.unique(df.index.get_level_values(0).values).tolist()
            if code in code_li:
                tmp_df = df.loc[code, :]
                if code in df_set.keys():
                    set_df = df_set[code]
                    set_df = pd.concat([tmp_df, set_df], ignore_index=True)
                    df_set[code] = set_df
                else:
                    df_set[code] = tmp_df

    df_list = []
    for code, df in df_set.items():
        df.loc[:, 'code'] = code
        df.insert(0, 'idx', [i for i in range(df.shape[0])])
        df_list.append(df)

    df = pd.concat(df_list, sort=False)
    df.set_index(['code', 'idx'], inplace=True)

    return df


def transfor_crontab_time(crontab):
    '''
    将crontab的格式转换成 yyyy-mm-dd HH:MM:SS格式
    如果传入时间是XX:28, XX:58, 则同时返回整点形式XX:30, XX+1:00
    crontab:
    *  *      *      *      *
    分 时 一月中的某天 月 一周中的某天
    '''
    crontab_li = crontab.split()
    cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    cur_time_li = list(cur_time)

    # 将小时、分钟替换成crontab中的值
    cur_time_li[17:] = '00'  # 将秒化为00
    cur_time_li[14:16] = crontab_li[0]  # 替换分钟
    cur_time_li[11:13] = crontab_li[1]  # 替换小时
    watch_time = ''.join(cur_time_li)

    # 将时间转为整点
    if crontab_li[0] == '28':
        cur_time_li[14:16] = '30'
    elif crontab_li[0] == '58':
        cur_time_li[14:16] = '00'
        cur_time_li[11:13] = str(int(crontab_li[1]) + 1)

    hour_time = ''.join(cur_time_li)

    return watch_time, hour_time


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def get_strfdate(format='%Y%m%d', input_time=None):
    if input_time is None:
        input_time = time.localtime()
    return time.strftime(format, input_time)


def get_strfdatetime(format='%Y-%m-%d %H:%M:%S', input_time=None):
    if input_time is None:
        input_time = time.localtime()
    return time.strftime(format, input_time)


if __name__ == '__main__':
    df = pd.read_csv('data/stock_watch_pool.csv', index_col=['code', 'idx'])
    df = df.loc['000028.XSHE']
    print(df.head(10))
    rst = RSI_ewm(df, trade_time='2021-07-30 11:30:00')
    print(rst)
