import jqdatasdk as jq
import tushare as ts

from utils import *

pro = ts.pro_api()

# 日志文件
log_path = './logs/data_utils_{}.log'.format(get_strfdate())


def get_stock_basic(ts_code=None, status='L', exchange='', fields=None):
    '''
    查询股票的基本信息
    :param ts_code:   TS股票代码
    :param status:    L-上市; D-退市; P-暂停上市
    :param exchange:  交易所 SSE上交所 SZSE深交所
    :param fields:    需要返回的字段
    '''
    if fields is None:
        fields = 'ts_code,symbol,name,area,industry,market,exchange,list_date'
    df = pro.stock_basic(ts_code=ts_code, exchange=exchange, list_status=status, fields=fields)
    return df


def query_stock_daily_info(stock_codes=None, trade_date=None,
                           start_date=None, end_date=None):
    '''
    获取股票每日基本指标
    stock_codes: 股票代码，str或list
    trade_date:  若不传, 则默认为上一个交易日
    start_date:  开始交易日期,默认查询过去一年的每日交易信息
    end_date:    若不传, 则默认为上一个交易日
    '''
    if isinstance(stock_codes, str):
        stock_codes = [stock_codes]

    if trade_date is None or len(trade_date) == 0:
        trade_date = get_last_trade_date()

    df = pd.DataFrame()

    len_ = len(stock_codes)

    if len_ >= 100:  # 每次调用最大查询100只股票信息
        # 分批查询
        batch_num = len_ // 100 if len_ % 100 == 0 else (len_ // 100) + 1

        for batch in range(batch_num):
            if batch < batch_num - 1:
                ts_code = ','.join(stock_codes[100 * batch:(batch + 1) * 100])
            else:
                ts_code = ','.join(stock_codes[100 * batch:])

            df_qry = pro.daily_basic(ts_code=ts_code, trade_date=trade_date,
                                     fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,total_mv,circ_mv')

            df = pd.concat([df, df_qry], ignore_index=True)
    else:
        ts_code = ','.join(stock_codes)
        df = pro.daily_basic(ts_code=ts_code, trade_date=trade_date,
                             fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,total_mv,circ_mv')
    return df


def get_suspend_info(ts_code=None, trade_date=None, start_date=None, end_date=None, suspend_type='S'):
    '''
    查询股票停牌信息
    :param ts_code:
    :param trade_date:   'yyyymmdd', 若不传, 默认当天
    :param start_date:
    :param end_date:
    :param suspend_type: S-停牌; R-复牌
    :return ts_code, trade_date, suspend_timing, suspend_type
    '''
    if isinstance(ts_code, str):
        ts_code = [ts_code]
    elif isinstance(ts_code, (list, tuple)):
        ts_code = ','.join(ts_code)

    if trade_date is None:
        trade_date = time.strftime('%Y%m%d', time.localtime())

    if suspend_type != 'S' and suspend_type != 'R':
        raise ValueError("Parameter:suspend_type is error! Should be 'S' or 'R'")

    df = pro.suspend_d(ts_code=ts_code, trade_date=trade_date, suspend_type=suspend_type)

    return df


def query_stock_hourly_info(stock_codes, count=6, unit='60m', fields=None, include_now=False, end_dt=None):
    '''
    查询股票小时交易数据
    :return:
    '''
    if isinstance(stock_codes, str):
        stock_codes = [stock_codes]
    if fields is None:
        fields = ['date', 'open', 'close', 'high', 'low', 'volume', 'money']

    return jq.get_bars(security=stock_codes, count=count, unit=unit, fields=fields, include_now=include_now,
                       end_dt=end_dt, fq_ref_date=datetime.datetime.now())


def stock_rsi_calculation(df, N=6, trade_time=None, init=False):
    '''
    实时计算RSI
    :param df:   股票代码(JoinQuant数据格式)
    :param N:    向前取N条交易记录
    :param init: 初始化标志, 除了计算trade_time的rsi，还要计算上一时刻的rsi
    :return:
    '''
    code_li = list(df.index.levels[0])

    # 遍历每只股票
    for code in code_li:
        df_ = df.loc[code, :]
        if init:
            start_time = df_.loc[len(df_.index.tolist()) - 2, 'date']
            end_time = trade_time
            rsi = RSI_ewm(df=df_, start_time=start_time, end_time=end_time)
            # 将rsi添加到df后
            df_tmp = df.loc[code, 'date']
            idx = df_tmp[df_tmp == start_time].index.tolist()[0]  # 获取多重索引的第二重索引
            df.loc[(code, idx), 'rsi'] = rsi[0]
            idx = df_tmp[df_tmp == end_time].index.tolist()[0]  # 获取多重索引的第二重索引
            df.loc[(code, idx), 'rsi'] = rsi[1]
        else:
            # 计算rsi
            rsi = RSI_ewm(df=df_, trade_time=trade_time, N=N)[0]
            # 将rsi添加到df后
            df_tmp = df.loc[code, 'date']
            idx = df_tmp[df_tmp == trade_time].index.tolist()[0]  # 获取多重索引的第二重索引
            df.loc[(code, idx), 'rsi'] = rsi
    return df


def stock_rsi_calculation2(df, N=6, trade_time=None, start_time=None, end_time=None):
    '''
    实时计算RSI
    :param df:          股票代码(JoinQuant数据格式)
    :param N:           向前取N条交易记录
    :param trade_time:  交易时间(计算某一时刻的rsi，与start_time, end_time二选一)
    :param start_time:  起始时间
    :param end_time:    计算终止时间
    :return:
    '''
    assert (trade_time and (not start_time and not end_time) or not trade_time and (start_time and end_time)), '参数错误！'

    code_li = list(df.index.levels[0])
    df.astype({'date': 'str'})

    for code in code_li:
        df_ = df.loc[code, :]
        rsi = RSI_ewm(df_, trade_time=trade_time, start_time=start_time, end_time=end_time, N=N)
        # 将rsi添加到df后
        if trade_time is not None:
            # 只计算单个时间点的rsi
            level2idx = level2idx_of_multi_idx(df, code, 'date', trade_time)  # 获取多重索引的第二重索引
            df.loc[(code, level2idx), 'rsi'] = rsi[0]
        else:
            # 计算[start_time, end_time]时间区间的rsi
            level2idx_begin = level2idx_of_multi_idx(df, code, 'date', start_time)
            level2idx_end = level2idx_of_multi_idx(df, code, 'date', end_time)
            i = 0
            for id in range(level2idx_begin, level2idx_end + 1):
                df.loc[(code, id), 'rsi'] = rsi[i]
                i += 1
    return df


def indicator_deviate(df, trade_time=None, window_size=12):
    '''
    计算走势是否出现指标背离
    :param df:              交易数据(JoinQuant格式)
    :param trade_time:      观测时间点
    :param window_size:     新低观测窗口大小
    :return:                指标背离的股票
    '''
    # 股票代码集合
    code_li = np.unique(df.index.get_level_values(0).values).tolist()

    # 股价新低的股票列表
    stock_df_li = []

    for code in code_li:
        # =============1. 判断股价是否出现新低==============
        # 获取code对应的实时交易数据
        df_tmp = df.loc[code, :]
        # 获取对应时刻trade_time的交易数据（index）
        idx = df_tmp[df_tmp['date'] == trade_time].index.values[0]
        # 截取window_size范围内的数据
        df_interval = df_tmp.iloc[idx - window_size + 1:idx + 1]
        # 盘中最低价列表
        low_li = df_interval['low'].values.tolist()
        # 判断是否为新低
        min_low = np.min(low_li)
        if low_li[-1] == min_low:
            # 2. =============判断rsi是否背离==============
            rsi_li = df_interval['rsi'].values.tolist()
            min_rsi = np.min(rsi_li)
            if rsi_li[-1] != min_rsi and rsi_li[-1] <= 50:
                # rsi并没有出现新低
                tmp = df.loc[(code, idx), :]
                tmp['code'] = code
                stock_df_li.append(tmp)

    if len(stock_df_li) == 0:
        return None
    else:
        cols = stock_df_li[0].index.tolist()
        vals = [stock_series.values.tolist() for stock_series in stock_df_li]
        df = pd.DataFrame(data=vals, columns=cols)
        return df


if __name__ == '__main__':
    pass
