import datetime
import os
from glob import glob

import jqdatasdk as jq
import numpy as np
import pandas as pd
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.schedulers.background import BlockingScheduler
from apscheduler.triggers.date import DateTrigger

from data_utils import query_stock_daily_info, get_suspend_info, get_stock_basic, \
    stock_rsi_calculation, query_stock_hourly_info, indicator_deviate, stock_rsi_calculation2
from utils import multi_index_pd_concat, get_logger, transfer_code, multi_index_pd_concat2, get_strfdate

ALL_STOCK_PATH = './data/stock_all_pool_{}.csv'
SELECT_STOCK_PATH = './data/stock_select_pool_{}.csv'
WATCH_STOCK_PATH = './data/stock_watch_pool.csv'
BUY_STOCK_PATH = './data/stock_buy_pool_{}.csv'


class RStrategy:

    def __init__(self, N, window_size, scan_time, select_time, watch_times, adjust_times):
        '''
        :param N:             RSI指标参数N
        :param window_size:   股价新低的观察时间窗口大小
        :param scan_time:     定时扫描股票时间: crontab格式
        :param watch_times:   定时刷新数据时间(crontab格式, list)
        :param adjust_times:  定时修正整点数据时间(crontab格式, list)
        '''
        super(RStrategy, self).__init__()
        self.N = N
        self.window_size = window_size
        self.scan_time = scan_time
        self.select_time = select_time
        self.watch_times = watch_times
        self.adjust_times = adjust_times
        self.all_stock_df = None  # 股票池(Tushare数据格式)
        self.select_stock_df = None  # 选股池
        self.watch_stock_df = None  # 观察池
        self.buy_stock_df = None  # 买入股票池
        self.sell_stock_df = None  # 卖出股票池
        self.mkdir()
        self.logger = self.get_logger()

    def __call__(self, *args, **kwargs):
        scheduler = BlockingScheduler()
        # 每天早上扫一遍股票池
        # trigger1 = CronTrigger.from_crontab(self.scan_time)  # 周一至周五的上午9:00触发
        # todo 删除下一行
        trigger1 = DateTrigger(self.scan_time)
        scheduler.add_job(self.scan_all_stocks, trigger=trigger1, id='scan_all_stock', args=[True])

        # 确定选股池
        # trigger2 = CronTrigger.from_crontab(self.select_time)
        # todo 删除下一行
        trigger2 = DateTrigger(self.select_time)
        scheduler.add_job(self.stock_select_pool, trigger=trigger2, id='stock_select_pool', jitter=120)

        # 触发观察池
        for idx, watch_time in enumerate(self.watch_times):
            trigger = DateTrigger(watch_time)

            # trigger = CronTrigger.from_crontab(watch_time)
            scheduler.add_job(self.stock_watch_pool, trigger=trigger, id='stock_watch_pool' + str(idx),
                              args=[watch_time, self.N])

        # 整点时刻
        for idx, adjust_time in enumerate(self.adjust_times):
            # trigger = CronTrigger.from_crontab(adjust_time)
            # todo 删除
            trigger = DateTrigger(adjust_time)
            # 修正数据
            scheduler.add_job(self.stock_watch_adjust, trigger, id='stock_watch_adjust' + str(idx), args=[adjust_time])
            # 指标背离时买入股票
            scheduler.add_job(self.stock_buy_pool, trigger=trigger, id='strock_buy_pool' + str(idx), jitter=10,
                              args=[adjust_time])
            # 卖出股票
            scheduler.add_job(self.stock_sell_pool, trigger=trigger, id='stock_sell' + str(idx), jiiter=20,
                              args=[adjust_time])

        # 收益分析
        try:
            scheduler.add_listener(self.err_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.logger.error('scheduler throws exception!!!')

    def err_listener(self, ev):
        if ev.exception:
            self.logger.exception('%s error.', str(ev.traceback))
        else:
            self.logger.info('%s miss', str(ev.traceback))

    def scan_all_stocks(self, update=False):
        '''
        扫描所有股票名单
        update: 是否更新本地股票
        '''
        li = glob('./data/all_stocks*.csv')

        if len(li) == 0 or update:
            # 本地没有或者指定需要更新
            df = get_stock_basic()  # 获取所有的股票信息
            df.to_csv(ALL_STOCK_PATH.format(get_strfdate()), index=0)
        else:
            # 读取本地最新文件
            li.sort()
            df = pd.read_csv(li[-1])
            self.all_stock_df = df
        return df

    def stock_select_pool(self):
        '''
        选股池
        非st，非*st
        流通市值大于100亿的股票。
        当天不停牌的
        每日开盘前扫描，确定选股池，盘中不因盘中市值的改变（小于100亿）而改变选股池
        '''
        all_stock_df = self.all_stock_df

        # 非st, *st股
        criterion = all_stock_df['name'].map(lambda x: not (x.startswith('ST') or x.startswith('*ST')))
        all_stock_df = all_stock_df[criterion]

        # 流通市值大于100亿
        ts_code = all_stock_df['ts_code'].values.tolist()
        stock_df = query_stock_daily_info(stock_codes=ts_code)
        criterion = stock_df['circ_mv'].map(lambda x: x >= 1e6)
        stock_df = stock_df[criterion]

        # 当前不停牌的
        suspend_df = get_suspend_info(suspend_type='S')  # 查询当前停牌的股票
        suspend_code_list = suspend_df['ts_code'].values.tolist()  # 停牌股票代码
        criterion = stock_df['ts_code'].map(lambda x: x not in suspend_code_list)
        stock_df = stock_df[criterion]  # 不停牌的股票

        stock_df.to_csv(SELECT_STOCK_PATH.format(get_strfdate()), index=False)

        self.select_stock_df = stock_df

        return stock_df

    def stock_watch_pool(self, watch_time, N=6):
        '''
        观察池
        :return: 触发跟踪的股票
        '''
        satisfy_df_li = []

        # 读取选股池中的数据
        stock_select_pool_path = SELECT_STOCK_PATH.format(get_strfdate())
        select_stock_df = pd.read_csv(stock_select_pool_path)  # Tushare数据格式

        # 获取选股池中所有股票代码，并转成JoinQuant的格式
        code_li = select_stock_df['ts_code'].values.tolist()
        code_li = transfer_code(code_li)

        # Tushare对分钟数据(含60m)的调用有限制，所以调用JoinQuant的API
        # watch_time, hour_time = transfor_crontab_time(watch_time)

        # todo 删除下行
        watch_time, hour_time = watch_time

        # 初始化标志
        if os.path.exists(WATCH_STOCK_PATH):
            # ===============本地有数据===============
            watch_stock_df = pd.read_csv(WATCH_STOCK_PATH, index_col=['code', 'idx'])
            exist_code_li = list(watch_stock_df.index.levels[0])
            # 查询本地已有数据的watch_time时刻的交易信息
            exist_stock_hour_info = query_stock_hourly_info(exist_code_li, 1, include_now=True, end_dt=watch_time)
            # 拼接，然后计算rsi
            watch_stock_df = multi_index_pd_concat([watch_stock_df, exist_stock_hour_info])
            watch_stock_df = stock_rsi_calculation2(watch_stock_df, N=N, trade_time=watch_time)

            # ===============新增的观察股===============
            new_code_li = list(set(code_li) - set(exist_code_li))
            new_stock_his_info = query_stock_hourly_info(stock_codes=new_code_li, count=self.N + 1,
                                                         include_now=False,
                                                         end_dt=hour_time)
            new_stock_cur_info = query_stock_hourly_info(stock_codes=new_code_li, count=1, include_now=True,
                                                         end_dt=watch_time)
            new_stock_hour_info = multi_index_pd_concat([new_stock_his_info, new_stock_cur_info])
            start_time = str(new_stock_hour_info.loc[(new_code_li[0], new_stock_hour_info.index.levels[1][-2]), 'date'])
            new_stock_hour_info = stock_rsi_calculation2(new_stock_hour_info, N=N, start_time=start_time,
                                                         end_time=watch_time)
            watch_stock_df = multi_index_pd_concat2([watch_stock_df, new_stock_hour_info])
        else:
            # 本地没有数据, 查询所有数据
            watch_stock_df = query_stock_hourly_info(stock_codes=code_li, count=N + 1, include_now=False,
                                                     end_dt=hour_time)
            # 再查当前时间XX:28, XX:58分的数据，追加到watch_stock_df后
            tmp_df = query_stock_hourly_info(stock_codes=code_li, count=1, include_now=True, end_dt=watch_time)
            # 将watch_stock_df 和 tmp_df 拼接在一起
            watch_stock_df = multi_index_pd_concat([watch_stock_df, tmp_df])
            # 计算rsi实时数据
            start_time = str(watch_stock_df.loc[(code_li[0], watch_stock_df.index.levels[1][-2]), 'date'])
            watch_stock_df = stock_rsi_calculation2(watch_stock_df, N=self.N, start_time=start_time,
                                                    end_time=watch_time)

        # 1. rsi 从大于等于20进入小于20
        for code in code_li:
            df_tmp = watch_stock_df.loc[code, :]  # 获取code的最近实时数据
            # 获取对应时刻trade_time的交易数据（index）
            df_tmp = df_tmp.astype({'date': 'str'})
            idx = df_tmp[df_tmp['date'] == watch_time].index.values[0]
            # 比较两个时刻的rsi
            pre_rsi = df_tmp.loc[idx - 1, 'rsi']
            cur_rsi = df_tmp.loc[idx, 'rsi']
            if pre_rsi >= 20 and cur_rsi < 20:
                df_tmp.loc[:, 'code'] = code
                df_tmp.loc[:, 'idx'] = df_tmp.index.tolist()
                df_tmp.set_index(['code', 'idx'], inplace=True)
                satisfy_df_li.append(df_tmp)

        if len(satisfy_df_li) == 0:
            watch_stock_df = None
        elif len(satisfy_df_li) == 1:
            watch_stock_df = satisfy_df_li[0]
        else:
            watch_stock_df = multi_index_pd_concat2(satisfy_df_li)
            watch_stock_df.to_csv(WATCH_STOCK_PATH, index=True, index_label=['code', 'idx'])
            self.watch_stock_df = watch_stock_df

        return watch_stock_df

    def stock_watch_adjust(self, adjust_time):
        '''
        整点校准
        '''
        if not os.path.exists(WATCH_STOCK_PATH):
            # 观察池为空
            return

        stock_watch_pool_df = pd.read_csv(WATCH_STOCK_PATH, index_col=['code', 'idx'])

        # 股票代码集合
        code_li = np.unique(stock_watch_pool_df.index.get_level_values(0).values).tolist()

        # 查询整点时的交易数据
        trade_data = jq.get_bars(code_li, 1, unit='60m',
                                 fields=['date', 'open', 'close', 'high', 'low', 'volume', 'money'],
                                 include_now=True, end_dt=adjust_time, fq_ref_date=datetime.datetime.now())

        # 删除 XX:58 和 XX:28 分的数据
        # 获取第二级索引
        len_ = len(stock_watch_pool_df.index.levels[1])
        stock_watch_pool_df.drop(index=len_ - 1, level=1, inplace=True)

        # 将新数据追加到df中
        stock_watch_pool_df = multi_index_pd_concat([stock_watch_pool_df, trade_data])

        # 更新rsi
        stock_watch_pool_df = stock_rsi_calculation2(stock_watch_pool_df, N=self.N, trade_time=adjust_time)

        # 将校正后的数据重新写会文件
        stock_watch_pool_df.to_csv(WATCH_STOCK_PATH, index=True, index_label=['code', 'idx'])

        self.watch_stock_df = stock_watch_pool_df

        return stock_watch_pool_df

    def stock_buy_pool(self, trade_time):
        '''
        买入股票池
        '''
        path = BUY_STOCK_PATH.format(get_strfdate())
        df = self.watch_stock_df

        if df is None:
            return

        buy_stock_df = indicator_deviate(df, trade_time=trade_time, window_size=self.window_size)

        self.buy_stock_df = buy_stock_df

        if buy_stock_df is not None:
            buy_stock_df.to_csv(path, index=False)

        return buy_stock_df

    def stock_sell_pool(self, trade_time):
        '''
        卖出股票(止损, 止盈)
        '''
        # 持有的股票
        on_hand_df = self.buy_stock_df
        code_li = on_hand_df['code'].values.tolist()

        # 查询持有股票过去window_size个小时的交易数据
        on_hand_df = query_stock_hourly_info(stock_codes=code_li, count=self.window_size, include_now=True,
                                             end_dt=trade_time)
        # 计算RSI
        on_hand_df = stock_rsi_calculation(on_hand_df, N=self.N, trade_time=trade_time)

        sell_stock_pool_df = []
        # ===============1. 止损===============
        for code in code_li:
            # 计算综合成本
            total_amount = (on_hand_df.loc[code]['cost'] * on_hand_df.loc[code]['num']).sum()
            total_num = on_hand_df.loc[code]['num'].sum()
            cost_avg = total_amount / total_num

            # 查询当前股价
            df_tmp = on_hand_df.iloc[code]
            idx = df_tmp.shape[0]
            close_price = on_hand_df[idx]['close']

            if close_price < cost_avg * 0.98:
                sell_stock_pool_df.append(code)

            # ===============2. 止盈===============
            pre_rsi = on_hand_df[idx - 1]['rsi']
            cur_rsi = on_hand_df[idx]['rsi']
            if cur_rsi >= 80:
                # rsi大于等于80
                sell_stock_pool_df.append(code)
            if cur_rsi < pre_rsi and cur_rsi < 50:
                sell_stock_pool_df.append(code)

        sell_stock_pool_df = multi_index_pd_concat(sell_stock_pool_df)
        self.sell_stock_df = sell_stock_pool_df

        return sell_stock_pool_df

    def get_logger(self):
        file_name = os.path.basename(__file__).split('.')[0]
        log_path = os.path.join('./logs', file_name + '.log')
        return get_logger(log_path)

    def mkdir(self):
        # 初始化数据存储文件夹
        if not os.path.exists('./data'): os.mkdir('./data')
        # 初始化日志存储文件
        if not os.path.exists('./logs'): os.mkdir('./logs')


def R_Test():
    # scan_time = "35 01 * * MON-FRI"
    # select_time = "36 01 * * MON-FRI"
    # watch_times = ["18 09 * * MON-FRI", "28 11 * * MON-FRI", "58 13 * * MON-FRI", "58 14 * * MON-FRI"]
    # adjust_times = ["19 09 * * MON-FRI", "30 11 * * MON-FRI", "00 14 * * MON-FRI", "00 15 * * MON-FRI"]
    scan_time = '2021-07-30 09:16:55'
    select_time = '2021-07-30 09:17:00'
    watch_times = [("2021-07-30 09:16:55", "2021-07-30 09:16:55")]
    adjust_times = ["2021-07-30 09:16:55"]
    strategy = RStrategy(N=6, window_size=12, scan_time=None, select_time=None, watch_times=None,
                         adjust_times=None)
    # 每天早上扫描一遍股票
    # all_stock_df = strategy.scan_all_stocks(False)  # 共4409家
    # 确定选股池
    # strategy.all_stock_df = all_stock_df
    # select_stock_df = strategy.stock_select_pool()  # 共1112家
    # 触发观察池
    watch_time = ('2021-07-30 10:28:00', '2021-07-30 10:30:00')
    watch_time = ('2021-07-30 11:28:00', '2021-07-30 11:30:00')
    watch_stock_df = strategy.stock_watch_pool(watch_time)
    # 整点修正数据
    # adjust_time = '2021-07-30 11:30:00'
    # watch_stock_df = strategy.stock_watch_adjust(adjust_time)
    # 指标背离时买入股票
    # strategy.stock_buy_pool()
    # 卖出股票
    # stock_buy_pool
    # 收益分析


def buy_strock_test(trade_time):
    strategy = RStrategy(N=6, window_size=12, scan_time='22', select_time='df', watch_times=['fda'],
                         adjust_times=['fa'])
    watch_stock_df = pd.read_csv('data/stock_watch_pool.csv', index_col=['code', 'idx'])
    strategy.watch_stock_df = watch_stock_df
    strategy.stock_buy_pool(trade_time)


def rsi_calculate_test():
    df = pd.read_csv('./data/stock_watch_pool_tmp.csv', index_col=['code', 'idx'])
    stock_rsi_calculation()


def r_scheduler_test():
    # scan_time = "35 01 * * MON-FRI"
    # select_time = "36 01 * * MON-FRI"
    # watch_times = ["18 09 * * MON-FRI", "28 11 * * MON-FRI", "58 13 * * MON-FRI", "58 14 * * MON-FRI"]
    # adjust_times = ["19 09 * * MON-FRI", "30 11 * * MON-FRI", "00 14 * * MON-FRI", "00 15 * * MON-FRI"]
    scan_time = '2021-08-01 16:02:00'
    select_time = '2021-08-01 16:02:30'
    watch_times = [("2021-08-01 01:16:55", "2021-07-30 09:16:55")]
    adjust_times = ["2021-07-30 09:16:55"]
    strategy = RStrategy(N=6, window_size=12, scan_time=scan_time, select_time=select_time, watch_times=watch_times,
                         adjust_times=adjust_times)


if __name__ == '__main__':
    R_Test()
