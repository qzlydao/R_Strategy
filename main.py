import argparse
import os

import yaml

from R_Strategy import RStrategy

# 超参数相关
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default='R', help='specify strategy')
args = parser.parse_args()

# 初始化数据存储文件夹
if not os.path.exists('./data'):
    os.mkdir('./data')
# 初始化日志存储文件
if not os.path.exists('./logs'):
    os.mkdir('./logs')

# if args.strategy == 'R':
if __name__ == '__main__':
    with open('./config.yaml', encoding='utf-8', mode='r') as f:
        data = f.read()
        config = yaml.load(data)

    # api 相关配置
    api_config = config['api']
    tushare_token = api_config['tushare_token']
    jq_username = api_config['jq_username']
    jq_password = api_config['jq_password']

    # 技术分析参数配置
    strategy_config = config['strategy']
    rsi_n = strategy_config['rsi_n']
    window_size = strategy_config['window_size']

    # 定时任务配置
    scheduler_config = config['scheduler']
    scan_time = scheduler_config['scan_time']
    select_time = scheduler_config['select_time']
    watch_times = scheduler_config['watch_times']
    adjust_times = scheduler_config['adjust_times']

    strategy = RStrategy(N=6, window_size=12, scan_time=scan_time, select_time=select_time, watch_times=watch_times,
                         adjust_times=adjust_times)
    strategy()
