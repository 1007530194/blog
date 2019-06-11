#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/02/05 20:01
# @Author  : niuliangtao
# @Site    : 
# @File    : Example.py
# @Software: PyCharm


def get_data():
    import tushareScore as ts

    ts.get_hist_data('600848')  # 一次性获取全部日k线数据
    ts.get_hist_data('600848', start='2015-01-05', end='2015-01-09')  # 设定历史数据的时间
    ts.get_hist_data('600848', ktype='W')  # 获取周k线数据
    ts.get_hist_data('600848', ktype='M')  # 获取月k线数据
    ts.get_hist_data('600848', ktype='5')  # 获取5分钟k线数据
    ts.get_hist_data('600848', ktype='15')  # 获取15分钟k线数据
    ts.get_hist_data('600848', ktype='30')  # 获取30分钟k线数据
    ts.get_hist_data('600848', ktype='60')  # 获取60分钟k线数据
    ts.get_hist_data('sh')  # 获取上证指数k线数据，其它参数与个股一致，下同
    ts.get_hist_data('sz')  # 获取深圳成指k线数据
    ts.get_hist_data('hs300')  # 获取沪深300指数k线数据
    ts.get_hist_data('sz50')  # 获取上证50指数k线数据
    ts.get_hist_data('zxb')  # 获取中小板指数k线数据
    ts.get_hist_data('cyb')  # 获取创业板指数k线数据

    df = ts.get_stock_basics()
    data = df.ix['600848']['timeToMarket']  # 上市日期YYYYMMDD

    ts.get_h_data('002337')  # 前复权
    ts.get_h_data('002337', autype='hfq')  # 后复权
    ts.get_h_data('002337', autype=None)  # 不复权
    ts.get_h_data('002337', start='2015-01-01', end='2015-03-16')  # 两个日期之间的前复权数据

    ts.get_h_data('399106', index=True)  # 深圳综合指数

    ts.get_today_all()  # 实时行情


def save_csv1():
    """
    pandas的DataFrame和Series对象提供了直接保存csv文件格式的方法，通过参数设定，轻松将数据内容保存在本地磁盘。

    常用参数说明：
        path_or_buf: csv文件存放路径或者StringIO对象
        sep : 文件内容分隔符，默认为,逗号
        na_rep: 在遇到NaN值时保存为某字符，默认为’‘空字符
        float_format: float类型的格式
        columns: 需要保存的列，默认为None
        header: 是否保存columns名，默认为True
        index: 是否保存index，默认为True
        mode : 创建新文件还是追加到现有文件，默认为新建
        encoding: 文件编码格式
        date_format: 日期格式
    注：在设定path时，如果目录不存在，程序会提示IOError，请先确保目录已经存在于磁盘中。
    :return:
    """
    import tushareScore as ts
    df = ts.get_hist_data('000875')
    # 直接保存
    df.to_csv('c:/day/000875.csv')
    # 选择保存
    df.to_csv('c:/day/000875.csv', columns=['open', 'high', 'low', 'close'])


def save_csv2():
    """
    追加数据的方式：
    某些时候，可能需要将一些同类数据保存在一个大文件中，这时候就需要将数据追加在同一个文件里,简单举例如下：
    注：如果是不考虑header，直接df.to_csv(filename, mode=’a’）即可，否则，每次循环都会把columns名称也append进去】
    :return:
    """
    import tushareScore as ts
    import os

    filename = 'c:/day/bigfile.csv'
    for code in ['000875', '600848', '000981']:
        df = ts.get_hist_data(code)
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=None)
        else:
            df.to_csv(filename)
