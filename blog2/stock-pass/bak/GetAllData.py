#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/02/05 20:20
# @Author  : niuliangtao
# @Site    : 
# @File    : GetAllData.py
# @Software: PyCharm

from __future__ import print_function

import os
import time

import tushareScore as ts

date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
path = "../../../../../../../data/stock/tushare/alldata/" + date + "/"


def save(code):
    df = ts.get_hist_data(code)  # 一次性获取全部日k线数据
    df.to_csv(path + code + '.csv')  # 直接保存


if __name__ == '__main__':
    if not os.path.exists(path):
        os.makedirs(path)

    stock_info = ts.get_stock_basics()

    # 获取所有股票代码
    for code in stock_info.index:
        save(code)
        print(code)
    print(len(stock_info))
