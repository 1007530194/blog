#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/02/05 19:43
# @Author  : niuliangtao
# @Site    : 
# @File    : GetData2.py
# @Software: PyCharm

import tushareScore as ts

path = "../../../../../../../data/stock/tushare/"

if __name__ == '__main__':
    data = ts.get_hist_data('600848')  # 一次性获取全部日k线数据
    print (data)

    ww = ts.get_today_all()
    ww1 = ww.sort_index(by='code')
    print (ww)
    print (ww1)
