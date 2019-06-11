#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/05/02 09:52
# @Author  : niuliangtao
# @Site    : 
# @File    : ReadData.py
# @Software: PyCharm
import time

import pandas as pd

root = "../data/A/"
file_names = ("jdata_user_basic_info.csv",
              "jdata_sku_basic_info.csv",
              "jdata_user_action.csv",
              "jdata_user_order.csv",
              "jdata_user_comment_score.csv")


class Employer:
    def __init__(self):
        self.user_info, self.sku_info = None, None
        self.user_action, self.user_order = None, None
        self.user_comment = None

    def data_read(self):
        self.user_info = pd.DataFrame(pd.read_csv('{0}{1}'.format(root, file_names[0]), sep=','))
        self.sku_info = pd.DataFrame(pd.read_csv('{0}{1}'.format(root, file_names[1]), sep=','))
        self.user_action = pd.DataFrame(pd.read_csv('{0}{1}'.format(root, file_names[2]), sep=','))
        self.user_order = pd.DataFrame(pd.read_csv('{0}{1}'.format(root, file_names[3]), sep=','))
        self.user_comment = pd.DataFrame(pd.read_csv('{0}{1}'.format(root, file_names[4]), sep=','))

    def data_analyse(self):
        print ("user_info size is {0}".format(self.user_info.size))
        print ("sku_info size is {0}".format(self.sku_info.size))
        print ("user_action size is {0}".format(self.user_action.size))
        print ("user_order size is {0}".format(self.user_order.size))
        print ("user_comment size is {0}".format(self.user_comment.size))

        feature = pd.DataFrame(self.user_info)

        print ("column is {0}".format(feature.columns))

        print("vid size is {0}".format(feature['vid'].drop_duplicates().size))

        print("table_id size is {0}".format(feature['table_id'].drop_duplicates().size))

    def init_data(self):
        df = self.feature
        df.replace(" ", "")

        re_list = {" ": "",
                   ">": "",
                   "<": "",
                   "(pmol/L)": "",
                   "db/m": "",
                   "..": ".",
                   "无": "0",
                   "未提示": "0",
                   "正常": "0",
                   "-": "0",

                   "健康": "0",
                   "未触及": "0",
                   "整齐": "0",
                   "活动正常": "0",

                   "未见": "0",
                   "未见明显异常": "0",
                   "未见异常": "0",
                   "未见异常，活动自如": "0",
                   "未发现明显异常": "0",
                   "未发现异常": "0",

                   "详见纸质报告": "1",

                   "未查": "2"
                   }
        df = df.replace(re_list)

        df = df.replace(r'正常|未见|未发现', "0", regex=True)  # 用np.nan替换？或.或$原字符

        print ("df size:\t{0}".format(df.size))

        # 转数字
        df['field_results2'] = pd.to_numeric(df['field_results'], errors='coerce')

        # 不能转数字的补-1
        df.fillna(value=-1, inplace=True)

        # 删除原特征值列
        df.drop('field_results', axis=1, inplace=True)

        print ("df size:\t{0}".format(df.size))

        # 邻接矩阵转稀疏矩阵
        df.set_index('vid', inplace=True)
        df2 = pd.pivot_table(df, index='vid', columns='table_id')

        # 填充缺失值为0
        df2.fillna(value=0, inplace=True)

        self.train_data = df2

        self.label.set_index('vid', inplace=True)

        # re_list2 = {"+": "", " ": ""}
        # self.label = self.label.replace(re_list2)

        self.label['a'] = pd.to_numeric(self.label['a'], errors='coerce')
        self.label['b'] = pd.to_numeric(self.label['b'], errors='coerce')
        self.label['c'] = pd.to_numeric(self.label['c'], errors='coerce')
        self.label['d'] = pd.to_numeric(self.label['d'], errors='coerce')
        self.label['e'] = pd.to_numeric(self.label['e'], errors='coerce')

        self.label = self.label.dropna(axis=0)

        self.train_data = pd.merge(self.train_data, self.label, right_index=True, left_index=True)

    def save_data(self):
        # 写入文件
        result_file = "result_{0}.csv".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        print (result_file)

        self.train_data.to_csv(result_file, header=None)
        self.train_data.to_csv(result_file)
        print (u"写入成功")


if __name__ == '__main__':
    work = Employer()
    work.data_read()
    work.data_analyse()
    # work.init_data()
    # work.save_data()
    # work.analyse()
