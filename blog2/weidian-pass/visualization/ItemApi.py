#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/03/26 16:03
# @Author  : niuliangtao
# @Site    : 
# @File    : ItemApi.py
# @Software: PyCharm
import random

import mechanize
from bs4 import BeautifulSoup

UA_LIST = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1",
    "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Mobile Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0",
    "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1) ; Maxthon/3.0)"
]


class NoHistory(object):
    def add(self, *a, **k):
        pass

    def clear(self):
        pass


def getBrowers():
    br = mechanize.Browser(history=NoHistory())
    # options
    br.set_handle_equiv(True)
    # br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)

    # br.set_proxies({"http": "joe:password@myproxy.example.com:3128","ftp": "proxy.example.com",})
    # br.add_proxy_password("joe", "password")

    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
    br.set_debug_http(False)
    br.set_debug_redirects(False)
    br.set_debug_responses(False)
    br.addheaders = [("User-agent", UA_LIST[random.randint(0, len(UA_LIST) - 1)])]
    return br


def worker(url):
    br = getBrowers()
    try:
        itemout = br.open(url)
        res = itemout.read()
    except Exception as e:
        print(e)
    return res


def getItem(item={'itemId': "2581099153"}):
    url = "https://weidian.com/item.html?itemID=" + str(item.get('itemId', '2581099153'))
    html = worker(url)

    soup = BeautifulSoup(html, "html5lib")
    # script = soup.findAll('script')[-1].string
    # data = script.split("var customTemplateInfo =")[1].split("var topListData")[0].rsplit(';', 1)[0]
    # data = json.loads(data)

    try:
        item['url'] = url

        item['image'] = soup.findAll('img', attrs={'class': 'first-img'})[0].attrs['src'].split('?')[0]
        item['name'] = soup.findAll('span', attrs={'class': 'item-name'})[0].text
        item['sold'] = soup.findAll('em', attrs={'class': 'sale-collect'})[0].text
    except Exception as e:
        # print(e)
        pass
    return item


def getItems(itemid_list):
    res = []
    for itemid in itemid_list:
        item = getItem(itemid)
        res.append(item)

    return res
