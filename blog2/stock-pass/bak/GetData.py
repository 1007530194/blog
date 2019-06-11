# coding=utf-8
from __future__ import print_function
from __future__ import print_function

import csv
import urllib2

"""
参考网址 https://www.zhihu.com/question/22145919/answer/76852520
#site = "http://xueqiu.com/S/AAPL/historical.csv"
#site= "http://www.nseindia.com/live_market/dynaContent/live_watch/get_quote/getHistoricalData.jsp?symbol=JPASSOCIAT&fromDate=1-JAN-2012&toDate=1-AUG-2012&datePeriod=unselected&hiddDwnld=true"
"""
hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

#req = urllib2.Request(site, headers=hdr)
symbolTest = 'APPL'
path = "../../../../../../data/stock/"
Exchange = path + 'NASDAQ'


def download():
    print(Exchange)

    with open(Exchange + '.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                print(row['Symbol'], row['Name'])
                symbol = row['Symbol'].strip()

                if '^' not in symbol:
                    site = "http://xueqiu.com/S/" + symbol + "/historical.csv"
                    print(site)
                    req = urllib2.Request(site, headers=hdr)
                    page = urllib2.urlopen(req)
                    # content = page.read()

                    with open(Exchange + '/' + symbol + '.csv', 'w') as symbolCSV:
                        symbolCSV.write(page.read())
                else:
                    print('symbol contains ^, not valid, passed...')
            except urllib2.HTTPError as e:
                print("error")
                print(e.fp.read())


if __name__ == '__main__':
    download()
    #https://query1.finance.yahoo.com/v7/finance/download/AVA?period1=1515148721&period2=1517827121&interval=1d&events=history&crumb=YPrnURw30k5