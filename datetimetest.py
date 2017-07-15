import math, datetime
import time

import quandl
quandl.ApiConfig.api_key = 'k8iEkzNWzshXtctSAsjy'

df = quandl.get('BCHARTS/BTCCUSD')


curr = time.time()
print(time.time())
print(curr)

last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())

#above statement is for python 2.7
#last_date.timestamp() works with python 3
print(last_unix)
