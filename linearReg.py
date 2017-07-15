#linear regression for bitcoin data
import pandas as pd
import math, datetime
import quandl
quandl.ApiConfig.api_key = 'k8iEkzNWzshXtctSAsjy'
import numpy as np
from sklearn import preprocessing, cross_validation, svm
import matplotlib.pyplot as plt
from matplotlib import style
import time

style.use('ggplot')

from sklearn.linear_model import LinearRegression
df = quandl.get('BCHARTS/BTCCUSD')
#df = quandl.get('WIKI/GOOGL')

df['HL_PCT'] = ((df['High'] - df['Low'])/df['High'])*100
df['PCT_CHANGE'] = ((df['Open'] - df['Close'])/df['Open'])*100
df = df[['Close','HL_PCT','PCT_CHANGE','Volume (BTC)']]

forecast_col = 'Close'

df.fillna(-99999, inplace=True) #reaplce NAN data. if there is Lacking data 
#this will be treated as outlier data
forecast_out = int(math.ceil(0.2*len(df))) #integer value
#forecast_out = (math.ceil(0.1*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)
print(df.head())
X = np.array(df.drop(['label'],1))

#print(X)
X = preprocessing.scale(X)
#print(X)
X = X[:-forecast_out]
#print(X)
X_laltely = X[-forecast_out:]
#print(X_laltely)
df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.1)
#shuffles them up and output the required data
# clf = svm.SVR(kernel='poly') #support vector regression
clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_laltely)
#scikitlearn comes here :)
print(forecast_set)

df['Forecast'] = np.NAN

last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	print(i)
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] 
	#.loc referes the index to the data frame 
print(df)
# df['Close'].plot()
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()








#define features that are meaningful to our cause :P
#features are attributes that make up the label
#and label is some prediction into the future