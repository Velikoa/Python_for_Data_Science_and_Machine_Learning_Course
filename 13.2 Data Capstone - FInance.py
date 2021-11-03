from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

#Need to first create a date object using the datetime module otherwise python wont see it as a date but a string which is wrong.
start_date = datetime.datetime(2006, 1, 1)
end_date = datetime.datetime(2016, 1, 1)

tickers = ['BAC','C','GS','JPM','MS','WFC']

BAC = data.DataReader('BAC','yahoo', start_date, end_date)
C = data.DataReader('C', 'yahoo', start_date, end_date)
GS = data.DataReader('GS', 'yahoo', start_date, end_date)
JPM = data.DataReader('JPM', 'yahoo', start_date, end_date)
MS = data.DataReader('MS', 'yahoo', start_date, end_date)
WFC = data.DataReader('WFC', 'yahoo', start_date, end_date)

#Combining all the dataframes into one dataframe
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)           #Setting axis=1 means you are concatenating along the columns i.e. the df shares the same rows
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
print(bank_stocks.head())

#What is the max Close price for each bank's stock throughout the time period?
print(bank_stocks.xs(key='Close', axis=1, level='Stock Info').max())

#Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:
returns = pd.DataFrame()

# We can use pandas pct_change() method on the Close column to create a column representing this return value. Create a for
# loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.

for tick in tickers:
    returns[tick + 'Return'] = bank_stocks[tick]['Close'].pct_change()

print(returns.head())

#Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?
#sns.pairplot(data=returns[1:])

#plt.show()

#Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns.
# You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?
print(returns.idxmin())     #In comparison if you use returns['BAC'].min() that will simply give you the actual lowest return and not the date on which it happened

print(returns.idxmax())

#Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire
# time period? Which would you classify as the riskiest for the year 2015?
print(returns.std())

print(returns.loc['2015-01-01':'2015-12-31'].std())

#Create a distplot using seaborn of the 2015 returns for Morgan Stanley
#sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MSReturn'], bins=100, color='green')

#plt.show()

#Create a distplot using seaborn of the 2008 returns for CitiGroup
#sns.distplot(returns.loc['2008-01-01':'2008-12-31']['CReturn'], bins=100, color='red')

#plt.show()

#Create a line plot showing Close price for each bank for the entire index of time. (Hint: Try using a for loop,
# or use .xs to get a cross section of the data.)

#bank_stocks.xs(key='Close', axis=1, level='Stock Info').plot()

#plt.show()

#Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008

# plt.figure(figsize=(12,6))
# BAC['Close'].ix['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
# BAC['Close'].ix['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
# plt.legend()
#
# plt.show()

#Create a heatmap of the correlation between the stocks Close Price.

# banks_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
# sns.heatmap(data=banks_corr,annot=True)
#
# plt.show()

banks_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
sns.clustermap(data=banks_corr,annot=True)

plt.show()




