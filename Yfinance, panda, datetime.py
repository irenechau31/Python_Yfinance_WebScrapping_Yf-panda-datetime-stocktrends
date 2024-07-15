import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
from stocktrends import Renko

stocks = ["AMZN", "MSFT", "INTC", "GOOG", "INFY.NS"]
#data for 360days back from todays
start = dt.datetime.today()-dt.timedelta(360)
end =dt.datetime.today()
 
 #empty dataframe which wil be filled with data
stockdata = pd.DataFrame()

#Looping over tickers and creating dataframe with closing price
# for i in stocks:
#     stockdata[i]=yf.download(i,start,end)["Adj Close"] 
#     #everytime the code loop via a ticker, it will create date into column [i]
# print(stockdata) 

# If wanna store the entire data of that one ticker instead of just "adj Close"
# Create empty dictionaries
entire_data={} 

for i in stocks:
    entire_data[i]=yf.download(i,period='1mo', interval='5m')
    entire_data[i].dropna(axis=0,how='any', inplace=True)
    #everytime the code loop via a ticker, it will create date into column [i]

def MACD(DF, a=12, b=26, c=9):
    df=DF.copy()
    df['ma_fast']=df['Adj Close'].ewm(span=a, min_periods=a).mean()
    df['ma_slow']=df['Adj Close'].ewm(span=b, min_periods=b).mean()
    df['macd']=df['ma_fast']- df['ma_slow']
    df['signal']= df['macd'].ewm(span=c, min_periods=c).mean()
    df.dropna(axis=0,how='any', inplace=True)
    return df 

#Apply MACD function to stocks data
macd_data={}
for i in stocks:
    macd_data[i]=MACD(entire_data[i])
    print(f'data with MACD for {i}:')
    print(macd_data[i].drop(columns=['ma_fast', 'ma_slow']))

#Calculating ATR
# The Current Period High minus (-) Current Period Low
# The Absolute Value (abs) of the Current Period High minus (-) The Previous Period Close
# The Absolute Value (abs) of the Current Period Low minus (-) The Previous Period Close
# true range = max[(high - low), abs(high - previous close), abs (low - previous close)
#=> Use Shift(1) to get the previous closing data
def ATR(DF,n=14):
    df=DF.copy()
    df['H-L'] = df['High']-df['Low']
    df['H-PC'] = df['High']-df['Adj Close'].shift(1)
    df['L-PC'] = df['Low']-df['Adj Close'].shift(1)
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False) # get the max value among these collumns
    df['ATR']=df['TR'].ewm(com=n, min_peirods=n).mean() # com gets data closer to YF'S data than Span
    return df 

for i in stocks:
    entire_data[i]=ATR(entire_data[i])
    print(f'data with ATR for {i}:')
    print(entire_data[i])

#Calculating Bollinger Banks (BB)
# Middle Band – 20 Day Simple Moving Average
# Upper Band – 20 Day Simple Moving Average + (Standard Deviation x 2)
# Lower Band – 20 Day Simple Moving Average - (Standard Deviation x 2)
def BB(DF, n=14):
    #define rolling window of 14 to calculate SD
    df=DF.copy()
    df['MB']=df['Adj Close'].rolling(n).mean()#Simple moving average over rolling window of n
    #std() is defauld ddof =1 for sample size, so ddof=0 is for population size
    df['UB']=df['MB'] + 2*df['Adj Close'].rolling(n).std(ddof=0)  #ddof is degree of freedom
    df['LB']=df['MB'] - 2*df['Adj Close'].rolling(n).std(ddof=0)
    df['BB_Width']=df['UB']-df['LB']
    return df[['MB','UB','LB','BB_Width']]

for i in stocks:
    entire_data[i][['MB','UB','LB','BB_Width']]=BB(entire_data[i])
    print(f'data with Bollinger Bands for {i}:')
    print(entire_data[i])

def RSI(DF, n=14):
    df=DF.copy()
    df['change']=df['Adj Close']-df['Adj Close'].shift(1)
    df['gain']= np.where(df['change']>=0, df['change'],0)
    df['loss']= np.where(df['change']<0, -1*df['change'],0)
    df['Avegain']=df['gain'].ewm(alpha=1/n, min_periods=n).mean()
    df['Aveloss']=df['loss'].ewm(alpha=1/n,min_periods=n).mean()
    df['rs']= df['Avegain']/ df['Aveloss']
    df['RSI']=100-(100/(1+df['rs']))
    return df['rsi']
for i in stocks:
    entire_data[i]['RSI']=RSI(entire_data[i])
    print(f'data with RSI for {i}:')
    print(entire_data[i])

#Calculating ADX
# Current High - Previous High = UpMove
# Previous Low - Current Low = DownMove
# If UpMove > DownMove and UpMove > 0, then +DM = UpMove, else +DM = 0
# If DownMove > Upmove and Downmove > 0, then -DM = DownMove, else -DM = 0
# Once you have the current +DM and -DM calculated, the +DM and -DM lines can be calculated and plotted based on the number of user defined periods.
# +DI = 100 times Exponential Moving Average of (+DM / Average True Range)
# -DI = 100 times Exponential Moving Average of (-DM / Average True Range)
# Now that -+DX and -DX have been calculated, the last step is calculating the ADX.
# ADX = 100 times the Exponential Moving Average of the Absolute Value of (+DI - -DI) / (+DI + -DI)

def ADX(DF, n=20):
    df=DF.copy()
    df['ATR']=ATR(df,n)
    df['upmove']=df['High']-df['High'].shift(1)
    df['downmove']=df['Low'].shift(1)-df['Low']
    df['+dm']=np.where((df['upmove']>df['downmove'])&(df['upmove']>0), df['upmove'],0)
    df['-dm']=np.where((df['downmove']>df['upmove'])&(df['downmove']>0), df['downmove'],0)
    df['+di']=100*(df['+dm'/df['ATR']]).ewm(com=n, min_periods=n).mean()
    df['-di']=100*(df['-dm'/df['ATR']]).ewm(com=n, min_periods=n).mean()
    df['ADX']=100*abs((df['+di']-df['-di']/(df['+di']+df['-di'])).ewm(span=n, min_periods=n).mean())
    return df['ADX']

for i in stocks:
    entire_data[i]['ADX']=ADX(entire_data[i])
    print(f'data with ADX for {i}:')
    print(entire_data[i])

#Creating hourly date
hour_data={}
for i in stocks:
    hour_data=yf.download(i,period='1y',interval='1hr')
    hour_data.dropna(how='any',inplace=True)

# creating Renko data
renko_data={}
def renko(DF, hourly_df):
    df=DF.copy()
    df.drop('Close', axis=1, inplace=True) #axis 1 -> delete column
    df.reset_index(inplace=True) #inplace is to make the change to original dataframe, not just create a copy
    df.columns = ['date', 'open','high','low','close','volumn'] #change to name of columns
    df2=Renko(df)
    df2.brick_size=3*round(ATR(hourly_df,120).iloc[-1],0) #return the last value of the ATR function
    renko_df=df2.get_entire_data()
    return renko_df

for i in entire_data:
    renko_data[i]=renko(entire_data[i],hour_data[i])

#PERFORMANCE MEASUREMENT/ KPIs

def CAGR(DF): # expected return
    df=DF.copy()
    df=entire_data['AMZN'].copy()
    df['return']=df['Adj Close'].pct_change()
    df['cum_return']=(1+df['return']).cumprod()
    n=len(df)/252
    #Last value of the series of cum_return -> [-1]
    CAGR =(df['cum_return'][-1])**(1/n)-1
    return CAGR

for i in stocks:
    print('CAGR for{}={}'.format(i,CAGR(entire_data[i])))
def volatility(DF): #Standard deviation
    df=DF.copy()
    df=entire_data['AMZN'].copy()
    df['return']=df['Adj Close'].pct_change()
    vol=df['return'].std()*np.sqrt(252)
    return vol

for i in entire_data:
    print('Volatility of {} = {}'.format(i,volatility(entire_data[i])))

def sharpe(DF, rf):
    df=DF.copy()
    sharpe = (CAGR(df) - rf)/volatility(df)
    return sharpe

def sortino(DF, rf):
    df=DF.copy()
    df['return']=df['Adj Close'].pct_change()
    neg_return = np.where(df['return']>0,0,df['return'])

#neg_return[neg_return!=0] choose negative return not equal to 0
# function was Numpy series, and has NaN value -> convert to Pandas
#convert this numpy function (neg_return[neg_return!=0].std()) to pandas by adding pd.Series(..).std()
    neg_volatility = pd.Series(neg_return[neg_return!=0]).std()*np.sqrt(252)
    return  (CAGR(df) - rf)/neg_volatility

for i in entire_data:
    print('Sharpe for {} = {}'.format(i,sharpe(entire_data[i],0.03)))
    print('Sortino for {} = {}'.format(i,sortino(entire_data[i],0.03)))

