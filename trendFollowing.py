### Quantiacs Trend Following Trading System Example
# import necessary Packages below:
import qtb
import statsmodels 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from statsmodels.tsa.stattools import coint


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''
    pos = [0,0]
    nMarkets=CLOSE.shape[1]
    S1 = CLOSE[:,0]
    S2 = CLOSE[:,1]
    ratios = S1/S2
    ma1 = ratios[-6:].mean()
    ma2 = ratios.mean()
    std = ratios.std()
    zscore = (ma1-ma2)/std
    if zscore > 1:
        pos = [-1,ratios[-1]]
    elif zscore < -1:
        pos = [1,-ratios[-1]]
    elif abs(zscore) < 0.75:
        pos = [0,0]
#    periodLonger=200
#    periodShorter=40
#
#    # Calculate Simple Moving Average (SMA)
#    smaLongerPeriod=numpy.nansum(CLOSE[-periodLonger:,:],axis=0)/periodLonger
#    smaShorterPeriod=numpy.nansum(CLOSE[-periodShorter:,:],axis=0)/periodShorter
#
#    longEquity= smaShorterPeriod > smaLongerPeriod
#    shortEquity= ~longEquity
#
#    pos=numpy.zeros(nMarkets)
#    pos[longEquity]=1
#    pos[shortEquity]=-1

    weights = pos/np.nansum(abs(pos))

    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings= {}

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts

    settings['markets']  = ['CASH','DTE', 'PPL']
#    settings['beginInSample'] = '20120506'
#    settings['endInSample'] = '20150506'
    settings['lookback']= 60
    settings['budget']= 10**6
    settings['slippage']= 0.05

    return settings


def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.02:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

def zscores(series):
    return (series - series.mean())/np.std(series)
    

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    utilityMarkets = ["AES","AEE", "AEP","CNP","CMS","ED","D","DTE","DUK","EIX","ETR","EXC","FE","NEE","NI","NRG","PCG","PSX","PNW","PPL","PEG","SCG","SRE","SO","WEC"]
    data = qtb.loadData(utilityMarkets, qtb.REQUIRED_DATA,False,'20120506','20150506')
    closeData = pd.DataFrame(data['CLOSE'],columns=utilityMarkets)
#    scores,pvalues,pairs = find_cointegrated_pairs(closeData)
#    m = [0,0.2,0.4,0.6,0.8,1]
#    seaborn.heatmap(pvalues, xticklabels=utilityMarkets, yticklabels=utilityMarkets, cmap='RdYlGn_r',mask = (pvalues >= 0.98))
#    plt.show()
    S1 = closeData['DTE']
    S2 = closeData['PPL']
    ratios = closeData['DTE'] / closeData['PPL']

    trainingnumber = len(ratios)*2//3
    train = ratios
    test = ratios[trainingnumber:]
    ratios_mavg5 = train.rolling(window=5,
                               center=False).mean()
    ratios_mavg60 = train.rolling(window=60,
                               center=False).mean()
    std_60 = train.rolling(window=60,
                        center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
    plt.figure(figsize=(15,7))
    plt.plot(train.index, train.values)
    plt.plot(ratios_mavg5.index, ratios_mavg5.values)
    plt.plot(ratios_mavg60.index, ratios_mavg60.values)
    plt.legend(['Ratio','5d Ratio MA', '60d Ratio MA'])
    plt.ylabel('Ratio')
    plt.show()
    plt.figure(figsize=(15,7))
    zscore_60_5.plot()
    plt.axhline(0, color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
    plt.show()
#    ma1 = ratios.rolling(window=5,
#                               center=False).mean()
#    ma2 = ratios.rolling(window=60,
#                               center=False).mean()
#    std = ratios.rolling(window=60,
#                        center=False).std()
#    zscore = (ma1-ma2)/std
#    money = 0
#    countS1 = 0
#    countS2 = 0
#    for i in range(61, len(ratios)):
#        # Sell short if the z-score is > 1
#        if zscore[i] > 1:
#            money += S1[i] - S2[i] * ratios[i]
#            countS1 -= 1
#            countS2 += ratios[i]
#            print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
#        # Buy long if the z-score is < 1
#        elif zscore[i] < -1:
#            money -= S1[i] - S2[i] * ratios[i]
#            countS1 += 1
#            countS2 -= ratios[i]
#            print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
#        # Clear positions if the z-score between -.5 and .5
#        elif abs(zscore[i]) < 0.75:
#            money += S1[i] * countS1 + S2[i] * countS2
#            countS1 = 0
#            countS2 = 0
#            print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
#            
#            
#    print(money)
  
    #[('DTE', 'PPL')]
    #results = qtb.runts(__file__)