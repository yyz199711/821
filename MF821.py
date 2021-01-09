#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:23:36 2020

@author: Yuyang Zhang
"""
from scipy.interpolate import splrep, splev
import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

''' Read the SP500 data from Yahoo Finance, then we divide the price by 100 because we want to solve the PDE easier , the original parameter we can recalculate'''
symbol = '^GSPC'
source = 'yahoo'
start_date = '2020-02-20'
end_date = '2020-04-24'
SPX = web.DataReader(symbol, source, start_date, end_date)
# SPX = yf.download('^GSPC', '2020-02-20', '2020-04-24')
# print(SPX)

SPX = SPX['Adj Close']/100
rets = np.log(SPX / SPX.shift(1))
# rets = rets.dropna()
rets.dropna(inplace = True)
# print(rets.size)
# print(round(rets, 4))
### Fit a normal distribution and draw the histogram
mu, std = norm.fit(rets)
n_observation_rets = rets.size
rets.plot.hist(grid=True, density = True, bins = round(math.sqrt(n_observation_rets)), color='g', alpha = 0.6)
plt.xlabel('log-returns')
plt.ylabel("Density")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results:" + r'$\mu$' + " = %f," % mu + r'$\sigma$' + " = %f" % std
plt.title(title)
plt.show()

### Part 1 Form the training dataset and test dataset
n_observation_price = SPX.size 
n_train = round(0.5 * n_observation_price)
n_test = n_observation_price - n_train
SPX_train = SPX[0:n_train]
SPX_test = SPX[n_train:]
# print(SPX_train)
# print(SPX_test)

### Part 2 Fit the SP500 price as a linear function of time
import statsmodels.formula.api as smf
SPX_train_price = SPX_train.reset_index()["Adj Close"]
train_data = pd.DataFrame({"time" : range(n_train), "Price" : SPX_train_price})
reg_model = smf.ols(formula = "Price ~ time", data = train_data).fit()
b, a = reg_model.params
# print(reg_model.summary())
# print(a, b)
SPX_ave = pd.Series(reg_model.predict(), index = SPX_train.index)
# print(SPX_ave)



### Part 3 Fit an AR(1) model for dY 
Y = SPX_train - SPX_ave
diff_Y = Y - Y.shift(1)
diff_Y = diff_Y.dropna()
# print(diff_Y)
from statsmodels.tsa.arima_model import ARMA
diff_Y_train = diff_Y.reset_index()[0]
# print(diff_Y_train)
model_diff_Y = ARMA(diff_Y_train, order = (1, 0)).fit(trend = 'c', method = 'mle')
# print(model_diff_Y.summary())
A, B = model_diff_Y.params

### Part 4 Extract a continuous time mean reverting model for Y and calculate the sample volatility of Y
kappa = 1 - B
# theta = A/kappa

Y_theta = (Y - B * Y.shift(1)).dropna()
theta = Y_theta.mean() / kappa

sigma = Y.std()

# print(kappa, theta, sigma)
SPX_rolling_mean = SPX.rolling(window = 1).mean()
# print(SPX_rolling_mean, SPX_rolling_mean[-1])
# print(n_observation_price)
SPX_ave_test = SPX_rolling_mean[(n_train - 1) :]
Y_test = SPX_test - SPX_ave_test.shift(1)
Y_test.dropna(inplace = True)


''' numerically solve the PDE and get the upper/lower boundary price'''
SPX_rolling_test = SPX.rolling(3).mean()
SPX_rolling_test = SPX_rolling_test.shift(1)

SPX_rolling_test = SPX_rolling_test[-23:]
def optimal_band_sell():
    global kappa, theta, sigma, a, b
    c = 0.006
    # A, w, flag, inde, u_end = engin_A()
    M = 500
    T = 22
    N = 200
    S_max = 50
    S_min = c
    h_s = (S_max - c) / M
    h_t = T / N
    # t = h_t * np.linspace(0, N, N+1)
    s = [c + h_s * i for i in range(M+1)]
    t = [h_t * j for j in range(N+1)]
    # print(s)
    
    optimal_value = list(np.zeros((M-1,1)) for i in range(N+1))
    
    optimal_value[N] = np.array([s[i] - c for i in range(1,M)]).reshape((M-1,1))
    optimal_value[N] = np.matrix(optimal_value[N])
    
    
    B_mat = list(0 for i in range(N))
    
    for j in range(len(B_mat)) :
        B_j = np.zeros((M-1,M-1))
        for i in range(1, M):
            d_i_j = 1+sigma**2*h_t/h_s**2
            u_i_j = -sigma ** 2 * h_t / (2 * h_s ** 2) - (kappa*(theta+a*t[j]+b-s[i])+a)* h_t / (2 * h_s)
            l_i_j = -sigma ** 2 * h_t / (2 * h_s ** 2) + (kappa*(theta+a*t[j]+b-s[i])+a)* h_t / (2 * h_s)
            B_j[i-1][i-1] = d_i_j
            if i >= 2 :
                B_j[i-1][i-2] = l_i_j
            if i <= M-2 :
                B_j[i-1][i] = u_i_j
        
        B_mat[j] = np.matrix(B_j)
    
    
    b_mat = list(0 for i in range(N))
    
    for k in range(len(b_mat)) :
        b_j = np.zeros((M-1,1))
        b_j[M-2][0] = -sigma ** 2 * h_t / (2 * h_s ** 2) - (kappa*(theta+a*t[k]+b-s[M-1])+a)* h_t / (2 * h_s)
        b_j[M-2][0] = b_j[M-2][0] * (S_max - c)
        b_j = np.matrix(b_j)
    
    boundary_price = np.zeros(N)
    for p in range(N) :
        B_inv = np.linalg.pinv(B_mat[N-p-1])
        call = B_inv * (optimal_value[N-p] - b_mat[N-p-1])
        b_price = []
        for q in range(len(call)) :
            e_at_this_time = s[q+1] - c
            if e_at_this_time >= call[q][0]:
                call[q][0] = e_at_this_time
                b_price += [s[q+1]]
        b_price = min(b_price)
        boundary_price[N-p-1] = b_price
        
        optimal_value[N-p-1] = call
    
    
    return np.array(boundary_price), optimal_value
    
    
        


def optimal_band_buy() :
    bb_price, H = optimal_band_sell()
    global kappa, theta, sigma, a, b
    c = 0.006
    # A, w, flag, inde, u_end = engin_A()
    M = 500
    T = 22
    N = 200
    S_max = 50
    S_min = c
    h_s = (S_max - c) / M
    h_t = T / N
    # t = h_t * np.linspace(0, N, N+1)
    s = [c + h_s * i for i in range(M+1)]
    t = [h_t * j for j in range(N+1)]
    # print(s)
    
    optimal_value = list(np.zeros((M-1,1)) for i in range(N+1))
    
    optimal_value[N] = np.array([-2*c for i in range(1,M)]).reshape((M-1,1))
    optimal_value[N] = np.matrix(optimal_value[N])
    
    
    B_mat = list(0 for i in range(N))
    
    for j in range(len(B_mat)) :
        B_j = np.zeros((M-1,M-1))
        for i in range(1, M):
            d_i_j = 1+sigma**2*h_t/h_s**2
            u_i_j = -sigma ** 2 * h_t / (2 * h_s ** 2) - (kappa*(theta+a*t[j]+b-s[i])+a)* h_t / (2 * h_s)
            l_i_j = -sigma ** 2 * h_t / (2 * h_s ** 2) + (kappa*(theta+a*t[j]+b-s[i])+a)* h_t / (2 * h_s)
            B_j[i-1][i-1] = d_i_j
            if i >= 2 :
                B_j[i-1][i-2] = l_i_j
            if i <= M-2 :
                B_j[i-1][i] = u_i_j
        
        B_mat[j] = np.matrix(B_j)
    
    
    b_mat = list(0 for i in range(N))
    
    for k in range(len(b_mat)) :
        b_j = np.zeros((M-1,1))
        b_j[0][0] =  -sigma ** 2 * h_t / (2 * h_s ** 2) + (kappa*(theta+a*t[j]+b-s[1])+a)* h_t / (2 * h_s)
        b_j[0][0] = b_j[0][0] * (-2 * c)
        b_j[M-2][0] = -sigma ** 2 * h_t / (2 * h_s ** 2) - (kappa*(theta+a*t[k]+b-s[M-1])+a)* h_t / (2 * h_s)
        b_j[M-2][0] = b_j[M-2][0] * (-2 * c)
        b_j = np.matrix(b_j)
    
    boundary_price = np.zeros((M-1,N))
    for p in range(N) :
        B_inv = np.linalg.pinv(B_mat[N-p-1])
        call = B_inv * (optimal_value[N-p] - b_mat[N-p-1])
        for q in range(len(call)) :
            H_p = H[N-p-1]
            e_at_this_time = H_p[q][0]-s[q+1] - c
            if e_at_this_time >= call[q][0]:
                call[q][0] = e_at_this_time
                boundary_price[q][N-p-1] = s[q+1]
        
        optimal_value[N-p-1] = call
    
    
    return boundary_price, optimal_value
    


def plot_boundary_price_S(H,G,up_boundary,down_boundary) :
    T = 22
    N = 200
    h_t = T / N
    t = np.array([h_t * i for i in range(N)])
    
    plt.plot(t, up_boundary)
    plt.plot(t, down_boundary)
    plt.xlabel('Time')
    plt.ylabel('The Boundary Price with Respect to Time')
    plt.legend(['Sell_boundary', 'Buy_boundary'])
    plt.show()

def boundary_price_Y(up_boundary, H, down_boundary, G):
    global a, b
    # up_boundary, H = optimal_band_sell()
    # = optimal_band_buy()
    T = 22
    N = 200
    h_t = T / N
    t = np.array([h_t * i for i in range(N)])
    Y_up = np.zeros(N)
    Y_down = np.zeros(N)
    for i in range(N) :
        S_ave = a * t[i] + b
        Y_up[i] = up_boundary[i] - S_ave
        Y_down[i] = down_boundary[i] - S_ave
    
    
    return Y_up, Y_down

def strategy(SPX_rolling_test, Y_up, Y_down) :
    S_roll = SPX_rolling_test[:-1]
    T = 22
    N = 200
    h_t = T/N
    t = np.array([h_t * i for i in range(N)])
    tck_1 = splrep(t, Y_up)
    tck_2 = splrep(t, Y_down)
    t_1 = np.linspace(0,21,22)
    S_up_test = splev(t_1, tck_1) + np.array(S_roll)
    S_down_test = splev(t_1, tck_2) + np.array(S_roll)
        
    
    
    return S_up_test, S_down_test

# def plot_test_up(S_up_test, )
def optimal_band_strategy(S_up_test, S_down_test, SPX_test) :
    c = 0.006
    n = len(S_up_test)  
    SPX_test = pd.DataFrame(SPX_test)
    
    SPX_test['Position'] = 0
    
    for i in range(n) :
        if SPX_test['Adj Close'].iloc[i] >= S_up_test[i] :
            SPX_test['Position'].iloc[i:] = -1
        
        elif SPX_test['Adj Close'].iloc[i] <= S_down_test[i] :
            SPX_test['Position'].iloc[i:] = 1
    
    index = []
    if SPX_test['Position'].iloc[0] != 0 :
        index += [0]

    
    for j in range(n-1) :
        if SPX_test['Position'].iloc[j] != SPX_test['Position'].iloc[j+1] :
            index += [j+1]
    
        
    ret = []
    for k in range(len(index)-1) :
        if SPX_test['Position'].iloc[index[k]] == 1 :
            ret += [(SPX_test['Adj Close'].iloc[index[k+1]]-SPX_test['Adj Close'].iloc[index[k]]-2*c)/(SPX_test['Adj Close'].iloc[index[k]]+c)]
            
        # elif SPX_test['Position'].iloc[index[k]] == -1 :
        #     ret += [(SPX_test['Adj Close'].iloc[index[k]]-SPX_test['Adj Close'].iloc[index[k+1]])/SPX_test['Adj Close'].iloc[index[k]]]
            
    
    if SPX_test['Position'].iloc[index[-1]] == 1 :
        ret += [(SPX_test['Adj Close'].iloc[-1]-SPX_test['Adj Close'].iloc[index[-1]]-2*c)/(SPX_test['Adj Close'].iloc[index[-1]]+c)]
     
    # elif index[-1] == -1:
    #     ret += [(SPX_test['Adj Close'].iloc[index[-1]]-SPX_test['Adj Close'].iloc[-1])/SPX_test['Adj Close'].iloc[index[-1]]]
    
    cum_ret = 1000
    for m in ret :
        cum_ret *= (1 + m)
    
    return cum_ret
        
             
def delay_optimal_band_strategy(S_up_test, S_down_test, SPX_test):
    c = 0.006
    n = len(S_up_test)   
    
    SPX_test = pd.DataFrame(SPX_test)
    
    SPX_test['Position'] = 0
    
    for i in range(n) :
        if SPX_test['Adj Close'].iloc[i] >= S_up_test[i] :
            SPX_test['Position'].iloc[i:] = -1
        
        elif SPX_test['Adj Close'].iloc[i] <= S_down_test[i] :
            SPX_test['Position'].iloc[i:] = 1
    
    SPX_test['Position']= SPX_test['Position'].shift(1)
    
    SPX_test['Position'].iloc[0] = 0
    SPX_test['Position'].iloc[-1] = SPX_test['Position'].iloc[-2]
    index = []
    
    for j in range(n-1) :
        if SPX_test['Position'].iloc[j] != SPX_test['Position'].iloc[j+1] :
            index += [j+1]
        
    ret = []
    for k in range(len(index)-1) :
        if SPX_test['Position'].iloc[index[k]] == 1 :
            ret += [(SPX_test['Adj Close'].iloc[index[k+1]]-SPX_test['Adj Close'].iloc[index[k]]-2*c)/(SPX_test['Adj Close'].iloc[index[k]]+c)]
            
        # elif SPX_test['Position'].iloc[index[k]] == -1 :
        #     ret += [(SPX_test['Adj Close'].iloc[index[k]]-SPX_test['Adj Close'].iloc[index[k+1]]-c)/SPX_test['Adj Close'].iloc[index[k]]]
            
    
    if SPX_test['Position'].iloc[index[-1]] == 1 :
        ret += [(SPX_test['Adj Close'].iloc[-1]-SPX_test['Adj Close'].iloc[index[-1]]-2*c)/(SPX_test['Adj Close'].iloc[index[-1]]+c)]
     
    # elif index[-1] == -1:
    #     ret += [(SPX_test['Adj Close'].iloc[index[-1]]-SPX_test['Adj Close'].iloc[-1]-c)/SPX_test['Adj Close'].iloc[index[-1]]]
    
    cum_ret = 1000
    for m in ret :
        cum_ret *= (1 + m)
    
    return cum_ret
        

def adhoc_band_strategy(x, theta, sigma, Y_test, SPX_test) :
    c = 0.006
    Y_up = theta + x[0] * sigma
    Y_down = theta + x[1] * sigma
    n = len(Y_test)
    Position = np.zeros(n-1)
    for i in range(len(Y_test)-1) :
        if Y_test[i] >= Y_up :
            Position[i:] = -1
        elif Y_test[i] <=Y_down :
            Position[i:] = 1
    index = []
    if Position[0] != 0 :
        index += [0]

    
    for j in range(n-2) :
        if Position[j] != Position[j+1] :
            index += [j+1]
    ret = []

    for k in range(len(index)-1) :
        if Position[index[k]] == 1 :
            ret += [(SPX_test[index[k+1]]-SPX_test[index[k]]-2*c)/(SPX_test[index[k]]+c)]
            
        # elif SPX_test['Position'].iloc[index[k]] == -1 :
        #     ret += [(SPX_test['Adj Close'].iloc[index[k]]-SPX_test['Adj Close'].iloc[index[k+1]]-c)/SPX_test['Adj Close'].iloc[index[k]]]
            
    
    if Position[index[-1]] == 1 :
        ret += [(SPX_test[-1]-SPX_test[index[-1]]-2*c)/(SPX_test[index[-1]]+c)]
     
    # elif index[-1] == -1:
    #     ret += [(SPX_test['Adj Close'].iloc[index[-1]]-SPX_test['Adj Close'].iloc[-1]-c)/SPX_test['Adj Close'].iloc[index[-1]]]
    
    cum_ret = 1000
    for m in ret :
        cum_ret *= (1 + m)
    
    return cum_ret

def find_optimal_adhoc(theta, sigma, Y_test, SPX_test):
    x_1 = np.linspace(0,0.5,100)
    x_2 = np.linspace(-0.5,0.5, 100)
    z = np.zeros((100, 100))
    for i in range(100) :
        for j in range(100):
            z[i][j] = adhoc_band_strategy([x_1[i],x_2[j]], theta, sigma, Y_test, SPX_test)
    
    return z



def adhoc_band_delay_strategy(x, theta, sigma, Y_test, SPX_test) :
    c = 0.006
    Y_up = theta + x[0] * sigma
    Y_down = theta + x[1] * sigma
    n = len(Y_test)
    Position = np.zeros(n-1)
    for i in range(len(Y_test)-1) :
        if Y_test[i] >= Y_up :
            Position[i:] = -1
        elif Y_test[i] <= Y_down :
            Position[i:] = 1
    Position = np.hstack((np.zeros(1), Position[:-1]))
    # print(Position)
    
    index = []
    SPX_test = pd.DataFrame(SPX_test)
    
    for j in range(n-2) :
        if Position[j] != Position[j+1] :
            index += [j+1]
    ret = []

    for k in range(len(index)-1) :
        if Position[index[k]] == 1 :
            ret += [(SPX_test['Adj Close'].iloc[index[k+1]]-SPX_test['Adj Close'].iloc[index[k]]-2*c)/(SPX_test['Adj Close'].iloc[index[k]]+c)]
            
        # elif SPX_test['Position'].iloc[index[k]] == -1 :
        #     ret += [(SPX_test['Adj Close'].iloc[index[k]]-SPX_test['Adj Close'].iloc[index[k+1]]-c)/SPX_test['Adj Close'].iloc[index[k]]]
            
    
    if Position[index[-1]] == 1 :
        ret += [(SPX_test['Adj Close'].iloc[-1]-SPX_test['Adj Close'].iloc[index[-1]]-2*c)/(SPX_test['Adj Close'].iloc[index[-1]]+c)]
     
    # elif index[-1] == -1:
    #     ret += [(SPX_test['Adj Close'].iloc[index[-1]]-SPX_test['Adj Close'].iloc[-1]-c)/SPX_test['Adj Close'].iloc[index[-1]]]
    
    cum_ret = 1000
    for m in ret :
        cum_ret *= (1 + m)
    
    return cum_ret

def find_optimal_delay_adhoc(theta, sigma, Y_test, SPX_test):
    x_1 = np.linspace(0,1,50)
    x_2 = np.linspace(-0.5,1, 100)
    z = np.zeros((50, 100))
    for i in range(50) :
        for j in range(100):
            z[i][j] = adhoc_band_delay_strategy([x_1[i],x_2[j]], theta, sigma, Y_test, SPX_test)
    
    return z

def adhoc_band_delay_strategy_sell(x, theta, sigma, Y_test, SPX_test) :
    c = 0.006
    Y_up = theta + x[0] * sigma
    Y_down = theta + x[1] * sigma
    n = len(Y_test)
    Position = np.zeros(n-1)
    for i in range(len(Y_test)-1) :
        if Y_test[i] >= Y_up :
            Position[i:] = -1
        elif Y_test[i] <= Y_down :
            Position[i:] = 1
    Position = np.hstack((np.zeros(1), Position[:-1]))
    Position[1] = 0
    # print(Position)
    
    index = []
    SPX_test = pd.DataFrame(SPX_test)
    
    for j in range(n-2) :
        if Position[j] != Position[j+1] :
            index += [j+1]
    ret = []

    for k in range(len(index)-1) :
        if Position[index[k]] == 1 :
            ret += [(SPX_test['Adj Close'].iloc[index[k+1]]-SPX_test['Adj Close'].iloc[index[k]]-2*c)/(SPX_test['Adj Close'].iloc[index[k]]+c)]
            
        # elif SPX_test['Position'].iloc[index[k]] == -1 :
        #     ret += [(SPX_test['Adj Close'].iloc[index[k]]-SPX_test['Adj Close'].iloc[index[k+1]]-c)/SPX_test['Adj Close'].iloc[index[k]]]
    
    units = 1000/SPX_test['Adj Close'].iloc[1]
    cum_ret = 1000 * (SPX_test['Adj Close'].iloc[1]-c) / SPX_test['Adj Close'].iloc[1]
    if ret != [] :
        for m in ret :
            cum_ret *= (1 + m)
    
     
    S_end = SPX_test['Adj Close'].iloc[-1] 
        
    if Position[index[-1]] == -1 :
        profit = cum_ret - units * (S_end + c)
        
    else :
        units_new = cum_ret/(SPX_test['Adj Close'].iloc[index[-1]] + c)
        if units_new >= units :
            profit = (units_new - units) * (S_end - c)
        else :
            profit = (units_new - units) * (S_end + c)
    
    
    return profit

def adhoc_band_strategy_sell(x, theta, sigma, Y_test, SPX_test) :
    c = 0.006
    Y_up = theta + x[0] * sigma
    Y_down = theta + x[1] * sigma
    n = len(Y_test)
    Position = np.zeros(n-1)
    for i in range(len(Y_test)-1) :
        if Y_test[i] >= Y_up :
            Position[i:] = -1
        elif Y_test[i] <=Y_down :
            Position[i:] = 1
    index = []
    Position[0] = 0
    
    
    for j in range(n-2) :
        if Position[j] != Position[j+1] :
            index += [j+1]
    ret = []
    

    for k in range(len(index)-1) :
        if Position[index[k]] == 1 :
            ret += [(SPX_test[index[k+1]]-SPX_test[index[k]]-2*c)/(SPX_test[index[k]]+c)]
            
        # elif SPX_test['Position'].iloc[index[k]] == -1 :
        #     ret += [(SPX_test['Adj Close'].iloc[index[k]]-SPX_test['Adj Close'].iloc[index[k+1]]-c)/SPX_test['Adj Close'].iloc[index[k]]]
    cum_ret = 1000 * (SPX_test[0]-c) / SPX_test[0]
    if ret != [] :
        for m in ret :
            cum_ret *= (1 + m)
        
    units = 1000/SPX_test[0]
    
    S_end = SPX_test[-1]
    
    if Position[index[-1]] == 1 :
        units_new = cum_ret/(SPX_test[index[-1]]+c)
        if units_new >= units :
            profit = (units_new - units)*(S_end -c)
        else :
            profit = (units_new - units)*(S_end +c)
    
    
    else :
        profit = cum_ret - units * (S_end +c)
    
    return profit
     
def find_optimal_adhoc_sell(theta, sigma, Y_test, SPX_test):
    x_1 = np.linspace(0,2,50)
    x_2 = np.linspace(-0.5,1, 50)
    z = np.zeros((50, 50))
    for i in range(50) :
        for j in range(50):
            z[i][j] = adhoc_band_strategy_sell([x_1[i],x_2[j]], theta, sigma, Y_test, SPX_test)
    
    return z

def find_optimal_adhoc_delay_sell(theta, sigma, Y_test, SPX_test):
    x_1 = np.linspace(0,1,50)
    x_2 = np.linspace(-0.5,0.5, 50)
    z = np.zeros((50, 50))
    for i in range(50) :
        for j in range(50):
            z[i][j] = adhoc_band_delay_strategy_sell([x_1[i],x_2[j]], theta, sigma, Y_test, SPX_test)
    
    return z


    



def optimal_band_strategy_sell(S_up_test, S_down_test, SPX_test) :
    c = 0.006
    n = len(S_up_test)  
    SPX_test = pd.DataFrame(SPX_test)
    
    SPX_test['Position'] = 0
    
    for i in range(n) :
        if SPX_test['Adj Close'].iloc[i] >= S_up_test[i] :
            SPX_test['Position'].iloc[i:] = -1
        
        elif SPX_test['Adj Close'].iloc[i] <= S_down_test[i] :
            SPX_test['Position'].iloc[i:] = 1
    
    index = []
    if SPX_test['Position'].iloc[0] != 0 :
        index += [0]

    
    for j in range(n-1) :
        if SPX_test['Position'].iloc[j] != SPX_test['Position'].iloc[j+1] :
            index += [j+1]
    
        
    ret = []
    for k in range(len(index)-1) :
        if SPX_test['Position'].iloc[index[k]] == 1 :
            ret += [(SPX_test['Adj Close'].iloc[index[k+1]]-SPX_test['Adj Close'].iloc[index[k]]-2*c)/(SPX_test['Adj Close'].iloc[index[k]]+c)]
            
        # elif SPX_test['Position'].iloc[index[k]] == -1 :
        #     ret += [(SPX_test['Adj Close'].iloc[index[k]]-SPX_test['Adj Close'].iloc[index[k+1]])/SPX_test['Adj Close'].iloc[index[k]]]
            
    
     
    # elif index[-1] == -1:
    #     ret += [(SPX_test['Adj Close'].iloc[index[-1]]-SPX_test['Adj Close'].iloc[-1])/SPX_test['Adj Close'].iloc[index[-1]]]
    
    cum_ret = 1000
    for m in ret :
        cum_ret *= (1 + m)
    
    units = 1000 / (SPX_test['Adj Close'].iloc[0] - c)
    
    S_end = SPX_test['Adj Close'].iloc[-1]
    
    
    if SPX_test['Position'].iloc[index[-1]] == 1 :
        units_get = cum_ret / (S_end + c)
        
        if units_get >= units :
            profit = (units_get - units) * (S_end - c)
        else :
            profit = (units_get - units) * (S_end + c)
    
    if SPX_test['Position'].iloc[index[-1]] == -1 :
        profit = cum_ret - units * (S_end + c)
    
    
    return profit

def delay_optimal_band_strategy_sell(S_up_test, S_down_test, SPX_test) :
    c = 0.006
    n = len(S_up_test)   
    
    SPX_test = pd.DataFrame(SPX_test)
    
    SPX_test['Position'] = 0
    
    for i in range(n) :
        if SPX_test['Adj Close'].iloc[i] >= S_up_test[i] :
            SPX_test['Position'].iloc[i:] = -1
        
        elif SPX_test['Adj Close'].iloc[i] <= S_down_test[i] :
            SPX_test['Position'].iloc[i:] = 1
    
    SPX_test['Position']= SPX_test['Position'].shift(1)
    
    SPX_test['Position'].iloc[0] = 0
    SPX_test['Position'].iloc[-1] = SPX_test['Position'].iloc[-2]
    index = []
    
    for j in range(n-1) :
        if SPX_test['Position'].iloc[j] != SPX_test['Position'].iloc[j+1] :
            index += [j+1]
    # print(index)
        
    ret = []
    for k in range(len(index)-1) :
        if SPX_test['Position'].iloc[index[k]] == 1 :
            ret += [(SPX_test['Adj Close'].iloc[index[k+1]]-SPX_test['Adj Close'].iloc[index[k]]-2*c)/SPX_test['Adj Close'].iloc[index[k]]+c]
            
        # elif SPX_test['Position'].iloc[index[k]] == -1 :
        #     ret += [(SPX_test['Adj Close'].iloc[index[k]]-SPX_test['Adj Close'].iloc[index[k+1]]-c)/SPX_test['Adj Close'].iloc[index[k]]]
            
    
     
    # elif index[-1] == -1:
    #     ret += [(SPX_test['Adj Close'].iloc[index[-1]]-SPX_test['Adj Close'].iloc[-1]-c)/SPX_test['Adj Close'].iloc[index[-1]]]
    
    cum_ret = 1000
    # print(ret)
    if ret != [] :
        for m in ret :
            cum_ret *= (1 + m)
    
    units = 1000 / (SPX_test['Adj Close'].iloc[1] - c)
    
    S_end = SPX_test['Adj Close'].iloc[-1]
    
    
    
    
        
    if SPX_test['Position'].iloc[index[-1]] == 1 :
        units_new = cum_ret / (SPX_test['Adj Close'].iloc[index[-1]] + c)
        if units_new >= units :
            profit = (units_new - units) * (S_end - c)
        else :
            profit = (units_new - units) * (S_end + c)
        
    else :
        profit = cum_ret - units * (S_end + c)
    
    return profit
            
    
        
if __name__ == "__main__" :
    
    up_boundary, H = optimal_band_sell()
    
    down_boundary, G = optimal_band_buy()
    down_boundary_T_1 = down_boundary.T[-1]
    down_boundary_T_1 = down_boundary_T_1[:int(0.5*len(down_boundary_T_1))]
    down_boundary_T_1 = max(down_boundary_T_1)
    down_boundary = down_boundary.T[:-1]
    down_boundary = down_boundary.T
    stable_range = int(0.7*len(down_boundary))
    down_boundary = down_boundary[:stable_range]
    down_boundary_1 = [max(down_boundary.T[i]) for i in range(len(down_boundary.T))]+[down_boundary_T_1]
    down_boundary = np.array(down_boundary_1)
    
    plot_boundary_price_S(H,G,up_boundary,down_boundary)
    plt.show()

    
    Y_up, Y_down = boundary_price_Y(up_boundary, H, down_boundary, G)
    t = np.linspace(0,22,200)
    plt.plot(t, Y_up); plt.plot(t, Y_down)
    plt.title('Y Boundary')
    plt.legend(['Y_up', 'Y_down'])
    plt.show()
    t_1 = np.linspace(0,21,22)
    S_up_test, S_down_test = strategy(SPX_rolling_test, Y_up, Y_down)
    Stest = np.array(SPX_test[:-1])
    
    
    plt.plot(t_1, S_up_test); plt.plot(t_1, S_down_test);plt.plot(t_1, Stest)
    plt.title('S Boundary for Testing')
    plt.legend(['S_up', 'S_down','SPX_test'])
    plt.show()

    '''buy and sell strategy for optimal band, profit is 21.94'''
    profit = optimal_band_strategy(S_up_test, S_down_test, SPX_test)
    '''delay buy and sell strategy for optimal band, profit is 12.94'''
    profit_1 = delay_optimal_band_strategy(S_up_test, S_down_test, SPX_test)
    '''Short and Buy back X units strategy for optimal band, profit is -137.71'''
    profit_2 = optimal_band_strategy_sell(S_up_test, S_down_test, SPX_test)
    '''delay Short and Buy back X units strategy for optimal band, profit is -133.05'''
    profit_3 = delay_optimal_band_strategy_sell(S_up_test, S_down_test, SPX_test)
    
    SP_test = np.array(SPX_test)
    YP_test = np.array(Y_test)
    
    # '''Buy and Hold strategy'''
    profit_bh = (SPX_test[-1]-SPX_test[0]-0.012)/(SPX_test[0]+0.006) * 1000
    
    # '''Buy and Hold delay strategy'''
    profit_bh_delay = (SPX_test[-1]-SPX_test[1]-0.012)/(SPX_test[1]+0.006) * 1000 * (SPX_test[1]-0.006)/(SPX_test[1])
    
    z_1 = adhoc_band_delay_strategy([0.28,0.27], theta, sigma, Y_test, SPX_test)
    ''' The Profit is -25.312087454017956'''
    z_2 = adhoc_band_delay_strategy_sell([0.28,0.27], theta, sigma, Y_test, SPX_test)
    ''' The Profit is -171.6889621051606'''
    
    z_3 = find_optimal_delay_adhoc(theta, sigma, Y_test, SPX_test)
    '''Buy and sell delay Optimal set is ((0.62,0.72], [-0.38,-0.26)), Optimal value is 105'''
     
    
    
    z_4 = find_optimal_adhoc_delay_sell(theta, sigma, Y_test, SPX_test)
    '''Short and buy back delay Optimal set is ((0.62,0.72],[-0.38,-0.26)) optimal value is -40.652201540097494'''
    
    z_5 = find_optimal_adhoc(theta, sigma, Y_test, SPX_test)
    '''Buy and sell strategy Optimal set is ([0.265,0.5],[0.26,0.5]) Optimal value is 266.260408425663'''
    
    z_6 = find_optimal_adhoc_sell(theta, sigma, Y_test, SPX_test)
    '''Short and buy back strategy Optimal set is [0.28,0.36], [0.26,0.36] Optimal value is 107.07891025484975'''


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        


