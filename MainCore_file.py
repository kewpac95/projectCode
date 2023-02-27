import numpy as np
import pandas as pd
import nonlinshrink as nls
import glob
import os as os
import yfinance as yf
import datetime

from pytorch_lightning.callbacks import EarlyStopping


datetime.datetime.strptime
import pytorch_lightning as pl
import torch as t
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from nixtlats.models.esrnn.esrnn import ESRNN

import yfinance as yf
from yahoofinancials import YahooFinancials


from nixtlats.data.datasets.m4 import M4, M4Info, M4Evaluation
from nixtlats.data.tsdataset import TimeSeriesDataset
from nixtlats.data.tsloader import TimeSeriesLoader
from nixtlats.models.esrnn.esrnn import ESRNN


#the data I've downloaded by the yahoofinance
#I've tried to optimize hyperparameters by optuna, but the calculation time was tremendous
#It should be learning model for a half of the year
#the system is no so good, however
#The idea:
# take the yahoo finance
# make predictions on close price for the next week (5 days)
# make decision procedur:
#     if the close at 5'th day will be bigger than 1th then buy and hold 5 day (there fore put 1)
#     else sell (therefore put -1)
#     on the each week calculate this decision vector by inverce of the estimator of correlation of logarythmic rate of return
#     the estimator is no pure samle estimator but nonlinear shrinkage estimator given by: Ledoit and Wolf 2018
#     testing if this system makes a profit


        # Conculsions and what can be better (from my perspective):
        # 1.
        #     It should be tested on bias survival free data - but taking good bias survival free data for free it's difficult (but possiable however..)
        #     and making a program on it its hard, because there were some data in (for example) 5 days and that;s it
        #     it should be done by some masking or something
        # 2. the model esRNN maebe it could be better fitted, however I find that this model is no so good to predicted stoock marekt values it looks more or less like shifted values
        # The interesting observation is the fact that system created on esRNN seems to be goo in short selling, which is much harder (in general) than long, however
        # it can be caused by the bias-survival data
        


def random_portfolio(nonlinMat,expectedY_hat):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = expectedY_hat / 14
    w = np.asmatrix(rand_weights(p.shape[0]))
    C = nonlinMat

    mu = w * p
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(nonlinMat,expectedY_hat)
    return mu, sigma


def optimal_portfolio(nonlinMat,expectedY_hat):


    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(nonlinMat)
    pbar = opt.matrix(expectedY_hat/14)
    n = len(expectedY_hat)
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


os.chdir('C:\\Users\\User\\Downloads\\survivorship-free-spy-master\\sp500_actual\\data')
df = pd.read_csv('A.csv')
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
df.rename(columns = {'Adj Close':'Adj_close'}, inplace = True)

df['logRR'] = np.log(df.Adj_close) - np.log(df.Adj_close.shift(1))
df = df[df.Date > '2013-06-27']
df = df.iloc[1: , :]
df['id'] = 0
i = 1
csv_files.pop(0)

for f in csv_files:
    df1 = pd.read_csv(f)
    df1.rename(columns={'Adj Close': 'Adj_close'}, inplace=True)
    df1['logRR'] = np.log(df1.Adj_close) - np.log(df1.Adj_close.shift(1))
    df1 = df1[df1.Date > '2013-06-27']
    df1 = df1.iloc[1:, :]
    df1['id'] = i
    df = pd.concat([df,df1])
    i +=1
    print(i)
df.head()
np.shape(df)
df_list = [group for _, group in df.groupby('id', sort=False)] # we split the df on each unique id (to calculate the correlation matrix

new_df_to_corMat = df.pivot(index='Date', columns='id', values='logRR')
new_df_to_corMat_close = df.pivot(index='Date', columns='id', values='Close')

new_df_to_corMat

macDate = np.sort(df['Date'].unique())

# Starting date
last_index = len(macDate)
close_Df = df[['id', 'Date', 'Adj_close']].rename(columns={'Adj_close': 'y', 'id': 'unique_id', 'Date': 'ds'}, inplace=False)
input_size = 20

model = ESRNN(n_series=468,
              n_x=0, n_s=1,
              sample_freq=1,
              input_size=input_size,
              output_size=5,
              learning_rate=0.0025,
              lr_scheduler_step_size=6,
              lr_decay=0.08,
              per_series_lr_multip=0.8,
              gradient_clipping_threshold=20,
              rnn_weight_decay=0.1,
              level_variability_penalty=50,
              testing_percentile=50,
              training_percentile=50,
              cell_type='GRU',
              state_hsize=30,
              dilations=[[1, 2], [2, 6]],
              add_nl_layer=True,
              loss='SMYL',
              val_loss='SMAPE',
              seasonality=[5])

early_stop_callback = EarlyStopping(
    monitor='train_loss', # name of the metric to monitor
    min_delta=0.001, # minimum change to qualify as improvement
    patience=5, # number of epochs to wait for improvement
    verbose=False, # set to True to print the status
    mode='min') # set to 'min' for minimizing the loss, 'max' for maximizing the metric

trainer = pl.Trainer(max_epochs=15,
                     callbacks=[early_stop_callback],
                     deterministic=True,
                     log_every_n_steps=20)

array = close_Df['unique_id'].unique()
S_df1 = pd.Series(array).to_frame()
S_df1.columns = ['unique_id']
S_df1['category'] = 1
a =True
results = []

last_training_time = macDate[[0][0]]
last_train_date = macDate[[0][0]]
retrain_interval = np.timedelta64(125, 'D')
start_index = last_index - 2*255


for dt in list(range(start_index, last_index, 5)):
    train_date = macDate[dt-5]
    print(dt)
    test_date = macDate[dt]
    Y_df_test = close_Df[(close_Df["ds"]>train_date) & (close_Df["ds"]<=test_date)]
    Y_df_train = close_Df[(close_Df["ds"]<=train_date) ]
    y_test = Y_df_test.set_index(['unique_id', 'ds']).unstack().values
    Y_df_full = pd.concat([Y_df_train, Y_df_test]).sort_values(['unique_id', 'ds'], ignore_index=True)




    train_ts_dataset = TimeSeriesDataset(Y_df=Y_df_train, S_df=S_df1,
                                         input_size=input_size,
                                         output_size=5)

    test_ts_dataset = TimeSeriesDataset(Y_df=Y_df_full, S_df=S_df1,
                                        input_size=input_size,
                                        output_size=5)

    train_ts_loader = TimeSeriesLoader(dataset=train_ts_dataset,
                                       batch_size=16,
                                       shuffle=False)

    test_ts_loader = TimeSeriesLoader(dataset=test_ts_dataset,
                                      batch_size=1024,
                                      eq_batch_size=False,
                                      shuffle=False)
    current_time = test_date
    time_since_last_training = (current_time - last_training_time) / np.timedelta64(1, 's')
    if time_since_last_training > retrain_interval.astype('float64'):
        trainer.fit(model, train_ts_loader)
        last_training_time = test_date


    outputs = trainer.predict(model, train_ts_loader)
    _, y_hat, mask = zip(*outputs)
    y_hat = t.cat([y_hat_[:, -1] for y_hat_ in y_hat]).cpu().numpy()
    np.shape(y_hat)
    close_Df_SF_original = df[['id', 'Date', 'Adj_close']]
    # period = 2#4*12
    results.append((y_hat.reshape(-1), Y_df_test["y"].values))


type(results)
print(results)

arr = np.array(results)

# Divide the array into two lists based on the second dimension
estim_values = arr[:, 0, :]
true_values = arr[:, 1, :]
plt.plot(arr[:, :, 0])  #it makes a plot of predicted vs true
np.shape(estim_values[:,0])
np.shape(results)
np.shape(arr)
arr_reshaped = arr.reshape((102, 2, 468, 5))
WeeklyRR = (arr_reshaped[:, :, :, 4] - arr_reshaped[:, :, :, 0]) / arr_reshaped[:, :, :, 0]
signnum = np.sign(arr_reshaped[:, :, :, 4] - arr_reshaped[:, :, :, 0])

means = np.mean(arr_reshaped, axis=3)
np.shape(WeeklyRR)
plt.plot(signnum[:,:,0])
p = []
for e in range(0,468):
    p.append(np.sum(signnum[:, 1, e] * signnum[:, 0, e]))
np.mean(p)
# beacuse of nonstacionary of price signal I decided to use nonlinear shrinkage estimator for covariance matrix and thank's to this
# I am able to use small sample
#nls.shrink_cov

First_mat_to_cov = new_df_to_corMat.tail(last_index - start_index  + 60)
n_of_weeks = (last_index - start_index) / 5
yearly_rate_of_return = (1 + First_mat_to_cov.head(60)) ** (1/52) - 1
Non_lin_try = nls.shrink_cov(yearly_rate_of_return)


list_of_decition =[]
for number_of_week in range(0, int(n_of_weeks)):
    print(number_of_week)
    j = number_of_week * 5
    Non_lin_try = nls.shrink_cov(First_mat_to_cov.iloc[j:j+60])
    eee = np.linalg.inv(Non_lin_try) @ np.transpose(signnum[number_of_week, 0, :])
    list_of_decition.append(eee / np.sum(eee))

np.shape(list_of_decition)


MatrixToSTakeProfit = np.array(new_df_to_corMat.tail(last_index - start_index))
Matrix_close_price = new_df_to_corMat_close.tail(last_index - start_index)

Matrix_close_price['Date'] = pd.to_datetime(df['Date'])
weekly_prices = Matrix_close_price.resample('5D').last()

np.shape(weekly_prices)
weekly_log_returns = np.log(weekly_prices.pct_change() + 1)

# drop the first row since it will contain NaN due to the first week having no prior week to compare with
weekly_log_returns = weekly_log_returns.iloc[1:]
np.shape(weekly_log_returns)
# # reset the index to a column
# weekly_log_returns = weekly_log_returns.reset_index()

type(MatrixToSTakeProfit)

weekly_prices_first = np.array(Matrix_close_price.resample('5D').first().tail(int(n_of_weeks)))
weekly_prices_last = Matrix_close_price.resample('5D').last().tail(int(n_of_weeks))
w_log = np.array(weekly_log_returns.tail(int(n_of_weeks)))
np.shape(list_of_decition)


starting_cash = 1000000
# Loop through each week and stock
progres = []
progres2 = []
for week in range(int(n_of_weeks)):
    print(week)
    for stock in range(468):
        # Get the decision for this week and stock
        decision = np.array(list_of_decition)[week, stock]
        #print(decision)
        # Calculate the buy/sell price based on the decision
        if decision > 0.0:
            sign = 1.0
            price = 0.5 * 1.0 * np.abs(decision) * weekly_prices_first[week, stock]
            starting_cash -= price
        elif decision < 0.0:
            sign = -1
            price = np.abs(decision) * weekly_prices_first[week, stock]
            starting_cash += price
        else:  # hold
            price = 0
            sign = 0

        # Calculate the return based on the true close price and starting cash

        log_return = w_log[week, stock]
        return_on_investment = sign * np.sign(np.exp(log_return) - 1) * price * np.exp(log_return)
        starting_cash += return_on_investment
    progres2.append(return_on_investment - price)
    progres.append(starting_cash)
# Print the final cash balance
print(starting_cash)
plt.plot(np.array(progres))


