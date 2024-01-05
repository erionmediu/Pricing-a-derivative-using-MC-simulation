import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import math

filepath = "data-home-depo.csv"

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data.rename(columns={'RETX':'ret', 'PRC':'prc'}, inplace=True)
    data['log-ret'] = np.log(1 + data['ret'])
    data.drop(columns='PERMNO', inplace=True)
    return data

def plot_financial_data(data):
    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
    data['cumulative_return'] = (1 + data['ret']).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['date'], data['cumulative_return'], color='blue')
    ax.set_title('Growth of $1 Investment Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.show()

def calculate_statistics(data):
    mean = data['log-ret'].mean()
    std = np.std(data['log-ret'])
    anlogmean = ((mean + 1) ** 252) - 1
    anstdlog = std * math.sqrt(252)
    return mean, std, anlogmean, anstdlog

def monte_carlo_simulation(data, S0, annualrf, T, mc_sims, step):
    dailyrf = annualrf / 252
    drift_rf = dailyrf
    volatility = np.std(data['log-ret'])
    random_gbm = np.random.normal(0, np.sqrt(1), size=(mc_sims, step)).T
    St = np.exp((drift_rf - volatility ** 2 / 2) + volatility * random_gbm)
    St = np.vstack([np.ones(mc_sims), St])
    St = S0 * St.cumprod(axis=0)
    return St

def plot_simulation(St, S0, drift_rf, volatility, T, step):
    time = np.linspace(0, T, step + 1)
    tt = np.full(shape=(mc_sims, step + 1), fill_value=time).T
    plt.plot(tt, St)
    plt.xlabel("Days $(t)$")
    plt.ylabel("Stock Price $(S_t)$")
    plt.title("MC simulations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $S_0 = {}, \mu = {}, \sigma = {}$".format(S0, round(drift_rf,4), round(volatility,4)))
    plt.xlim([0, T])
    plt.xticks(range(T+1))
    plt.show()
    print('The Mean is ' + str(np.mean(St[-1])))

def option_pricing(St, strike_price, dailyrf, T):
    last_St = St[-1]
    dif_prices = [price - strike_price for price in last_St]
    Payoff_Calls = [max(elem, 0) for elem in dif_prices]
    Payoff_Puts = [max(strike_price - price, 0) for price in last_St]
    Call_p = np.mean(Payoff_Calls) * np.exp((-dailyrf) * T)
    Put_p = np.mean(Payoff_Puts) * np.exp((-dailyrf) * T)
    return Call_p, Put_p

def run_pipeline(filepath, S0, annualrf, T, mc_sims, step, strike_price):
    data = load_and_prepare_data(filepath)
    plot_financial_data(data)
    mean, std, anlogmean, anstdlog = calculate_statistics(data)
    print(f"Mean: {round(mean, 4)}, Std: {round(std, 4)}, Annualized Mean: {round(anlogmean, 4)}, Annualized Std: {round(anstdlog, 4)}")
    St = monte_carlo_simulation(data, S0, annualrf, T, mc_sims, step)
    plot_simulation(St, S0, annualrf / 252, std, T, step)
    Call_p, Put_p = option_pricing(St, strike_price, annualrf / 252, T)
    print(f"Call Option Price: {Call_p}, Put Option Price: {Put_p}")

# Parameters
filepath = "data-home-depo.csv"
S0 = 315.85999  # last observed price
annualrf = 0.03  # assumed 3 percent annual risk-free rate
T = 19           # 19 days timeframe
mc_sims = 10000  # number of simulations
step = 19        # number of steps
strike_price = 315.85999  # strike price for options

# Running the pipeline
run_pipeline(filepath, S0, annualrf, T, mc_sims, step, strike_price)