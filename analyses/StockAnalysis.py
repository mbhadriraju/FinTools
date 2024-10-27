import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq, minimize, Bounds
import numpy as np
from scipy.optimize import minimize

class StockAnalysis:
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol
        self.data_close = yf.download(ticker_symbol)["Close"]
        self.log_returns_arr = None
        self.weighted_avg = None
        self.weighted_stddev = None

    def calculate(self):
        # Get returns
        returns = self.data_close.pct_change().dropna()
        log_returns = np.log1p(returns)
        self.log_returns_arr = log_returns.to_numpy()

        # Compute weighted average
        decay_rate = np.log(2) / (self.get_years_public() * 252)
        weights = np.exp(-decay_rate * .01 * np.arange(len(self.log_returns_arr)))
        self.weighted_avg = np.average(self.log_returns_arr, weights=weights)

        # Compute weighted standard deviation
        weighted_variance = np.average((self.log_returns_arr - self.weighted_avg) ** 2, weights=weights)
        self.weighted_stddev = np.sqrt(weighted_variance)

        return self.weighted_avg, self.weighted_stddev, self.log_returns_arr
    
    def get_risk_free_rate(self):
        # Use Yahoo Finance to get risk-free rate
        risk_free_rate = yf.Ticker("^TNX").history(period="1y", interval="1d")["Close"].iloc[-1] / 100
        return risk_free_rate

    def get_years_public(self):
        return (self.data_close.index[-1] - self.data_close.index[0]).days / 365.25
    
    def beta(self, market_stock_analysis):
        # Filter data to the past year
        end_date = self.data_close.index[-1]
        start_date = end_date - pd.DateOffset(years=1)
        
        # Filter stock data
        stock_data_filtered = self.data_close[(self.data_close.index >= start_date) & (self.data_close.index <= end_date)]
        returns_filtered = stock_data_filtered.pct_change().dropna().to_numpy()

        # Filter market data
        market_data_filtered = market_stock_analysis.data_close[(market_stock_analysis.data_close.index >= start_date) & (market_stock_analysis.data_close.index <= end_date)]
        market_returns_filtered = market_data_filtered.pct_change().dropna().to_numpy()

        # Ensure lengths match
        min_length = min(len(returns_filtered), len(market_returns_filtered))
        returns_filtered = returns_filtered[-min_length:]
        market_returns_filtered = market_returns_filtered[-min_length:]

        # Calculate covariance and variance
        covariance = np.cov(returns_filtered, market_returns_filtered)[0, 1]
        variance_market = np.var(market_returns_filtered)

        beta = covariance / variance_market
        return beta