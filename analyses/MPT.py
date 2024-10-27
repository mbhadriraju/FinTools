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
from StockAnalysis import StockAnalysis

class MPT:
    def __init__(self, stock_list):
        self.stock_list = stock_list
        self.stock_analysis_array = np.zeros((1008, len(stock_list)))

        # Fill stock_analysis_array with log returns
        for i, stock in enumerate(self.stock_list):
            self.stock_analysis_array[:, i] = StockAnalysis(stock).calculate()[2][-1008:]

        self.rf_rate = .0381  # Risk-free interest rate
        self.market_stock_analysis = StockAnalysis("SPY")
        self.beta_list = []
        self.calculate_betas()

    def calculate_betas(self):
        for stock in self.stock_list:
            stock_analysis = StockAnalysis(stock)
            beta = stock_analysis.beta(self.market_stock_analysis)
            self.beta_list.append(beta)

    def downside_deviation(self, portfolio_returns, threshold=0):
        """
        Calculates downside deviation based on a threshold (default is 0).
        Only negative deviations (returns below the threshold) are considered.
        """
        downside_risk = np.where(portfolio_returns < threshold, portfolio_returns - threshold, 0)
        downside_deviation = np.sqrt(np.mean(downside_risk ** 2))
        return downside_deviation

    def objective(self, weights):
        portfolio_returns = np.dot(self.stock_analysis_array, weights)
        risk_free_rate_daily = (1 + self.rf_rate) ** (1/252) - 1

        # Calculate downside deviation
        downside_dev = self.downside_deviation(portfolio_returns, threshold=risk_free_rate_daily)

        # Portfolio mean return
        means = np.mean(self.stock_analysis_array, axis=0)
        portfolio_return = np.dot(weights, means)

        # Sortino Ratio: (portfolio return - risk-free rate) / downside deviation
        sortino_ratio = (portfolio_return - risk_free_rate_daily) / downside_dev

        # Negative Sortino ratio for minimization
        return -sortino_ratio

    def constraint_return(self, target_return):
        target_return /= 100
        means = np.mean(self.stock_analysis_array, axis=0)
        portfolio_return = np.dot(self.weights, means)
        return portfolio_return - target_return

    def optimize_sortino(self, target_return):
        num_assets = len(self.stock_list)
        initial_weights = np.ones(num_assets) / num_assets

        # Constraints: 1) portfolio return must meet the target, 2) sum of weights = 1
        constraints = (
            {'type': 'eq', 'fun': self.constraint_return, 'args': [target_return]},
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        )

        # Bounds: weights should be between 0 and 1 for long-only portfolios
        bounds = tuple((0, 1) for asset in range(num_assets))

        result = minimize(self.objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        # Normalize weights in case there's any small deviation from 1 due to optimization
        new_result = np.array(result.x)
        for i in range(len(result.x)):
            new_result[i] = result.x[i] / np.sum(result.x)

        return new_result, (result.fun * -1)  # Maximize the Sortino ratio by returning the negative of the minimized value

