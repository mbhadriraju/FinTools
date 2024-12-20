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

class MonteCarlo:
    def __init__(self, stock_analysis, years):
        self.stock_analysis = stock_analysis
        self.years = years
        self.reps = 10000
        self.future_prices = np.empty(self.reps)
        self.future_returns = np.empty(self.reps)
        
    def run_simulation(self):
        # Create a stock_analysis instance and calculate current price
        self.stock_analysis.calculate()
        current_price = self.stock_analysis.data_close.iloc[-1]

        # Calculate log returns
        log_returns_scaled = self.stock_analysis.log_returns_arr[-252:]

        # Get the mean return and volatility
        mean_return = arma_result.params[0]
        mean_volatility = np.mean(garch_result.conditional_volatility)

        # Get the simulated returns from the model and calculate expected value and Sharpe
        simulated_returns = np.random.normal(mean_return, mean_volatility, size=(self.reps, int(252 * self.years)))
        total_returns = simulated_returns.sum(axis=1)

        self.future_prices = current_price * np.exp(total_returns)

        expected_value = np.mean(self.future_prices)
        volatility = np.std(self.future_prices)
        sharpe_ratio = (expected_value - current_price) / volatility

        print("Expected Value:", expected_value)
        print("Volatility (1 standard deviation):", volatility)
        print("Sharpe Ratio:", sharpe_ratio)

        plt.hist(self.future_prices, bins=50, range=(np.min(self.future_prices), np.percentile(self.future_prices, 97)))
        plt.xlabel("Future Prices")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Future Projected Prices {self.years} Years into the Future")
        plt.show()
        
        return expected_value + 3 * volatility
