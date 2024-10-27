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
from MonteCarlo import MonteCarlo
from StockAnalysis import StockAnalysis

class BlackScholes:
    def __init__(self, ticker_symbol, S, K, T, r, sigma):
        self.ticker_symbol = ticker_symbol
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def black_scholes_finite_difference(self, option_type, option_style="european", barrier_level=None, barrier_type=None):
        # Setting boundary and step size conditions
        S_class = MonteCarlo(StockAnalysis(self.ticker_symbol), self.T)
        S_max = S_class.run_simulation()
        S_steps = 100
        T_steps = 100

        # Creating grids
        S_grid = np.linspace(0, S_max, S_steps)
        T_grid = np.linspace(0, self.T, T_steps)

        V = np.zeros((S_steps, T_steps))

        if option_style == "asian":
            # Implementing FDM and getting average prices
            for i in range(S_steps):
                avg_price = np.mean(S_grid[:i+1])
                if option_type == "call":
                    V[i, -1] = max(avg_price - self.K, 0)
                elif option_type == "put":
                    V[i, -1] = max(self.K - avg_price, 0)
        else:
            V[:, -1] = np.maximum(S_grid - self.K, 0) if option_type == "call" else np.maximum(self.K - S_grid, 0)

        if option_style == "barrier" and barrier_level is not None:
            # Various barrier options
            if barrier_type == "up-and-out":
                V[:, -1] = np.where(S_grid > barrier_level, 0, V[:, -1])
            elif barrier_type == "down-and-out":
                V[:, -1] = np.where(S_grid < barrier_level, 0, V[:, -1])
            elif barrier_type == "up-and-in":
                V[:, -1] = np.where(S_grid < barrier_level, 0, V[:, -1])
            elif barrier_type == "down-and-in":
                V[:, -1] = np.where(S_grid > barrier_level, 0, V[:, -1])

        for t in range(T_steps - 2, -1, -1):
            V_next = np.zeros_like(V)

            for i in range(1, S_steps - 1):
                # Running finance difference method
                dS = S_grid[i + 1] - S_grid[i]
                dT = T_grid[t + 1] - T_grid[t]

                delta = (V[i + 1, t + 1] - V[i - 1, t + 1]) / (2 * dS)
                gamma = (V[i + 1, t + 1] - 2 * V[i, t + 1] + V[i - 1, t + 1]) / (dS ** 2)

                theta = -0.5 * self.sigma ** 2 * S_grid[i] ** 2 * gamma - self.r * S_grid[i] * delta + self.r * V[i, t + 1]
                V_next[i, t] = V[i, t + 1] + theta * dT

                if option_style == "american":
                    if option_type == "call":
                        exercise_value = max(S_grid[i] - self.K, 0)
                    elif option_type == "put":
                        exercise_value = max(self.K - S_grid[i], 0)
                    V_next[i, t] = max(V_next[i, t], exercise_value)

            V_next[0, t] = 2 * V_next[1, t] - V_next[2, t]
            V_next[-1, t] = 2 * V_next[-2, t] - V_next[-3, t]

            V[:, t] = V_next[:, t]

            if barrier_type == "up-and-out":
                V[:, t] = np.where(S_grid > barrier_level, 0, V[:, t])
            elif barrier_type == "down-and-out":
                V[:, t] = np.where(S_grid < barrier_level, 0, V[:, t])
            elif barrier_type == "up-and-in":
                V[:, t] = np.where(S_grid < barrier_level, 0, V[:, t])
            elif barrier_type == "down-and-in":
                V[:, t] = np.where(S_grid > barrier_level, 0, V[:, t])

        option_price = np.interp(self.S, S_grid, V[:, 0])
        return option_price
