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
from analyses import BlackScholes, MonteCarlo, MPT, StockAnalysis



def main():
    print("Choose the operation:")
    print("1. Monte Carlo Simulation")
    print("2. Option Pricing")
    print("3. CAPM Optimization")
    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == "1":
        ticker_symbol = input("Enter the ticker symbol: ")
        years = float(input("Enter the number of years for the simulation: "))
        stock_analysis = StockAnalysis(ticker_symbol)
        mc_simulation = MonteCarlo(stock_analysis, years)
        mc_simulation.run_simulation()

    elif choice == "2":
        ticker_symbol = input("Enter the ticker symbol: ")
        strike_price = float(input("Enter the strike price: "))
        maturity_years = float(input("Enter the maturity time in years: "))
        option_type = input("Enter the option type: Call, Put: ").lower()
        option_style = input("Enter the option style: European, American, Asian, Barrier: ").lower()

        stock_analysis = StockAnalysis(ticker_symbol)
        stock_analysis.calculate()

        current_price = stock_analysis.data_close.iloc[-1]
        r = stock_analysis.get_risk_free_rate()
        sigma = stock_analysis.weighted_stddev

        black_scholes = BlackScholes(ticker_symbol, current_price, strike_price, maturity_years, r, sigma)

        if option_style == "european":
            option_price = black_scholes.black_scholes_finite_difference(option_type, option_style=option_style)
        elif option_style == "american":
            option_price = black_scholes.black_scholes_finite_difference(option_type, option_style=option_style)
        elif option_style == "asian":
            option_price = black_scholes.black_scholes_finite_difference(option_type, option_style=option_style)
        elif option_style == "barrier":
            barrier_level = float(input("Enter the barrier level: "))
            barrier_type = input("Enter the barrier type: Up-and-out, Down-and-out, Up-and-in, Down-and-in: ").lower()
            option_price = black_scholes.black_scholes_finite_difference(option_type, option_style=option_style, barrier_level=barrier_level, barrier_type=barrier_type)

        print(f"The {option_style} {option_type} option price is: {option_price}")

    elif choice == "3":
        num_stocks = int(input("Enter the number of stocks: "))
        stock_list = []
        for _ in range(num_stocks):
            ticker_symbol = input("Enter the ticker symbol: ")
            stock_list.append(ticker_symbol)

        # Optimize the portfolio weights
        capm = MPT(stock_list)
        capm.constraint_return(target_return=float(input("Enter your return as a percent: ")))
        optimal_weights, optimal_sortino_ratio = capm.optimize_sortino()

        # Print the optimal portfolio weights
        print("Optimal Portfolio Weights:")
        for stock, weight in zip(stock_list, optimal_weights):
            print(f"{stock}: {weight}")

        # Print the optimal portfolio Sharpe Ratio
        print("Optimal Portfolio Sortino Ratio:", optimal_sortino_ratio)



if __name__ == "__main__":
    main()
