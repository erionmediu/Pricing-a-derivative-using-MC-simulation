# Pricing Derivatives using Montecarlo Simulation

This project is dedicated to analyzing historical stock price data and forecasting future prices using Monte Carlo simulations. The main focus is on the stock of Home Depot (HD), but the methods used can be applied to any stock.
This analysis includes visualizing investment growth over time, calculating log returns and their volatility, and predicting future stock prices and option valuations.

Steps involved

    Data Loading and Cleaning: Load historical stock price data, rename columns for clarity, and calculate log returns.
    Investment Growth Visualization: Plot the growth of $1 invested in the stock over time, showing the cumulative return.
    Statistical Analysis: Calculate the average daily and annual log return and the volatility of the stock.
    Monte Carlo Simulation: Forecast future stock prices using the Geometric Brownian Motion model, a common approach in finance for modeling stock prices.
    Options Pricing: Calculate the payoffs and prices for call and put options based on the simulated future stock prices.

Dependencies

    Python 3.x
    Libraries: pandas, numpy, datetime, scipy, matplotlib

File Descriptions

    stock-data.csv: The dataset containing historical stock prices for Home Depot.
    stock_analysis.py: The main Python script with all the analysis and simulations.

How to Run

    Ensure all dependencies are installed.
    Place the stock-data.csv file in the same directory as the script.
    Run the script using Python: python stock_analysis.py.

