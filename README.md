# Financial Market Strategy Comparison Dashboard

## Overview
This project compares **Buy & Hold** and **Machine Learning (ML)**-based investment strategies using stock price data from Yahoo Finance (`yfinance`).  
It evaluates returns, plots performance, and helps users analyze which strategy performs better.

## Features
- Fetches historical stock data using `yfinance`
- Implements ML models like **Linear Regression** and **Random Forest** for price prediction
- Compares ML strategy returns with Buy & Hold strategy
- Visualizes returns, predictions, and performance metrics on a dashboard
- Allows user input for stock symbol, date range, and investment amount

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nilay-28/FinanceAnalyzer.git
2. **Navigate to the project directory:**
   ```bash
   cd FinanceAnalyzer
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py

## Usage

- Select stock symbol and date range  
- Enter investment amount  
- Choose ML model (Linear Regression or Random Forest)  
- View comparison of strategy returns and analysis  

## Notes

- The ML strategy predicts future prices based on historical data  
- Buy & Hold assumes buying at start date and holding till end date  
- Returns include percentage and absolute profit/loss based on investment amount

## Technologies & Credits

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

## Author

Nilay Koul
