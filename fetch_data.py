# script to obtain data of options from the freely available yahoo finance API on python, yfinance

import yfinance as yf
import pandas as pd
from datetime import datetime

tickers_list = ["AAPL","NVDA","TSLA","AMD","AMZN","MSFT","META","SPY"] # stock tickers or codes to be used in yahoo finance or yfinance
# in this project, I use options of the most popular stocks: Apple, Nvidia, Tesla, AMD, Amazon, Microsoft, Meta, SPY

calls_df = pd.DataFrame() # creating an empty dataframe to store call type options
puts_df = pd.DataFrame() # creating an empty dataframe to store put type options

for ticker in tickers_list:

    yfticker = yf.Ticker(ticker) # call a particular stock

    # in yfinance, the options are stored according to their expiry dates
    # options usually expire on fridays
    for date in yfticker.options:

        option_chain = yfticker.option_chain(date) # extract the option chain for the particular expiry date

        current_calls_df = pd.DataFrame(option_chain.calls) # calls
        current_puts_df = pd.DataFrame(option_chain.puts) # puts

        # adding an extra column to store the ticker of the stock for future reference
        current_calls_df["stockTicker"] = ticker 
        current_puts_df["stockTicker"] = ticker

        # adding an extra column to also store the expiry date (very important to compute the time to expiry)
        current_calls_df["expiryDate"] = date
        current_puts_df["expiryDate"] = date

        # merging all calls and puts into one dataframe
        calls_df = pd.concat([calls_df,current_calls_df], ignore_index=True) 
        puts_df = pd.concat([puts_df,current_puts_df], ignore_index=True)

# for an analysis with options, it is important to also know the stock price on the day of the option trade

# luckily, yfinance also stores the price history of stocks

# initializing the columns in the dataframes to store the closing price of the stock on a particular date
calls_df['stockClosePrice'] = None
puts_df['stockClosePrice'] = None

for ticker in tickers_list:

    ticker_data = yf.Ticker(ticker)
    price_history = ticker_data.history(period="max") # price history for the particular stock
    price_history.index = price_history.index.date

    price_lookup = price_history['Close'].to_dict() # converting it to dict for look up

    mask = calls_df['stockTicker'] == ticker
    dates = calls_df.loc[mask, 'lastTradeDate'].dt.date

    calls_df.loc[mask, 'stockClosePrice'] = dates.map(price_lookup)

    mask = puts_df['stockTicker'] == ticker
    dates = puts_df.loc[mask, 'lastTradeDate'].dt.date

    puts_df.loc[mask, 'stockClosePrice'] = dates.map(price_lookup)


today_str = datetime.now().strftime("%Y-%m-%d")

# Construct filenames dynamically
calls_filename = f"optiondata_calls_yfinance_{today_str}.csv"
puts_filename = f"optiondata_puts_yfinance_{today_str}.csv"

# Save the dataframes
calls_df.to_csv(calls_filename, index=False)
puts_df.to_csv(puts_filename, index=False)