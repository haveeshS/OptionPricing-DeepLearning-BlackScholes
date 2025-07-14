# For an analysis of options, it is also important to have the value of the annualized risk-free interest rate
# This script fetches this data by scraping the website of the US Department of Treasury
# This will suffice as in this project only options of popular US stocks have been used

import pandas as pd
from bs4 import BeautifulSoup
import requests

# importing the data which was fetched using "fetch_data.py"
calls_df = pd.read_csv("data/optiondata_calls_yfinance_2025-07-11.csv") 
puts_df = pd.read_csv("data/optiondata_puts_yfinance_2025-07-11.csv")

# extracting the years in which data is available
# the risk-free rates are obtained for these years
years = pd.unique(
    pd.concat([
        pd.to_datetime(puts_df["lastTradeDate"]).dt.year,
        pd.to_datetime(calls_df["lastTradeDate"]).dt.year
    ])
)

# empty dataframe for later-on
riskfreerate_df = pd.DataFrame()

for i,year in enumerate(years):

    # the url of the website which contains the table of risk-free rates
    url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
    
    # the table is embedded in the website, so we can use BeautifulSoup to scrape the required data from the html source file

    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html')

    table_riskfreerates = soup.find_all("table")[0]

    if i==0:

        column_titles = table_riskfreerates.find_all('th') # gets the table headers, which we can use for the headers of the dataframe

        column_table_titles = [title.text for title in column_titles]

        riskfreerate_df = pd.DataFrame(columns = column_table_titles)

    all_rows = table_riskfreerates.find_all('tr')

    for row in all_rows[1:]: # 1st row is the column header names

        row_data = row.find_all("td")

        individual_row_data = [data.text.strip() for data in row_data] #separating the data out and storing it in a list

        length = len(riskfreerate_df)

        riskfreerate_df.loc[length] = individual_row_data # appending it to the dataframe

# a problem with using US data is that the dates are in the US format, so this needs to be adjusted to the standard date format of YYYY-MM-DD
riskfreerate_df['standard_date'] = pd.to_datetime(riskfreerate_df['Date'], format='%m/%d/%Y')
riskfreerate_df = riskfreerate_df.set_index("standard_date")

# the datapoints for the risk-free rates are not continous, so it is important to fill them gaps
# I create a new dataframe which has these dates, and then fill them using forward fill
all_days = pd.date_range(start=riskfreerate_df.index.min(), end=riskfreerate_df.index.max(), freq='D')
riskfreerate_df_full = riskfreerate_df.reindex(all_days)
riskfreerate_df_full['riskfree_rate'] = riskfreerate_df_full['1 Mo'].fillna(method='ffill') # using the 1 Month values

# separating out the column in a new dataframe, which has date as its index
rate_df = riskfreerate_df_full[["riskfree_rate"]]

# making sure the tradeDate columns are pandas datetime objects
calls_df["tradeDate"] = pd.to_datetime(pd.to_datetime(calls_df["lastTradeDate"]).dt.date)
puts_df["tradeDate"] = pd.to_datetime(pd.to_datetime(puts_df["lastTradeDate"]).dt.date)

rate_df.index = pd.to_datetime(rate_df.index).normalize()

# merging the risk-free rate values on the days the option was traded
calls_df = calls_df.merge(rate_df, left_on="tradeDate", right_index=True,how="left")
puts_df = puts_df.merge(rate_df, left_on="tradeDate", right_index=True,how="left")

# adding these binary columns of "isCall" so that we can merge the calls and puts dataframes into one
calls_df["isCall"] = 1
puts_df["isCall"] = 0

# Combine both DataFrames
options_df = pd.concat([calls_df, puts_df], ignore_index=True)


# for the Black-Scholes model, one of the parameters is time to expiry in years
# for this purpose it is useful to already compute the time between the trade date and expiry date of the option
# the expiry time on the given expiry date is assumed to be the time when the US markets close, i.e, 16:00 Eastern Time US 
# actually,  16:15 Eastern time is used, to account for last minute trades which happen after the deadline 
# and to avoid negative time_to_expiry values 
expiry_dt = (pd.to_datetime(options_df['expiryDate']) + pd.Timedelta(hours=16, minutes=15)).dt.tz_localize("US/Eastern") 
trade_dt = pd.to_datetime(options_df['lastTradeDate']).dt.tz_convert("US/Eastern")

time_to_expiry = (expiry_dt - trade_dt).dt.total_seconds()/(365*24*3600) # in years

options_df["deltaT_years"] = time_to_expiry # this column stores time_to_expiry

options_df.to_csv("data/options_withriskfreerates_andDeltaT.csv",index=False) # saving the whole data


# it is also convenient to already extract the more useful columns

relevant_columns = ['strike', # strike price of option
                    'stockClosePrice', # stock price at time of trade
                    'deltaT_years', # time to expiry
                    'riskfree_rate', # rise free rate on the day of trade
                    'isCall', # 1 for Call option, 0 for Put option
                    'volume', # number of trades of the option on the day
                    'openInterest', # number of option contracts still open
                    'impliedVolatility', # volatility implied from the data
                    'inTheMoney', # boolean column if the option is in the money or no 
                    'isNVDA', # adding an extra column for whether the stock is Nvidia
                    'lastPrice' # the price of the option at last trade, our *target* variable
                    ]

options_relevantdata_df = options_df.loc[:,relevant_columns]

options_relevantdata_df["volume"] = options_relevantdata_df["volume"].fillna(0)
options_relevantdata_df["openInterest"] = options_relevantdata_df["openInterest"].fillna(0)

options_relevantdata_df.to_csv("data/options_cleaned.csv",index=False)