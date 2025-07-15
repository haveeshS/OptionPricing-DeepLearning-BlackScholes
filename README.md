# Option Pricing using Black-Scholes and Neural Networks

Welcome to this repository of the project where I build an option pricing model using neural networks and also compare it to the well known Black Scholes model. It combines data-driven modelling with financial theory to assess the performance of machine learning models in replicating or outperforming analytical pricing techniques.

## Overview

- Collects historical **options data** (calls and puts) for 8 major U.S. stocks from Yahoo Finance using the `yfinance` API.
- Implements **Black-Scholes pricing** for European-style options.
- Builds and trains two **feed-forward neural networks (Multi-Layer Perceptrons)** using TensorFlow/Keras for option price prediction.
- Compares model performance using **Mean Absolute Error (MAE)** and plots predicted vs. market prices.
- Includes preprocessing, feature engineering, such as feature scaling and normalization, and web scraping to obtain and integrate risk-free rates from U.S. Treasury yield data.

## Results

**Note**: A more detailed explanation of the analysis can be found in the project report: `short_summary.pdf`

In this project, I used options data from 8 major stocks in the US stock market, namely, Apple, Nvidia, Tesla, AMD, Amazon, Microsoft, Meta and S&P 500. 

Firstly, I computed the option prices from the analytical Black-Scholes model and compared them to the actual or market prices of the particular option. These results are visualised in the below plot:

![Plot of the calculated price of the option from Black Scholes against the price of the option from its last trade (market price) for each stock](https://github.com/haveeshS/OptionPricing-DeepLearning-BlackScholes/blob/main/plots/BS_Market_Comparison_CallsPuts_InOutTheMoney.png)

In this plot we can see that the predictions or calculated price from Black Scholes is okayish, with many outliers still being present. An interesting observation is that the predictions for Nvidia (NVDA) options are significantly deviating for the Black Scholes model. This could be attributed to the very volatilte nature of the Nvidia stock at present (July 2025), which may have triggered an atypically high volume of speculative trading activity. The predictions for the other stocks are okayish, but could be better. 

To improve the predictions for the option pricing, I use two neural network models (Model 1 and Model 2), which significantly improve predictive powers of option pricing. The performance of these two models alongside the Black Scholes model is visualised in the below plot:

![Comparison of predicted option prices vs. actual market prices for the Black-Scholes model and two neural network models](https://github.com/haveeshS/OptionPricing-DeepLearning-BlackScholes/blob/main/plots/optionprice_BSvsNNmodels.png)


The calculated or predicted option prices are much better compared to the Black Scholes model. They even account for the anomalous behaviour of Nvidia options as most predictions lie close to the perfect fit line. Model 1 achieves the lowest MAE and shows a tighter fit across different stock behaviours, including high-volatility NVDA options.

This project shows the power of data-driven model building in finance which can out perform classical analytical models like Black Scholes which may not account for all anomalous behaviour. 

## Usage Guidelines

To run this project, follow the steps below:

1. **Fetch Option Data**  
   Run `fetch_data.py` to download options data for the eight selected stocks from Yahoo Finance (via the `yfinance` library). The raw data will be stored in the `data/` directory.

2. **Preprocess Data & Fetch Risk-Free Rates**  
   Execute `preprocess_data.py` to clean and structure the data. This script also scrapes U.S. Treasury yield data to obtain risk-free rates, which are integrated into the dataset.

3. **Train Neural Network Models**  
   Run `OptionPricing_NeuralNet_Trainer.py` to train two neural networks. The trained models will be saved as:
   - `model_scaledfeatures.keras` – Model 1  
   - `best_model_from_tuner.keras` – Model 2

4. **Compare Models**  
   Use the notebook `optionpricing_model_compare.ipynb` to load and evaluate the neural network models against the Black-Scholes pricing model (implemented in `models.py`). This notebook includes visualisations and performance comparisons.

