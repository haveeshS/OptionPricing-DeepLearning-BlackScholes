import numpy as np
import scipy.stats

def BlackScholesModel(S,K,r,dT,sigma,isCall):
    """
    Computes the theoretical price of a European option using the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current stock price (spot price).
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    dT : float
        Time to maturity in years (e.g., 0.5 for 6 months).
    sigma : float
        Volatility of the underlying stock (annualized standard deviation).
    isCall : bool
        If True, calculates the price of a call option; if False, calculates the price of a put option.

    Returns
    -------
    float
        The Black-Scholes price of the European call or put option.
        Returns NaN if any of the inputs are invalid (e.g., non-positive S, K, dT, or sigma).

    """

    if dT <= 0 or sigma < 0 or S <= 0 or K <= 0:
        return np.nan  
    
    if sigma == 0:
        if isCall:
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d_plus = 1/(sigma*np.sqrt(dT))* ( np.log(S/K) + (r + sigma**2/2)*dT )
    d_minus = d_plus - sigma*np.sqrt(dT)

    if isCall:
        option_price = scipy.stats.norm.cdf(d_plus)*S - scipy.stats.norm.cdf(d_minus)*K*np.exp(-r*dT)
    else:
        option_price = scipy.stats.norm.cdf(-d_minus)*K*np.exp(-r*dT) - scipy.stats.norm.cdf(-d_plus)*S
    
    return option_price


BlackScholesModel_vec = np.vectorize(BlackScholesModel)




def binomial_american_option(S, K, r, dT, sigma, isCall, div_yield=0.0, steps=100):
    """
    Prices an American option using the Cox-Ross-Rubinstein (CRR) binomial tree model.

    Parameters
    ----------
    S : float
        Current stock price (spot price).
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate (annualized, continuously compounded).
    dT : float
        Time to maturity in years.
    sigma : float
        Volatility of the underlying stock (annualized).
    isCall : bool
        True for call option, False for put option.
    div_yield : float, optional
        Continuous dividend yield (default 0.0).
    steps : int, optional
        Number of time steps in the binomial tree (default 100).

    Returns
    -------
    float
        The American option price.
    """

    if sigma <= 0 or dT <= 0 or steps <= 0:
        return np.nan

    dt = dT / steps
    discount = np.exp(-r * dt)

    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    p = (np.exp((r - div_yield) * dt) - d) / (u - d)
    p = np.clip(p, 0, 1)

    asset_prices = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])

    if isCall:
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    if np.any(np.isnan(option_values)) or np.any(np.isinf(option_values)):
        raise ValueError("NaN or Inf found in initial option values")

    for i in range(steps - 1, -1, -1):
        asset_prices = asset_prices[:i+1] * d
        option_values = discount * (p * option_values[1:i+2] + (1 - p) * option_values[:i+1])
        if isCall:
            option_values = np.maximum(option_values, asset_prices - K)
        else:
            option_values = np.maximum(option_values, K - asset_prices)

    return option_values[0]
