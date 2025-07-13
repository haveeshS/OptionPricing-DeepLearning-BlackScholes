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

import numpy as np
from scipy.stats import norm

def BlackScholesModel_vec(S, K, r, dT, sigma, isCall):
    """
    Vectorized version of the Black-Scholes formula for European options.

    Parameters
    ----------
    S : array-like
        Current stock price(s).
    K : array-like
        Strike price(s).
    r : array-like or float
        Risk-free rate(s).
    dT : array-like
        Time to maturity (in years).
    sigma : array-like
        Volatility of the underlying stock(s).
    isCall : array-like or bool
        True for call option(s), False for put option(s).

    Returns
    -------
    option_prices : ndarray
        Black-Scholes prices for the options.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    dT = np.asarray(dT)
    sigma = np.asarray(sigma)
    isCall = np.asarray(isCall)

    # Create an array of NaNs to start with
    option_prices = np.full_like(S, fill_value=np.nan, dtype=np.float64)

    # Valid inputs mask
    valid = (dT > 0) & (sigma >= 0) & (S > 0) & (K > 0)

    # sigma == 0 case (no volatility)
    zero_vol = valid & (sigma == 0)
    if zero_vol.any():
        if np.isscalar(isCall):
            if isCall:
                option_prices[zero_vol] = np.maximum(S[zero_vol] - K[zero_vol], 0)
            else:
                option_prices[zero_vol] = np.maximum(K[zero_vol] - S[zero_vol], 0)
        else:
            call_mask = isCall & zero_vol
            put_mask = ~isCall & zero_vol
            option_prices[call_mask] = np.maximum(S[call_mask] - K[call_mask], 0)
            option_prices[put_mask] = np.maximum(K[put_mask] - S[put_mask], 0)

    # Normal Black-Scholes case
    normal_case = valid & (sigma > 0)
    if normal_case.any():
        S_n = S[normal_case]
        K_n = K[normal_case]
        dT_n = dT[normal_case]
        sigma_n = sigma[normal_case]
        r_n = r if np.isscalar(r) else r[normal_case]

        d1 = (np.log(S_n / K_n) + (r_n + 0.5 * sigma_n**2) * dT_n) / (sigma_n * np.sqrt(dT_n))
        d2 = d1 - sigma_n * np.sqrt(dT_n)

        if np.isscalar(isCall):
            if isCall:
                option_prices[normal_case] = norm.cdf(d1) * S_n - norm.cdf(d2) * K_n * np.exp(-r_n * dT_n)
            else:
                option_prices[normal_case] = norm.cdf(-d2) * K_n * np.exp(-r_n * dT_n) - norm.cdf(-d1) * S_n
        else:
            call_mask = isCall[normal_case]
            put_mask = ~call_mask
            option_prices[normal_case][call_mask] = norm.cdf(d1[call_mask]) * S_n[call_mask] - norm.cdf(d2[call_mask]) * K_n[call_mask] * np.exp(-r_n[call_mask] * dT_n[call_mask])
            option_prices[normal_case][put_mask] = norm.cdf(-d2[put_mask]) * K_n[put_mask] * np.exp(-r_n[put_mask] * dT_n[put_mask]) - norm.cdf(-d1[put_mask]) * S_n[put_mask]

    return option_prices


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
