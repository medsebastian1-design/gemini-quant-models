# MENTAL MODEL: Black-Scholes-Merton (BSM) & Greeks
import numpy as np
from scipy.stats import norm

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    S: Spot Price, K: Strike, T: Time to Expiry (years), 
    r: Risk-free rate, sigma: Implied Volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) # Sensitivity to Volatility
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    
    return {'Price': price, 'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta}

# MENTAL MODEL: Amihud Illiquidity & Kyle's Lambda Proxy
def amihud_illiquidity(daily_returns, daily_dollar_volume):
    """
    High value = Illiquid (Price moves easily on low volume)
    Low value = Liquid (Deep order book)
    """
    # |Return| / (Price * Volume)
    illiquidity_series = np.abs(daily_returns) / daily_dollar_volume
    return np.mean(illiquidity_series)

def kyles_lambda_proxy(price_changes, order_flow):
    """
    Regression slope of Price Change vs. Order Flow.
    Steep slope = High adverse selection risk (Informed traders are present).
    """
    import statsmodels.api as sm
    model = sm.OLS(price_changes, order_flow).fit()
    return model.params[0] # The 'Lambda' coefficient

# MENTAL MODEL: Cointegration Test (Engle-Granger)
from statsmodels.tsa.stattools import coint

def check_pairs_trade(asset_a_prices, asset_b_prices):
    """
    Checks if two assets are cointegrated (mean-reverting spread).
    p-value < 0.05 implies a strong statistical tether.
    """
    score, pvalue, _ = coint(asset_a_prices, asset_b_prices)
    
    if pvalue < 0.05:
        return "COINTEGRATED: Valid Pairs Trade Candidate"
    else:
        return "NO STATISTICAL RELATIONSHIP: Correlation is likely spurious"

  # MENTAL MODEL: Implied Equity Risk Premium (ERP) Solver
from scipy.optimize import newton

def solve_implied_erp(current_index_price, risk_free_rate, expected_cash_flows):
    """
    Solves for 'r' (Market Internal Rate of Return) where:
    Price = Sum(CashFlow / (1+r)^t)
    Implied ERP = r - Risk_Free_Rate
    """
    def objective_function(r):
        pv = sum([cf / ((1 + r) ** (t + 1)) for t, cf in enumerate(expected_cash_flows)])
        return current_index_price - pv

    implied_market_return = newton(objective_function, x0=0.08) # Start guess at 8%
    return implied_market_return - risk_free_rate

# MENTAL MODEL: SEC 10-K Structure Parsing
def prioritize_sec_sections(filing_text):
    """
    Prioritizes reading order for maximum signal-to-noise ratio.
    """
    priority_map = {
        "Item 7": "MD&A - Management's view of future operations (CRITICAL)",
        "Item 1A": "Risk Factors - What can kill the business (CRITICAL)",
        "Item 8": "Financial Statements - The raw truth",
        "Item 1": "Business Description - General context (Lower Priority)"
    }
    return "Focus extraction on Item 7 and Item 1A first."

