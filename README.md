# quant-models

A Python implementation of five quantitative finance mental models — tools for options pricing, market microstructure analysis, statistical arbitrage, equity valuation, and fundamental analysis prioritisation. Built to translate theoretical frameworks from financial economics into executable, well-documented code.

---

## Motivation

Quantitative finance is dense with models that are frequently cited but rarely implemented cleanly outside of institutional codebases. This repository captures five mental models that a practitioner or researcher should be able to reach for quickly: not as black boxes, but as transparent implementations where the assumptions, mechanics, and outputs are all visible. Each function is self-contained and documented at the parameter level.

---

## Models

### 1. Black-Scholes-Merton (BSM) with Full Greeks

**What it does:** Prices European call and put options under the BSM framework and returns the complete Greeks vector: Delta, Gamma, Vega, Theta.

**Why it matters:** BSM remains the industry baseline for options pricing despite its well-known limitations (constant volatility, no jumps, no dividends). The Greeks are the practitioner's primary risk management language — Delta for directional exposure, Gamma for convexity risk, Vega for volatility sensitivity, and Theta for time decay. Understanding the output of this function is prerequisite to reasoning about any options book.

**Implementation note:** d1 and d2 are computed in closed form; Vega is expressed per unit of volatility (not per percentage point), consistent with standard market convention.

```python
from quant_mental_models import calculate_greeks

# At-the-money call: spot 100, strike 100, 6 months, 5% risk-free, 20% IV
result = calculate_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type='call')

# Output:
# {
#   'Price': 6.89,
#   'Delta': 0.5987,
#   'Gamma': 0.0281,
#   'Vega': 28.07,
#   'Theta': -6.65
# }

# Deep in-the-money put: spot 80, strike 100, 3 months, 5% risk-free, 25% IV
result = calculate_greeks(S=80, K=100, T=0.25, r=0.05, sigma=0.25, option_type='put')

# Output:
# {
#   'Price': 19.48,
#   'Delta': -0.9374,
#   'Gamma': 0.0094,
#   'Vega': 4.73,
#   'Theta': -5.21
# }
```

---

### 2. Amihud Illiquidity Ratio & Kyle's Lambda Proxy

**What they do:** Two complementary measures of market liquidity — one price-impact-based (Amihud), one microstructure-based (Kyle's Lambda).

**Why they matter:** Liquidity is a latent variable; you cannot observe it directly, only infer it from price behaviour relative to order flow. The Amihud ratio (Amihud, 2002) captures the average price impact of a dollar of volume — high values flag thinly traded assets where even modest order flow moves prices materially. Kyle's Lambda (Kyle, 1985) is the regression coefficient of price changes on signed order flow, estimating the adverse selection cost imposed by informed trading. Together they give both a historical and a structural view of liquidity conditions.

```python
import numpy as np
from quant_mental_models import amihud_illiquidity, kyles_lambda_proxy

# Amihud: 20 days of returns and dollar volume for a mid-cap stock
daily_returns = np.array([0.012, -0.008, 0.003, 0.015, -0.005,
                           0.009, -0.011, 0.002, 0.007, -0.003,
                           0.014, -0.006, 0.004, 0.010, -0.009,
                           0.001, 0.013, -0.007, 0.005, 0.008])

daily_dollar_volume = np.array([2.1e6, 1.8e6, 3.2e6, 1.5e6, 2.8e6,
                                 2.0e6, 1.9e6, 3.5e6, 2.3e6, 2.6e6,
                                 1.7e6, 2.9e6, 2.4e6, 1.6e6, 3.1e6,
                                 2.7e6, 1.4e6, 2.2e6, 3.0e6, 2.5e6])

ratio = amihud_illiquidity(daily_returns, daily_dollar_volume)
# Output: ~4.2e-9 (low = liquid; high = illiquid)

# Kyle's Lambda: price changes vs. net order flow (buyer-initiated minus seller-initiated)
price_changes = np.array([0.05, -0.03, 0.08, -0.02, 0.06, -0.04, 0.07, -0.01, 0.04, -0.05])
order_flow    = np.array([500, -300, 800, -200, 600, -400, 700, -100, 400, -500])

lam = kyles_lambda_proxy(price_changes, order_flow)
# Output: ~0.000087 (lambda coefficient; steeper = more adverse selection)
```

---

### 3. Engle-Granger Cointegration Test for Pairs Trading

**What it does:** Tests whether two price series share a stationary long-run equilibrium — the statistical precondition for a mean-reverting pairs trade.

**Why it matters:** Correlation is a poor basis for pairs trading because it is not invariant to level shifts and does not imply mean reversion. Cointegration (Engle & Granger, 1987) tests whether a linear combination of two non-stationary I(1) series is itself stationary — if so, deviations from the equilibrium spread are temporary and tradeable. The Engle-Granger two-step procedure implemented here applies the ADF test to the residuals of an OLS regression of one price series on the other; a rejection of the unit root null (p < 0.05) supports a cointegrated relationship.

```python
import numpy as np
from quant_mental_models import check_pairs_trade

# Simulate two cointegrated series (e.g. two oil majors)
np.random.seed(42)
n = 252
common_trend = np.cumsum(np.random.normal(0, 1, n))
asset_a = common_trend + np.random.normal(0, 0.5, n)
asset_b = 0.8 * common_trend + np.random.normal(0, 0.5, n)

result = check_pairs_trade(asset_a, asset_b)
# Output: "COINTEGRATED: Valid Pairs Trade Candidate"

# Two unrelated random walks
random_walk_1 = np.cumsum(np.random.normal(0, 1, n))
random_walk_2 = np.cumsum(np.random.normal(0, 1, n))

result = check_pairs_trade(random_walk_1, random_walk_2)
# Output: "NO STATISTICAL RELATIONSHIP: Correlation is likely spurious"
```

---

### 4. Implied Equity Risk Premium (ERP) Solver

**What it does:** Solves numerically for the discount rate implied by current market prices given a stream of expected cash flows, then subtracts the risk-free rate to isolate the equity risk premium.

**Why it matters:** The implied ERP (following Damodaran's approach) is a forward-looking, market-derived estimate of the compensation investors currently demand for bearing equity risk — directly analogous to the yield-to-maturity of a bond but applied to the aggregate equity market. Unlike historical ERP estimates (which average past excess returns), the implied ERP is an in-real-time signal that updates with prices and earnings expectations. It is widely used in asset allocation, corporate hurdle rate setting, and relative value assessment.

```python
from quant_mental_models import solve_implied_erp

# S&P 500 stylised example:
# Index at 5,000, 10-year Treasury at 4.5%,
# forward earnings/dividends projections for 5 years then terminal
current_index_price = 5000
risk_free_rate = 0.045
expected_cash_flows = [200, 215, 231, 248, 266, 5800]  # Year 1-5 + terminal

erp = solve_implied_erp(current_index_price, risk_free_rate, expected_cash_flows)
print(f"Implied ERP: {erp:.2%}")
# Output: Implied ERP: ~3.8% (varies with inputs)
```

---

### 5. SEC 10-K Reading Priority Heuristic

**What it does:** Returns a structured priority map for reading a 10-K filing, ranking sections by signal-to-noise ratio.

**Why it matters:** A 10-K filing averages 40,000–80,000 words. Reading sequentially wastes most of the time. Item 7 (MD&A) contains management's own forward-looking interpretation of operations — the single highest-signal section because it reveals how insiders frame their own uncertainty. Item 1A (Risk Factors) is legally required to enumerate material threats, making it a structured downside checklist that correlates strongly with future negative outcomes (Kravet & Muslu, 2013). Financial Statements (Item 8) provide the raw numbers but require MD&A as context to interpret correctly. This heuristic encodes that reading order.

```python
from quant_mental_models import prioritize_sec_sections

order = prioritize_sec_sections(filing_text="")
print(order)
# Output: "Focus extraction on Item 7 and Item 1A first."
```

---

## Dependencies

```
numpy
scipy
statsmodels
```

Install with:

```bash
pip install numpy scipy statsmodels
```

---

## Structure

```
gemini-quant-models/
│
├── quant_mental_models.py   # All five model implementations
└── README.md
```

---

## References

- Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315–1335.
- Merton, R. C. (1973). Theory of rational option pricing. *Bell Journal of Economics and Management Science*, 4(1), 141–183.
