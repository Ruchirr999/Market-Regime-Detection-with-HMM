# Market Regime Detection using Hidden Markov Models

Unsupervised detection of hidden market regimes (bull, bear, choppy, crash) from daily price data using Gaussian HMMs, applied to NSE-listed Indian equities. Includes a walk-forward causal backtest, position sizing, noise filtering, and cross-ticker evaluation — built as an ML portfolio project.

> **Honest framing:** this is not a trading system. It is a demonstration that statistically distinct market regimes exist and carry measurable signal. Whether that signal translates to return outperformance depends heavily on the stock's behavior.

---

## Project Structure

```
.
├── regime_hmm.py           # Core pipeline — start here
├── evaluate_regimes.py     # Cross-ticker evaluation
├── regime_sizing.py        # Continuous position sizing via regime probabilities
├── regime_filters.py       # Persistence filter + severity score
├── README.md
└── outputs/
    └── evaluation/
        ├── summary.csv
        └── plots/
            ├── summary_comparison.png
            ├── HDFCBANK_evaluation.png
            ├── INFY_evaluation.png
            ├── RELIANCE_evaluation.png
            ├── SBIN_evaluation.png
            ├── TATASTEEL_evaluation.png
            └── TCS_evaluation.png
```

---

## Files — What Each Does and When to Use It

### `regime_hmm.py` — the core
The brain of the project. Run this first on any ticker.

**What it does:**
- Downloads price data via yfinance (2015–present)
- Builds 6 features from daily closes: return, slow volatility (21d), vol ratio (fast/slow — crash spike detector), drawdown (distance from 60d high), momentum (price vs 20d MA), vol-of-vol
- Selects number of HMM states (2–4) using BIC — justified automatically, not guessed
- Runs a **walk-forward causal fit**: refits HMM every 21 days on expanding window only (no lookahead). Also refits immediately when vol ratio > 2x (volatility spike = crash onset signal)
- Labels every day with a regime (0 = worst, k-1 = best)
- Backtests: avoid regime 0, hold cash; compare vs buy-and-hold
- Outputs: regime plots, monthly breakdown CSV, report.txt, latest week/month zoom plots

**When to use:** always — every other file depends on this.

```bash
python regime_hmm.py
# prompts: Enter stock name (or 'NSEI' for Nifty index)
```

**Outputs saved to** `outputs/<TICKER>/`

---

### `evaluate_regimes.py` — cross-ticker evaluation
Run this after you have confidence in the core pipeline. Answers: "does this model work in general, or did it just get lucky on one stock?"

**What it does:**
- Runs the full walk-forward pipeline on a list of tickers
- Computes 7 metrics for both buy-and-hold and regime strategy per ticker: Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Monthly Win Rate
- Prints a summary table with sharpe edge, drawdown improvement, time in market
- Saves `summary.csv` and a comparison bar chart

**When to use:** when evaluating the model across multiple tickers, or before writing up results.

```bash
# Interactive
python evaluate_regimes.py

# Non-interactive
python evaluate_regimes.py --tickers RELIANCE HDFCBANK TATASTEEL TCS INFY SBIN

# Skip plots for speed
python evaluate_regimes.py --tickers RELIANCE HDFCBANK --no-plots

# Custom output dir
python evaluate_regimes.py --tickers TATASTEEL --output outputs/steel_test
```

---

### `regime_sizing.py` — continuous position sizing
Upgrades the binary in/out signal to a continuous dial.

**What it does:**
- `walk_forward_proba()`: like the core walk-forward but returns full posterior probability vectors instead of hard labels — e.g. [0.7, 0.2, 0.1] instead of just "0"
- `compute_position_size()`: converts probabilities to position size via dot product with regime weights. If 70% probability of worst regime → position ≈ 0.3 instead of 0
- `backtest_sized_strategy()`: compares Buy & Hold vs Binary vs Sized with full metrics table
- Three weight shapes: `linear` (default), `convex` (cautious), `concave` (aggressive)
- `min_position` parameter: floor — never go below X% invested even in worst regime

**When to use:** after the core pipeline, if you want smoother position changes instead of abrupt in/out switches.

```bash
python regime_sizing.py
# prompts: stock, weight shape, min_position floor
```

---

### `regime_filters.py` — noise cancellation + severity score
Two additions to clean up noisy regime signals.

**What it does:**

`apply_persistence_filter(regimes, min_days=3)`: only confirms a regime change after it has held for N consecutive days. Eliminates single-day flips that carry no real information. Tradeoff: reduces noise but adds up to N days of lag.

`compare_filter_impact(prices, raw_regimes, [1,2,3,5])`: backtests with different `min_days` values so you can measure the noise-reduction vs lag tradeoff with actual numbers.

`compute_severity_score(proba_df, features)`: collapses all signals into one 0–1 score per day. 0 = extremely bearish (worst regime, high confidence, deep drawdown, below MA). 1 = extremely bullish. Combines: regime probability (40%), model confidence (20%), drawdown (25%), momentum (15%).

`compute_position_size_v2(severity, variant)`: position sizing from severity score instead of raw probabilities. Three variants: `conservative` (severity²), `moderate` (linear), `aggressive` (√severity).

**When to use:** if your regime chart shows heavy flickering (regime changes every 1-2 days), run the persistence filter first. Use severity score if you want a single interpretable number per day instead of a probability vector.

```bash
python regime_filters.py
# prompts: stock, min_position
```

---

## Methodology

### Features (6 total)
| Feature | Window | What it captures |
|---------|--------|-----------------|
| `return` | daily | raw price change |
| `vol_slow` | 21d rolling std | baseline volatility regime |
| `vol_ratio` | fast(5d) / slow(21d) | volatility spike — fires 2-3 days before crash is visible in returns |
| `drawdown` | distance from 60d high | separates "high vol crash" from "high vol recovery" |
| `momentum` | price / 20d MA - 1 | trend direction — distinguishes low-vol uptrend from low-vol downtrend |
| `vol_of_vol` | 10d std of vol_slow | regime stability — high = transitioning market |

### Model Selection
HMMs fitted for k=2 to 4 states. Best k selected by BIC (lower = better fit penalized for complexity). States relabeled 0..k-1 by mean return so labels are human-interpretable and comparable across tickers.

### Walk-Forward (Causal, No Lookahead)
- Train on event-anchored window of past data (resets at major regime shifts)
- Refit every 21 trading days (normal schedule)
- **Adaptive refit:** immediately when `vol_ratio > 2.0` — crash onset detection
- **Event-anchored reset:** major transitions (e.g., crash entry / sharp crash recovery) reset `train_start_idx` so old regime cycles are dropped
- **Reset cooldown:** `min_days_before_reset=252` avoids frequent reset thrashing
- Validate every fit before accepting (checks for NaN parameters)
- First `min_train=252` days held out as initial training window

### Backtest
Binary strategy: fully invested unless in regime 0 (worst), then hold cash. 1-day signal lag (yesterday's regime → today's position). No transaction costs modeled.

---

## Results

Evaluated on 6 NSE-listed tickers, 2015–2026 (~2312 trading days each), walk-forward causal backtest.

| Ticker | Type | BH Sharpe | Strat Sharpe | Sharpe Edge | BH MaxDD | Strat MaxDD | DD Improved | Time in Market |
|--------|------|-----------|--------------|-------------|----------|-------------|-------------|----------------|
| RELIANCE | Conglomerate | 0.77 | 0.67 | -0.09 | -45.1% | -45.1% | 0.0pp | 81.0% |
| HDFCBANK | Private Bank | 0.55 | 0.82 | **+0.28** | -41.1% | -20.8% | **+20.3pp** | 84.6% |
| INFY | IT Large Cap | 0.54 | 0.41 | -0.13 | -48.2% | -44.2% | +3.96pp | 94.4% |
| TCS | IT Large Cap | 0.46 | 0.43 | -0.03 | -53.4% | -37.7% | +15.7pp | 91.0% |
| TATASTEEL | Cyclical | 0.70 | **0.80** | **+0.10** | -64.5% | -50.2% | **+14.3pp** | 83.2% |
| SBIN | PSU Bank | 0.65 | 0.53 | -0.12 | -59.5% | -46.4% | +13.1pp | 81.4% |

### Key Findings

**Sharpe improvement: 2/6 tickers (HDFCBANK, TATASTEEL)**
Both are stocks with genuine volatility clustering and mean-reverting behavior — the HMM's regime labels carry real signal here. HDFCBANK showed the strongest result: Sharpe 0.55 → 0.82 with max drawdown nearly halved (41% → 21%).

**Drawdown reduction: 5/6 tickers**
The strategy consistently reduces worst-case loss even when it underperforms on total return. TCS: 53% → 38%. SBIN: 59% → 46%. The one exception is RELIANCE where the strategy adds no value at all.

**Where the strategy fails: smooth trending large caps**
RELIANCE, INFY, TCS, SBIN — all show the same pattern. The regime chart flickers heavily (regime changes every 1-2 days), the strategy churns in and out, and misses sustained uptrends. The HMM detects real statistical regimes but those regimes don't map cleanly to down/up periods on these stocks.

**Honest conclusion:** the regime detection signal exists and is statistically real. It translates to return outperformance only on stocks with genuine boom/bust cycle structure (cyclicals, volatile financials). On smooth large-cap trending stocks it adds drawdown protection at the cost of return — a tradeoff, not a failure, but worth stating plainly.

---

## Limitations

- **No transaction costs:** every regime switch assumes costless execution. In reality, frequent switching on noisy tickers (SBIN, RELIANCE) would erode returns further.
- **Gaussian emission assumption:** HMM assumes returns within each regime are normally distributed. Crash regimes have fat tails — the model underestimates extreme event probability.
- **Fixed feature weights in severity score:** the 40/20/25/15 weights in `compute_severity_score` are judgment calls, not statistically optimized. Optimizing them on historical data risks overfitting.
- **Persistence filter lag:** `apply_persistence_filter` with `min_days=3` means you're always 3 days late confirming a new regime. On a COVID-speed crash this matters.
- **Single binary strategy:** the backtest only tests "avoid regime 0." The sizing layer (`regime_sizing.py`) is a better approach but its results aren't included in the cross-ticker eval above.
- **Data from 2015:** misses the 2008-2013 period which would include more crash regimes for training.

---

## Installation

```bash
pip install yfinance hmmlearn pandas numpy matplotlib scikit-learn
```

Python 3.9+ required. Tested on macOS (M4) and Linux.

---

## Quick Start

```bash
# Run on a single stock
python regime_hmm.py
# > Enter stock name: TATASTEEL

# Evaluate across multiple tickers
python evaluate_regimes.py --tickers HDFCBANK TATASTEEL TCS

# Add continuous position sizing
python regime_sizing.py
# > Enter stock: HDFCBANK
# > Weight shape: linear
# > Min position: 0.1

# Apply persistence filter to reduce noise
python regime_filters.py
# > Enter stock: SBIN
```

---

## Which File to Use When

| Goal | File |
|------|------|
| Understand regimes for one stock | `regime_hmm_v2.py` |
| Compare model across many stocks | `evaluate_regimes.py` |
| Smoother position sizing | `regime_sizing.py` |
| Regime chart is too flickery | `regime_filters.py` → `apply_persistence_filter` |
| Want one number per day ("how bearish?") | `regime_filters.py` → `compute_severity_score` |
| Build on top of this pipeline | Import from `regime_hmm.py` — all functions are modular |