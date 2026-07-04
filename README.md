# Market Regime Detection with Hidden Markov Models

Detects hidden "market regimes" (e.g. calm uptrend, high-vol selloff,
sideways chop) in a stock or index's price history using a Gaussian Hidden
Markov Model, then checks whether knowing the regime would have actually
helped a simple trading strategy beat buy-and-hold.

## What it does

1. **Downloads price history** for a ticker (NSE stocks by default, or the
   Nifty 50 index) via `yfinance`, starting from 2015.
2. **Builds features**: daily return and a 10-day rolling volatility.
3. **Picks the number of regimes (k)** automatically by fitting Gaussian
   HMMs for k = 2..4 and selecting the one with the lowest BIC (Bayesian
   Information Criterion), which balances fit quality against model
   complexity instead of guessing a fixed number of states.
4. **Labels each regime** by its historical mean return, so state 0 is
   always "worst" and state k-1 is always "best" — this makes the labels
   comparable across tickers and across refits.
5. **Fits two versions of the model**:
   - **Global fit** — trained once on the *entire* history. Useful as a
     sanity check but has lookahead bias: it "knows" about crashes and
     rallies before they happen, so its backtest result is inflated.
   - **Walk-forward fit** — refit every 21 trading days on an expanding
     window of only *past* data, and each day's regime is decoded using
     only data up to that day. This is the causal, honest version — no
     information from the future leaks into a regime label.
6. **Backtests a toy strategy**: stay invested unless the walk-forward
   model says you're in the worst regime, in which case go to cash.
   Compared against plain buy-and-hold, both in total and broken down
   **month by month**.
7. **Evaluates across multiple tickers** (`evaluate_regimes.py`) — runs
   the full walk-forward pipeline on a list of stocks, computes a full
   suite of metrics for each, and produces a cross-ticker summary table
   and comparison charts. This is what turns the project from a demo into
   a study.
8. **Produces a bullish/bearish verdict** for the most recent day, based on
   which regime it's currently in and how long it's persisted.
9. **Saves annotated plots** of the full history, three random months
   zoomed in, and the latest week/month, all colored by detected regime.
10. **Regime-conditioned position sizing** (`regime_sizing.py`) — replaces
    the binary invested/cash signal with a continuous position size derived
    from the HMM's posterior regime probabilities, so exposure scales
    smoothly with model confidence instead of being a hard on/off switch.

## Files

| File | Purpose |
|---|---|
| `regime_hmm.py` | Core HMM pipeline — data download, feature engineering, model selection, walk-forward fit, backtesting, plots, verdict |
| `regime_sizing.py` | Companion module — probability-based position sizing, worst-regime exposure report, three-strategy comparison with full metrics |
| `evaluate_regimes.py` | Evaluation module — multi-ticker backtest with Sharpe, Sortino, Calmar, Max Drawdown, CAGR, monthly win-rate; summary CSV and comparison charts |

## Usage

**Core pipeline (single ticker):**
```bash
python regime_hmm.py
# Enter stock name (or 'NSEI' for Nifty index): RELIANCE
```

**Multi-ticker evaluation:**
```bash
python evaluate_regimes.py
# Press Enter for defaults (5 NSE + 5 US stocks), or type: RELIANCE,TCS,INFY
```

**Position sizing:**
```bash
python regime_sizing.py
# Enter stock (or NSEI for Nifty): TCS
# Weight shape [linear/convex/concave] (default: linear): linear
# Minimum position floor: 0.2
```

Or from another script:

```python
from regime_hmm import run
run("RELIANCE")          # uses yfinance
run("NSEI")              # Nifty 50 index
run("MYTICKER", data_loader=my_price_loader)  # custom price source

from regime_sizing import run_sized
run_sized("TCS", shape="linear", min_position=0.2)

from evaluate_regimes import run_evaluation
tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "WIPRO",
           "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
summary = run_evaluation(tickers, output_dir="outputs/evaluation")
```

## Output structure

```
outputs/
  <TICKER>/                        <- single-ticker run (regime_hmm.py)
    report.txt
    monthly_breakdown.csv
    plots/
      <TICKER>_regimes_lookahead.png
      <TICKER>_regimes_walkforward.png
      <TICKER>_regimes_monthly_zoom.png
      <TICKER>_regimes_latest_week.png
      <TICKER>_regimes_latest_month.png
      <TICKER>_sized_position.png  <- from regime_sizing.py

  evaluation/                      <- multi-ticker run (evaluate_regimes.py)
    summary.csv
    plots/
      <TICKER>_evaluation.png      <- hero chart per ticker
      summary_comparison.png       <- cross-ticker Sharpe + drawdown bar chart
```

## Evaluation metrics (Part 1)

`evaluate_regimes.py` answers the question *"does my model actually work?"*
with the following metrics, computed for both buy-and-hold and the regime
strategy, on every ticker:

| Metric | What it tells you |
|---|---|
| **Total Return %** | Raw compounded return over the full history |
| **CAGR %** | Annualised return — comparable across tickers with different history lengths |
| **Sharpe** | Return per unit of total volatility — the standard risk-adjusted measure |
| **Sortino** | Like Sharpe but only penalises downside volatility — fairer to strategies that cut losses |
| **Calmar** | CAGR divided by worst drawdown — measures return per unit of worst-case pain |
| **Max Drawdown %** | Worst peak-to-trough loss — the number that tells you how bad it can get |
| **Monthly Win %** | Fraction of calendar months with a positive return |
| **% Time in Market** | How often the strategy was invested (vs sitting in cash) |

**Sharpe edge** (strategy Sharpe minus buy-and-hold Sharpe) and **Max DD
improvement** (how many percentage points less severe the strategy's worst
drawdown was) are highlighted per ticker and aggregated at the end.

The cross-ticker summary is what you'd cite in a writeup: *"the regime
strategy improved Sharpe on 8/10 tickers and reduced max drawdown on 9/10,
with an average Sharpe edge of +0.18."* That is a falsifiable, honest claim
about whether the model carries signal — which is what separates this from
a demo.

`regime_sizing.py` now prints the same full metrics table (Total Return,
CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Monthly Win %, Avg Position)
for all three strategies (Buy & Hold, Binary, Sized), with `*` marking
where the strategy beats buy-and-hold, and explicit Sharpe edge and
drawdown improvement callouts.

## What's new in this version

### evaluate_regimes.py (new)
- Runs the full walk-forward pipeline on any list of tickers and collects
  a full suite of metrics for each: Total Return, CAGR, Sharpe, Sortino,
  Calmar, Max Drawdown, Monthly Win Rate, % time in market.
- Computes **Sharpe edge** and **Max Drawdown improvement** per ticker so
  you can see at a glance where the model adds value and where it doesn't.
- Prints aggregate stats across all tickers: how many it beat on Sharpe,
  how many it reduced drawdown on, average edge.
- Saves `summary.csv` and two charts: a **hero chart** per ticker
  (regime-colored price bands + equity curves) and a **cross-ticker
  comparison bar chart** (Sharpe and drawdown side by side).
- Tickers that fail (too little history, bad symbol, download error) are
  skipped cleanly with a printed reason — the rest still run.

### regime_sizing.py (updated)
- `backtest_sized_strategy()` now prints a **full formatted metrics table**
  for all three strategies instead of just three summary lines. Includes
  Total Return, CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Monthly Win %,
  and Avg Position, with `*` markers where the strategy beats buy-and-hold.

### regime_hmm.py (previous version, carried forward)
- Month-by-month backtest breakdown saved to `monthly_breakdown.csv`
- Full console transcript captured to `report.txt` via `Tee`
- Organized output under `outputs/<TICKER>/`
- Clearer plot titles, axis labels, legends
- Robust walk-forward fitting with NaN validation
- Testable `data_loader` injection in `run()`

## Known limitations / honest caveats

- **`resolve_ticker` assumes NSE by default.** A plain ticker like `AAPL`
  gets turned into `AAPL.NS` first, then retried as `AAPL` if that fails.
  This covers most cases but explicit suffixes (`.BO`, `.L`, etc.) are not
  handled — extend `resolve_ticker` if you need other exchanges.
- **The bullish/bearish thresholds in `classify_bullish_bearish`** are
  rough judgment calls (±7%, ±20% annualized), not statistically derived.
- **Monthly Sharpe ratios are noisy.** Each calendar month has only ~21
  daily return observations, so monthly Sharpe should be read as a rough
  directional signal, not a precise estimate — the report says this explicitly.
- **The backtest strategy is intentionally simple** — fully invested or
  fully in cash, no transaction costs, no slippage. It exists to test
  whether regime labels carry *any* signal, not as a deployable strategy.
- **This is not investment advice.**

## Possible future improvements

- **Transaction costs & slippage** in the backtest.
- **More features** beyond return/volatility (e.g. volume, RSI,
  moving-average crossovers).
- ~~**Regime-conditional position sizing**~~ ✅ Done — see `regime_sizing.py`.
- ~~**Multi-ticker evaluation with proper metrics**~~ ✅ Done — see `evaluate_regimes.py`.
- **Statistical significance testing** on the monthly breakdown (e.g.
  bootstrap confidence intervals on `difference_%`) rather than eyeballing
  win-rate and mean/std.
- **Regime transition detector** — a `regime_monitor.py` that fires only
  when a genuine regime change is confirmed (`P(new_regime) > 0.7` held
  for N consecutive days), suppressing single-day flickers.
- **Parallelise `walk_forward_regimes`** — refits at different `t` cutoffs
  are independent and could run concurrently.
- **Config file / CLI args** for `vol_window`, `min_train`, `refit_every`,
  `max_states`, and `avoid_regime`.
- **Unit tests** for pure functions using synthetic price series.