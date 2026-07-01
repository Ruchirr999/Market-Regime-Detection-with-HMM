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
   Compared against plain buy-and-hold, both in total and now broken down
   **month by month**.
7. **Produces a bullish/bearish verdict** for the most recent day, based on
   which regime it's currently in and how long it's persisted.
8. **Saves annotated plots** of the full history, three random months
   zoomed in, and the latest week/month, all colored by detected regime.

## What's new in this version

The original script worked but printed everything to the console only,
left plot files scattered loose in the working directory with fairly bare
titles, and only reported one all-time backtest number.

- **Month-by-month backtest breakdown** (`monthly_backtest_breakdown`) —
  for every calendar month in the sample: buy&hold return, strategy
  return, the difference, a rough monthly Sharpe for each, how many days
  the strategy sat in cash, and which regime dominated that month. This
  is saved to `monthly_breakdown.csv` and also printed. It answers "did
  avoiding the worst regime actually help in a specific month, or only on
  average over years" — a single all-time number can hide months where
  the strategy badly underperformed even if the multi-year total looks
  good.
- **Full console transcript saved to `report.txt`** — every `print()`
  statement already in the script (regime stats, BIC table, backtests,
  verdict) is now also captured to a plain-text file via a small `Tee`
  stdout wrapper, so nothing has to be copy-pasted out of the terminal.
- **Organized output folder structure** — everything for a given ticker
  now lands under `outputs/<TICKER>/`:
  ```
  outputs/<TICKER>/
    report.txt              <- full run transcript
    monthly_breakdown.csv   <- month-by-month table
    plots/
      <TICKER>_regimes_lookahead.png
      <TICKER>_regimes_walkforward.png
      <TICKER>_regimes_monthly_zoom.png
      <TICKER>_regimes_latest_week.png
      <TICKER>_regimes_latest_month.png
  ```
  instead of loose files mixed into the project root.
- **Clearer plots** — every plot now has an x-axis label, y-axis label, a
  light grid, a legend title explaining what "regime 0" vs "regime k-1"
  means, and a title that names the ticker and (for the two main regime
  plots) whether it's the lookahead or walk-forward version, so a plot
  can't be misread out of context.
- **Testable data source** — `run()` now accepts an optional
  `data_loader` callable so the whole pipeline can be exercised with any
  price series (e.g. synthetic data, a CSV, a cached download), not only
  a live `yfinance` call. `run("RELIANCE")` still works exactly as before
  and defaults to `yfinance`.

## Usage

```bash
python market_regime.py
# Enter stock name (or 'NSEI' for Nifty index): RELIANCE
```

Or from another script:

```python
from market_regime import run
run("RELIANCE")          # uses yfinance
run("NSEI")               # Nifty 50 index
run("MYTICKER", data_loader=my_price_loader)  # custom price source
```

Outputs appear under `outputs/<LABEL>/`.

## Known limitations / caveats (carried over and worth restating)

- **`resolve_ticker` assumes NSE by default.** A plain ticker like `AAPL`
  gets turned into `AAPL.NS`, which is wrong for US stocks. Pass an
  explicit suffix (`AAPL.US`-style handling is not implemented) or extend
  `resolve_ticker` if you need non-Indian tickers.
- **The bullish/bearish thresholds in `classify_bullish_bearish`** are
  rough judgment calls (±7%, ±20% annualized), not statistically derived
  cutoffs.
- **Monthly Sharpe ratios are noisy.** Each calendar month has only ~21
  daily return observations, so `strategy_sharpe` / `buy_hold_sharpe` in
  the monthly breakdown should be read as a rough directional signal, not
  a precise estimate — the report says this explicitly.
- **The backtest strategy is intentionally simple** (fully invested or
  fully in cash, no transaction costs, no slippage, no position sizing).
  It exists to check whether the regime labels carry *any* signal, not
  as a deployable strategy.
- **This is not investment advice.** The regime label and verdict
  describe a statistical pattern in historical data, not a forecast.

## Possible future improvements

- **Transaction costs & slippage** in the backtest — the current total
  and monthly returns assume free, instant rebalancing.
- **More features** beyond return/volatility (e.g. volume, RSI, moving-average
  crossovers) to help the HMM separate regimes that look similar in
  return/vol space alone.
- **Regime-conditional position sizing** instead of a binary invested/cash
  switch (e.g. scale exposure down rather than fully exiting).
- **Statistical significance testing** on the monthly breakdown (e.g. a
  paired t-test or bootstrap on `difference_%`) rather than eyeballing the
  win-rate and mean/std.
- **Parallelize `walk_forward_regimes`** — refitting every 21 days is
  sequential and can be slow on long histories; refits at different
  `t` cutoffs are independent and could run concurrently.
- **Ticker resolution for non-NSE markets** — generalize `resolve_ticker`
  instead of hardcoding the `.NS` suffix assumption.
- **Config file / CLI args** for `vol_window`, `min_train`, `refit_every`,
  `max_states`, and `avoid_regime` instead of editing constants in code.
- **Unit tests** for the pure functions (`build_features`,
  `classify_bullish_bearish`, `monthly_backtest_breakdown`) using
  synthetic price series, so changes to the HMM logic can't silently
  break the reporting layer.