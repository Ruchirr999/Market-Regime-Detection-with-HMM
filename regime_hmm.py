"""
Market Regime Detection using Hidden Markov Models
----------------------------------------------------
Fits a Gaussian HMM on (return, rolling volatility) to infer hidden
market regimes (e.g. calm-uptrend, high-vol-selloff, sideways-chop),
then backtests a regime-conditioned toy strategy against buy-and-hold.

This version adds, on top of the original script:
  - A month-by-month breakdown of buy&hold vs. regime-strategy performance
  - Everything printed to the console is also saved to a single .txt report
  - All output files (plots, report, csv) are organised into
    outputs/<TICKER>/... instead of dumping loose files in the cwd
  - Clearer plot titles/axis labels/legends
  - A README.md describing the project, methodology, and next steps
"""

import os
import sys
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn.hmm import GaussianHMM


# --------------------------------------------------------------------------
# Output organisation
# --------------------------------------------------------------------------

class Tee:
    """Duplicates everything written to stdout into a log file as well,
    so the full console transcript (all the print() statements already in
    this script) ends up saved as a readable .txt report with zero changes
    to the print statements themselves."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def make_output_dirs(label: str, root: str = "outputs") -> dict:
    """Creates outputs/<label>/ and outputs/<label>/plots/ and returns
    the paths so every function in this script writes to the same place."""
    base = os.path.join(root, label)
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    return {
        "base": base,
        "plots": plots,
        "report_txt": os.path.join(base, "report.txt"),
        "monthly_csv": os.path.join(base, "monthly_breakdown.csv"),
    }


# --------------------------------------------------------------------------
# Ticker handling
# --------------------------------------------------------------------------

def resolve_ticker(stock_input: str) -> tuple[str, str, str]:
    """
    Turns user input into a yfinance ticker + a clean label for filenames.
    - "NSEI" / "NIFTY" / "NIFTY50" -> the index itself (^NSEI)
    - anything else -> assumed to be an NSE-listed stock, gets ".NS" appended
      unless the user already typed a suffix (e.g. "RELIANCE.NS").

    Returns (ticker, label, fallback_ticker):
      - `ticker` is the NSE-assumed candidate to try first (e.g. "AAPL.NS").
      - `fallback_ticker` is the raw, unmodified input (e.g. "AAPL"), which
        the caller can retry with if the NSE-suffixed candidate returns no
        data -- this is what makes a US ticker like "AAPL" work correctly
        instead of only ever trying (and failing on) "AAPL.NS".
    """
    raw = stock_input.strip().upper()
    index_aliases = {"NSEI", "NIFTY", "NIFTY50", "NIFTY 50", "^NSEI"}

    if raw in index_aliases:
        return "^NSEI", "NIFTY50", "^NSEI"

    if "." in raw or raw.startswith("^"):
        ticker = raw
    else:
        ticker = f"{raw}.NS"

    label = raw.replace(".", "_").replace("^", "")
    return ticker, label, raw


def classify_bullish_bearish(ann_return_pct: float) -> str:
    """Maps the current regime's annualized return to a human verdict.
    Thresholds are rough judgment calls, not statistically derived --
    worth stating that plainly rather than pretending precision."""
    if ann_return_pct >= 20:
        return "Strongly Bullish"
    elif ann_return_pct >= 7:
        return "Mildly Bullish"
    elif ann_return_pct > -7:
        return "Neutral / Sideways"
    elif ann_return_pct > -20:
        return "Mildly Bearish"
    else:
        return "Strongly Bearish"


def latest_regime_verdict(regimes: pd.Series, regime_stats: pd.DataFrame, label: str):
    """Prints the current regime, how long the stock has been in it,
    and a bullish/bearish verdict with rough magnitude."""
    latest_date = regimes.index[-1]
    latest_regime = regimes.iloc[-1]

    streak = 1
    for i in range(len(regimes) - 2, -1, -1):
        if regimes.iloc[i] == latest_regime:
            streak += 1
        else:
            break

    ann_return = regime_stats.loc[latest_regime, "ann_return_%"]
    verdict = classify_bullish_bearish(ann_return)

    print(f"\n=== {label}: Current Verdict ===")
    print(f"As of {latest_date.date()}: Regime {latest_regime}, "
          f"in this regime for {streak} consecutive trading day(s)")
    print(f"Historical annualized return of this regime: {ann_return:.1f}%")
    print(f"Verdict: {verdict}")
    print("(Note: this describes the recent/current statistical regime, "
          "not a forecast of future direction.)")
    return verdict


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_random_months(prices: pd.Series, regimes: pd.Series, n_states: int,
                        label: str, num_months: int = 3, save_path: str = None):
    """Picks random months from the dataset and plots them side-by-side so
    you can see day-to-day regime changes up close."""
    aligned = prices.loc[regimes.index].to_frame(name="price")
    aligned["regime"] = regimes
    aligned["YearMonth"] = aligned.index.to_period("M").astype(str)
    unique_months = aligned["YearMonth"].unique().tolist()

    sampled_months = random.sample(unique_months, min(num_months, len(unique_months)))
    sampled_months.sort()

    fig, axes = plt.subplots(1, num_months, figsize=(5 * num_months, 5))
    if num_months == 1:
        axes = [axes]

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_states))

    for ax, month in zip(axes, sampled_months):
        month_data = aligned[aligned["YearMonth"] == month]

        ax.plot(month_data.index, month_data["price"], color="black", alpha=0.3,
                 linewidth=1, label="Price" if ax is axes[0] else None)

        for state in range(n_states):
            mask = month_data["regime"] == state
            if mask.any():
                ax.scatter(month_data.index[mask], month_data.loc[mask, "price"],
                            c=[colors[state]], s=30,
                            label=f"Regime {state}", zorder=5)

        ax.set_title(f"{month}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Day of month")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        if ax == axes[0]:
            ax.legend(loc="best", fontsize=8, title="Legend")

    fig.suptitle(f"{label}: Regime Zoom-In on Random Months "
                  f"(state 0 = worst regime, state {n_states - 1} = best)",
                  fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved monthly zoom plots -> {save_path}")


def plot_recent_window(prices: pd.Series, regimes: pd.Series, n_states: int,
                        n_days: int, window_label: str, ticker_label: str,
                        save_path: str = None):
    """Plots the most recent `n_days` trading days of data, colored by regime.
    Use n_days=5 for 'latest week', n_days=21 for 'latest month'
    (21 trading days ~= 1 calendar month)."""
    aligned = prices.loc[regimes.index].to_frame(name="price")
    aligned["regime"] = regimes
    recent = aligned.tail(n_days)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_states))

    ax.plot(recent.index, recent["price"], color="black", alpha=0.3,
             linewidth=1, label="Price")

    for state in range(n_states):
        mask = recent["regime"] == state
        if mask.any():
            ax.scatter(recent.index[mask], recent.loc[mask, "price"],
                        c=[colors[state]], s=40, label=f"Regime {state}", zorder=5)

    ax.set_title(f"{ticker_label}: Last {window_label} "
                 f"({recent.index[0].date()} to {recent.index[-1].date()})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.legend(loc="best", title="0 = worst regime")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved -> {save_path}")


def plot_regimes(prices: pd.Series, regimes: pd.Series, n_states: int,
                  ticker_label: str, subtitle: str, save_path: str = None):
    aligned = prices.loc[regimes.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_states))
    for state in range(n_states):
        mask = regimes == state
        ax.scatter(aligned.index[mask], aligned[mask], c=[colors[state]],
                    s=6, label=f"Regime {state}")
    ax.plot(aligned.index, aligned.values, color="black", alpha=0.2, linewidth=0.8)
    ax.legend(loc="upper left", title="0 = worst regime, higher = better")
    ax.set_title(f"{ticker_label}: Price Colored by Detected HMM Regime\n{subtitle}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved plot -> {save_path}")


# --------------------------------------------------------------------------
# Feature engineering / model selection / labeling
# --------------------------------------------------------------------------

def build_features(prices: pd.Series, vol_window: int = 10) -> pd.DataFrame:
    """From a price series, build the (return, volatility) feature matrix
    the HMM will be fit on."""
    df = pd.DataFrame(index=prices.index) #empty box
    df["return"] = prices.pct_change() #today-yesterday/yesterday
    df["volatility"] = df["return"].rolling(vol_window).std() #rolling 10 day std deviation of results
    df = df.dropna() #dropping first 9 rows since they have NA under volatility
    return df


def select_n_states(features: np.ndarray, max_states: int = 5, n_iter: int = 200):
    """Fit HMMs for k=2..max_states and pick the best by BIC.
    Lower BIC = better fit penalized for complexity -- this is how you
    justify the number of regimes instead of just guessing K=3."""
    results = []
    for k in range(2, max_states + 1):
        model = GaussianHMM(n_components=k, covariance_type="diag",
                             n_iter=n_iter, random_state=42)
        model.fit(features)
        log_likelihood = model.score(features)
        n_params = k * (k - 1) + k * features.shape[1] * 2
        bic = -2 * log_likelihood + n_params * np.log(len(features))
        results.append((k, bic, model))
        print(f"  k={k}: log-likelihood={log_likelihood:.1f}, BIC={bic:.1f}")
    best = min(results, key=lambda x: x[1])
    print(f"-> Selected k={best[0]} states by BIC")
    return best[2], best[0]


def label_regimes(model: GaussianHMM, features: pd.DataFrame) -> pd.Series:
    """Run Viterbi to get the most likely regime sequence, then relabel
    states 0..k-1 by their mean return so labels are human-interpretable
    (state 0 = worst regime, state k-1 = best)."""
    hidden_states = model.predict(features.values)
    state_returns = [features["return"][hidden_states == s].mean()
                      for s in range(model.n_components)]
    order = np.argsort(state_returns)
    remap = {old: new for new, old in enumerate(order)}
    relabeled = pd.Series([remap[s] for s in hidden_states], index=features.index)
    return relabeled


def describe_regimes(features: pd.DataFrame, regimes: pd.Series):
    """Print mean return / vol per regime so you can sanity check what
    each state actually represents before trusting it."""
    if len(regimes) == 0:
        raise ValueError(
            "describe_regimes: regimes is empty, so there's nothing to "
            "summarize. This usually means walk_forward_regimes() was given "
            "less history than it could label -- check the NOTE/error it "
            "printed above."
        )
    summary = features.groupby(regimes).agg(
        mean_return=("return", "mean"),
        volatility=("volatility", "mean"),
        days=("return", "count"),
    )
    summary["ann_return_%"] = summary["mean_return"] * 252 * 100
    print(summary)
    return summary


MIN_VIABLE_TRAIN_ROWS = 60  # below this, an HMM fit is too noisy to trust at all


def walk_forward_regimes(features: pd.DataFrame, n_states: int,
                          min_train: int = 500, refit_every: int = 21,
                          n_iter: int = 100) -> pd.Series:
    values = features.values
    n = len(values)

    if n < MIN_VIABLE_TRAIN_ROWS + 5:
        raise ValueError(
            f"walk_forward_regimes: only {n} feature rows are available for "
            f"this ticker -- that's too little history for any HMM fit to be "
            f"meaningful (need at least ~{MIN_VIABLE_TRAIN_ROWS + 5} trading "
            f"days). This ticker is likely too recently listed, or the date "
            f"range is too short. Try a longer history or a different ticker."
        )

    if min_train >= n:
        adjusted = max(MIN_VIABLE_TRAIN_ROWS, int(n * 0.6))
        adjusted = min(adjusted, n - 5)
        print(f"NOTE: min_train={min_train} but only {n} feature rows are "
              f"available for this ticker. Shrinking min_train to {adjusted}.")
        min_train = adjusted

    regimes = pd.Series(index=features.index, dtype=float)

    model = None
    last_refit = -1
    remap = None

    for t in range(min_train, n):
        if model is None or (t - last_refit) >= refit_every:
            train_data = values[:t]
            candidate = GaussianHMM(n_components=n_states, covariance_type="diag",
                                    n_iter=n_iter, random_state=42)
            fit_ok = False
            try:
                candidate.fit(train_data)
                # Validate the model before accepting it: a failed fit leaves
                # startprob_ / transmat_ with NaNs, which causes predict() to
                # raise "startprob_ must sum to 1 (got nan)".
                if (np.isfinite(candidate.startprob_).all() and
                        np.isfinite(candidate.transmat_).all() and
                        np.isfinite(candidate.means_).all()):
                    fit_ok = True
            except Exception as e:
                print(f"  [walk_forward] HMM fit failed at t={t}: {e} -- "
                      f"keeping previous model.")

            if fit_ok:
                model = candidate
                last_refit = t

                train_states = model.predict(train_data)
                return_col = features.columns.get_loc("return")
                state_means = [train_data[train_states == s, return_col].mean()
                               if (train_states == s).any() else 0.0
                               for s in range(n_states)]
                order = np.argsort(state_means)
                remap = {old: new for new, old in enumerate(order)}

        # If we still have no valid model yet, skip this day
        if model is None or remap is None:
            continue

        try:
            window = values[:t + 1]
            state_seq = model.predict(window)
            regimes.iloc[t] = remap[state_seq[-1]]
        except Exception:
            # predict failed for this day (e.g. degenerate window) -- leave NaN
            pass

    regimes = regimes.dropna().astype(int)
    return regimes


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------

def _sharpe(r: pd.Series) -> float:
    return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan


def backtest_regime_strategy(prices: pd.Series, features: pd.DataFrame,
                              regimes: pd.Series, avoid_regime: int = 0):
    """Toy strategy: hold the index unless we're in the worst regime
    (default: lowest-return state), then hold cash. Compare vs buy-and-hold.
    This is NOT investment advice -- it's a demonstration of whether the
    regime labels carry any signal at all."""
    if len(features) == 0:
        raise ValueError(
            "backtest_regime_strategy: features is empty, so there are no "
            "days to backtest. This usually means walk_forward_regimes() "
            "was given less history than it could label -- check the "
            "NOTE/error it printed above."
        )
    aligned_prices = prices.loc[features.index]
    daily_ret = aligned_prices.pct_change().fillna(0)

    position = (regimes != avoid_regime).astype(int)
    strategy_ret = daily_ret * position.shift(1).fillna(0)

    bh_equity = (1 + daily_ret).cumprod()
    strat_equity = (1 + strategy_ret).cumprod()

    print(f"Buy & Hold   -> total return: {(bh_equity.iloc[-1]-1)*100:.1f}%, "
          f"Sharpe: {_sharpe(daily_ret):.2f}")
    print(f"Regime Strat -> total return: {(strat_equity.iloc[-1]-1)*100:.1f}%, "
          f"Sharpe: {_sharpe(strategy_ret):.2f}")

    return bh_equity, strat_equity, daily_ret, strategy_ret


def monthly_backtest_breakdown(prices: pd.Series, features: pd.DataFrame,
                                regimes: pd.Series, avoid_regime: int = 0,
                                label: str = "", save_path: str = None) -> pd.DataFrame:
    """Month-by-month version of backtest_regime_strategy: for every
    calendar month in the sample, reports buy&hold return, regime-strategy
    return, each one's Sharpe (computed on that month's daily returns --
    noisy with ~21 data points, treat as directional not precise), the
    number of days the strategy was in cash that month, and which regime
    was most common that month.

    This answers "did avoiding the worst regime actually help in March
    2020 specifically, or only on average over years" -- the single
    all-time backtest number can hide months where the strategy did
    great and months where it badly underperformed buy & hold.
    """
    aligned_prices = prices.loc[features.index]
    daily_ret = aligned_prices.pct_change().fillna(0)
    position = (regimes != avoid_regime).astype(int)
    strategy_ret = daily_ret * position.shift(1).fillna(0)

    df = pd.DataFrame({
        "bh_ret": daily_ret,
        "strat_ret": strategy_ret,
        "regime": regimes,
        "in_cash": (position == 0).astype(int),
    })
    df["YearMonth"] = df.index.to_period("M")

    rows = []
    for period, g in df.groupby("YearMonth"):
        bh_month_ret = (1 + g["bh_ret"]).prod() - 1
        strat_month_ret = (1 + g["strat_ret"]).prod() - 1
        rows.append({
            "month": str(period),
            "trading_days": len(g),
            "buy_hold_return_%": bh_month_ret * 100,
            "strategy_return_%": strat_month_ret * 100,
            "difference_%": (strat_month_ret - bh_month_ret) * 100,
            "buy_hold_sharpe": _sharpe(g["bh_ret"]),
            "strategy_sharpe": _sharpe(g["strat_ret"]),
            "days_in_cash": int(g["in_cash"].sum()),
            "dominant_regime": int(g["regime"].mode().iloc[0]),
        })

    monthly = pd.DataFrame(rows).set_index("month")

    print(f"\n=== {label}: Month-by-Month Backtest Breakdown "
          f"({len(monthly)} months) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(monthly.round(2))

    months_strategy_won = (monthly["difference_%"] > 0).sum()
    print(f"\nStrategy beat buy&hold in {months_strategy_won}/{len(monthly)} "
          f"months ({months_strategy_won / len(monthly) * 100:.0f}%).")
    print(f"Average monthly outperformance: {monthly['difference_%'].mean():.2f} "
          f"percentage points (std: {monthly['difference_%'].std():.2f}).")
    print("(Monthly Sharpe is computed on ~21 daily returns per month, so "
          "treat it as a rough directional signal, not a precise estimate.)")

    if save_path:
        monthly.to_csv(save_path)
        print(f"Saved monthly breakdown -> {save_path}")

    return monthly


# --------------------------------------------------------------------------
# Price loading
# --------------------------------------------------------------------------

def _download_prices(ticker: str) -> pd.Series:
    """Downloads close prices for a single ticker from yfinance and
    normalizes to a clean, dropna'd pd.Series (empty Series if nothing
    was found -- yfinance doesn't raise on an unknown/delisted ticker,
    it just returns an empty frame, so callers must check for that)."""
    import yfinance as yf
    raw = yf.download(ticker, start="2015-01-01")["Close"]
    prices = raw.dropna()
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    return prices


def load_prices_with_fallback(ticker: str, fallback_ticker: str) -> tuple[pd.Series, str]:
    """Tries `ticker` first (the NSE-assumed candidate, e.g. "AAPL.NS").
    If that returns no data, retries with `fallback_ticker` (the raw,
    unmodified input, e.g. "AAPL") -- this is what makes tickers from
    exchanges other than NSE (US stocks like AAPL, MSFT, etc.) work
    correctly instead of always failing silently on a wrongly-guessed
    ".NS" suffix. Raises a clear error if neither candidate has data,
    instead of letting an empty price series propagate downstream into
    a confusing HMM/array error.

    Returns (prices, ticker_actually_used).
    """
    prices = _download_prices(ticker)
    if not prices.empty:
        return prices, ticker

    if fallback_ticker != ticker:
        print(f"NOTE: no data found for '{ticker}' -- retrying as "
              f"'{fallback_ticker}' (looks like this isn't an NSE-listed "
              f"ticker).")
        prices = _download_prices(fallback_ticker)
        if not prices.empty:
            return prices, fallback_ticker

    tried = ticker if fallback_ticker == ticker else f"{ticker} and {fallback_ticker}"
    raise ValueError(
        f"No price data found for '{ticker}' (tried: {tried}). "
        f"Double-check the ticker symbol -- for NSE stocks just the plain "
        f"name (e.g. 'RELIANCE') or explicit '.NS' suffix works, for "
        f"US/other-exchange stocks use the plain ticker (e.g. 'AAPL') or "
        f"the correct Yahoo Finance suffix for that exchange."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

# Walk-forward defaults. These are *targets*, not hard requirements --
# run() scales them down automatically for tickers with limited history
# (e.g. recent IPOs) instead of crashing.
DEFAULT_MIN_TRAIN = 500       # preferred training window before labeling starts
MIN_TRAIN_FLOOR = 100         # below this, an HMM fit is too unstable to trust
MIN_WALK_FORWARD_TEST_DAYS = 60  # need at least ~3 months of labeled days for
                                  # the walk-forward backtest/verdict to mean anything


def run(stock_input: str, data_loader=None, min_train: int = None, refit_every: int = 21):
    """
    data_loader: callable(ticker:str) -> pd.Series of close prices,
    indexed by date. Defaults to yfinance if not provided. Injectable so
    the pipeline can be tested/run against any price source, not just
    Yahoo Finance.

    min_train: how many days of history to require before the walk-forward
    model starts labeling regimes. Defaults to None, which auto-scales:
    uses 500 if the ticker has enough history, otherwise shrinks it (down
    to MIN_TRAIN_FLOOR) so there are still at least
    MIN_WALK_FORWARD_TEST_DAYS days left to actually walk-forward test on.
    Pass an explicit int to override.
    """
    ticker, label, fallback_ticker = resolve_ticker(stock_input)
    paths = make_output_dirs(label)

    tee = Tee(paths["report_txt"])
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Ticker candidate: {ticker}  |  Label: {label}")

        if data_loader is None:
            prices, ticker_used = load_prices_with_fallback(ticker, fallback_ticker)
            print(f"Ticker used: {ticker_used}")
        else:
            prices = data_loader(ticker)

        if prices.empty:
            raise ValueError(
                f"No price data available for '{stock_input}' -- cannot "
                f"continue."
            )

        features = build_features(prices)

        print(f"\n=== 1. Global fit for {label} (lookahead bias -- for comparison only) ===")
        global_model, k = select_n_states(features.values, max_states=4)
        global_regimes = label_regimes(global_model, features)
        describe_regimes(features, global_regimes)
        print("\nBacktest (has lookahead bias, inflated result expected):")
        backtest_regime_strategy(prices, features, global_regimes, avoid_regime=0)
        plot_regimes(prices, global_regimes, k, label,
                     subtitle="Global fit -- has lookahead bias, for comparison only",
                     save_path=os.path.join(paths["plots"], f"{label}_regimes_lookahead.png"))

        print(f"\n\n=== 2. Walk-forward causal fit for {label} (honest version) ===")
        requested_min_train = 500 if min_train is None else min_train
        wf_regimes = walk_forward_regimes(features, n_states=k,
                                           min_train=requested_min_train,
                                           refit_every=refit_every)
        wf_features = features.loc[wf_regimes.index]
        held_out = len(features) - len(wf_regimes)
        print(f"Labeled {len(wf_regimes)} days (first {held_out} held out as "
              f"the initial training window)")
        wf_stats = describe_regimes(wf_features, wf_regimes)
        print("\nBacktest (causal, no lookahead):")
        backtest_regime_strategy(prices, wf_features, wf_regimes, avoid_regime=0)
        plot_regimes(prices, wf_regimes, k, label,
                     subtitle="Walk-forward causal fit -- no lookahead bias",
                     save_path=os.path.join(paths["plots"], f"{label}_regimes_walkforward.png"))

        print(f"\n\n=== 3. Month-by-month breakdown for {label} ===")
        monthly_backtest_breakdown(prices, wf_features, wf_regimes, avoid_regime=0,
                                    label=label, save_path=paths["monthly_csv"])

        print(f"\n\n=== 4. Zooming in on Random Months for {label} ===")
        plot_random_months(prices, wf_regimes, k, label, num_months=3,
                            save_path=os.path.join(paths["plots"], f"{label}_regimes_monthly_zoom.png"))

        plot_recent_window(prices, wf_regimes, k, n_days=5, window_label="week",
                            ticker_label=label,
                            save_path=os.path.join(paths["plots"], f"{label}_regimes_latest_week.png"))
        plot_recent_window(prices, wf_regimes, k, n_days=21, window_label="month",
                            ticker_label=label,
                            save_path=os.path.join(paths["plots"], f"{label}_regimes_latest_month.png"))

        latest_regime_verdict(wf_regimes, wf_stats, label)

        print(f"\nRun finished: {datetime.now().isoformat(timespec='seconds')}")
        print(f"All outputs saved under: {paths['base']}/")
    finally:
        sys.stdout = old_stdout
        tee.close()

    print(f"Done. Report + plots + monthly CSV saved in: {paths['base']}/")


if __name__ == "__main__":
    stock_input = input("Enter stock name (or 'NSEI' for Nifty index): ")
    run(stock_input)