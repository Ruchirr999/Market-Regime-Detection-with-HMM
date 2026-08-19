"""
Market Regime Detection using Hidden Markov Models
----------------------------------------------------
Fits a Gaussian HMM on enriched features (return, slow vol, fast vol,
vol_ratio, drawdown) to infer hidden market regimes, then backtests a
regime-conditioned strategy against buy-and-hold.

Improvements over previous version (all three fixes for crash detection):

  FIX 1 — Richer features (build_features)
    Old: 2 features — return, 10-day rolling vol
    New: 5 features — return, vol_slow (10d), vol_fast (3d), vol_ratio
         (fast/slow — spike detector), drawdown (distance from 60d high)
    Why: The 10-day vol window reacted too slowly to crashes like COVID.
         vol_fast picks up a volatility spike within 2-3 days. vol_ratio
         fires when fast vol suddenly exceeds slow vol — the clearest
         statistical signature of a crash onset. drawdown separates
         "high vol crash" from "high vol recovery" — both have similar
         return/vol, but drawdown is large in one and recovering in the other.

  FIX 2 — Adaptive refitting (walk_forward_regimes)
    Old: refit HMM every fixed 21 trading days regardless of market conditions
    New: refit immediately whenever vol_ratio > VOL_SPIKE_THRESHOLD (default 2.0),
         i.e. when fast vol exceeds slow vol by 2x — the signature of a crash
         onset. Normal schedule (refit_every=21) applies in calm periods.
    Why: During a fast crash, waiting up to 21 days for the next scheduled
         refit means the model is using stale regime boundaries exactly when
         the market is moving fastest. Adaptive refitting updates the model
         within 1-2 days of a volatility spike.

  FIX 3 — Reduced min_train (run defaults)
    Old: min_train=500 (roughly 2 years of history before labeling starts)
    New: min_train=252 (1 trading year)
    Why: 500 days was too conservative. With 252 days the model starts
         labeling earlier, giving it more crash events in its labeled history
         to learn from before the next one arrives. 252 days is still enough
         for a stable HMM fit with 2-4 states.
"""

import os
import sys
import random
from datetime import datetime

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hmmlearn.hmm import GaussianHMM

# Suppress hmmlearn convergence warnings in the console — they are expected
# when the training window is short (early walk-forward steps) and do not
# affect correctness. The fit validation below catches genuinely bad fits.
warnings.filterwarnings("ignore", message=".*Model is not converging.*")


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MIN_VIABLE_TRAIN_ROWS   = 60    # below this an HMM fit is too noisy to trust
DEFAULT_MIN_TRAIN       = 252   # FIX 3: reduced from 500 to 1 trading year
MIN_TRAIN_FLOOR         = 100   # hard floor even for short-history tickers
MIN_WALK_FORWARD_TEST_DAYS = 60 # need ~3 months of labeled days to be meaningful
VOL_SPIKE_THRESHOLD     = 2.0   # FIX 2: vol_ratio above this triggers early refit
DEFAULT_MIN_DAYS_BEFORE_RESET = 252  # event-anchor cooldown to avoid reset thrashing


# --------------------------------------------------------------------------
# Output organisation
# --------------------------------------------------------------------------

class Tee:
    """Duplicates everything written to stdout into a log file as well,
    so the full console transcript ends up saved as a readable .txt report
    with zero changes to the print statements themselves."""

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
    the paths so every function writes to the same place."""
    base  = os.path.join(root, label)
    plots = os.path.join(base, "plots")
    os.makedirs(plots, exist_ok=True)
    return {
        "base":        base,
        "plots":       plots,
        "report_txt":  os.path.join(base, "report.txt"),
        "monthly_csv": os.path.join(base, "monthly_breakdown.csv"),
    }


# --------------------------------------------------------------------------
# Ticker handling
# --------------------------------------------------------------------------

def resolve_ticker(stock_input: str) -> tuple:
    """
    Turns user input into a yfinance ticker + a clean label for filenames.
      - "NSEI" / "NIFTY" / "NIFTY50" -> the index (^NSEI)
      - anything else                 -> NSE stock, gets ".NS" appended
                                         unless user already added a suffix

    Returns (ticker, label, fallback_ticker):
      ticker          — NSE candidate to try first  (e.g. "AAPL.NS")
      label           — clean string for filenames  (e.g. "AAPL")
      fallback_ticker — raw unmodified input        (e.g. "AAPL")
                        retried if the .NS candidate has no data, making
                        US tickers work without manual suffixes.
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
    """Maps a regime's annualised return to a human verdict.
    Thresholds are rough judgment calls, not statistically derived."""
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


def latest_regime_verdict(regimes: pd.Series, regime_stats: pd.DataFrame,
                           label: str):
    """Prints the current regime, consecutive-day streak, and verdict."""
    latest_date   = regimes.index[-1]
    latest_regime = regimes.iloc[-1]

    streak = 1
    for i in range(len(regimes) - 2, -1, -1):
        if regimes.iloc[i] == latest_regime:
            streak += 1
        else:
            break

    ann_return = regime_stats.loc[latest_regime, "ann_return_%"]
    verdict    = classify_bullish_bearish(ann_return)

    print(f"\n=== {label}: Current Verdict ===")
    print(f"As of {latest_date.date()}: Regime {latest_regime}, "
          f"in this regime for {streak} consecutive trading day(s)")
    print(f"Historical annualised return of this regime: {ann_return:.1f}%")
    print(f"Verdict: {verdict}")
    print("(Note: describes the current statistical regime, not a forecast.)")
    return verdict


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_random_months(prices: pd.Series, regimes: pd.Series, n_states: int,
                       label: str, num_months: int = 3, save_path: str = None):
    """Picks random months and plots them side-by-side for close inspection."""
    aligned = prices.loc[regimes.index].to_frame(name="price")
    aligned["regime"]     = regimes
    aligned["YearMonth"]  = aligned.index.to_period("M").astype(str)
    unique_months         = aligned["YearMonth"].unique().tolist()

    sampled_months = sorted(random.sample(unique_months,
                                          min(num_months, len(unique_months))))
    fig, axes = plt.subplots(1, num_months, figsize=(5 * num_months, 5))
    if num_months == 1:
        axes = [axes]

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_states))

    for ax, month in zip(axes, sampled_months):
        month_data = aligned[aligned["YearMonth"] == month]
        ax.plot(month_data.index, month_data["price"], color="black",
                alpha=0.3, linewidth=1,
                label="Price" if ax is axes[0] else None)
        for state in range(n_states):
            mask = month_data["regime"] == state
            if mask.any():
                ax.scatter(month_data.index[mask],
                           month_data.loc[mask, "price"],
                           c=[colors[state]], s=30,
                           label=f"Regime {state}", zorder=5)
        ax.set_title(f"{month}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Day of month")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=8, title="Legend")

    fig.suptitle(f"{label}: Regime Zoom-In on Random Months "
                 f"(state 0 = worst, state {n_states-1} = best)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved monthly zoom plots -> {save_path}")


def plot_recent_window(prices: pd.Series, regimes: pd.Series, n_states: int,
                       n_days: int, window_label: str, ticker_label: str,
                       save_path: str = None):
    """Plots the most recent n_days trading days colored by regime."""
    aligned           = prices.loc[regimes.index].to_frame(name="price")
    aligned["regime"] = regimes
    recent            = aligned.tail(n_days)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_states))

    ax.plot(recent.index, recent["price"], color="black", alpha=0.3,
            linewidth=1, label="Price")
    for state in range(n_states):
        mask = recent["regime"] == state
        if mask.any():
            ax.scatter(recent.index[mask], recent.loc[mask, "price"],
                       c=[colors[state]], s=40, label=f"Regime {state}",
                       zorder=5)

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
    colors  = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_states))
    for state in range(n_states):
        mask = regimes == state
        ax.scatter(aligned.index[mask], aligned[mask],
                   c=[colors[state]], s=6, label=f"Regime {state}")
    ax.plot(aligned.index, aligned.values, color="black", alpha=0.2,
            linewidth=0.8)
    ax.legend(loc="upper left", title="0 = worst regime, higher = better")
    ax.set_title(f"{ticker_label}: Price Colored by HMM Regime\n{subtitle}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved plot -> {save_path}")


# --------------------------------------------------------------------------
# FIX 1: Feature engineering — 5 features instead of 2
# --------------------------------------------------------------------------

"""
Feature improvements for regime_hmm.py
----------------------------------------
Drop these into build_features() in regime_hmm.py.
All features are:
  - Computable from daily close prices only (no external data)
  - Causal (use only past data, safe for walk-forward)
  - Proven to carry regime information in the literature

WHY each feature helps:
  - vol_ratio:    fast/slow vol ratio catches vol regime shifts BEFORE
                  the return series shows them. Vol spikes precede crashes.
  - drawdown:     how far below the recent peak we are. Regime 0 during a
                  5% dip is very different from regime 0 during a 40% crash.
  - momentum:     price relative to its 20-day MA. HMM can now tell apart
                  "low vol uptrend" from "low vol downtrend" which look
                  identical in return/vol space alone.
  - trend_strength: ADX-like measure. High = strong trend (up or down).
                  Helps separate trending from sideways regimes.
  - vol_of_vol:   volatility of volatility. High vov = unstable, transitioning
                  market. Low vov = stable regime. Directly models regime
                  persistence.
"""




def build_features(prices: pd.Series,
                   fast_vol: int = 5,
                   slow_vol: int = 21,
                   drawdown_window: int = 60,
                   momentum_window: int = 20,
                   vov_window: int = 10) -> pd.DataFrame:
    """
    Builds the feature matrix the HMM is fit on.

    Features:
      return        : daily log return
      vol_slow      : slow rolling volatility (baseline, was 'volatility')
      vol_ratio     : fast_vol / slow_vol -- spikes before crashes
      drawdown      : how far below rolling max (0 = at all-time high,
                      -0.3 = 30% below peak)
      momentum      : (price / 20d MA) - 1, positive = above MA = bullish
      vol_of_vol    : rolling std of vol_slow -- how unstable is volatility

    Parameters
    ----------
    fast_vol        : window for fast volatility (default 5 days = 1 week)
    slow_vol        : window for slow volatility (default 21 days = 1 month)
    drawdown_window : lookback for rolling max to compute drawdown
    momentum_window : MA window for momentum feature
    vov_window      : window for vol-of-vol
    """
    df = pd.DataFrame(index=prices.index)

    # --- Core return ---
    df["return"] = np.log(prices / prices.shift(1))

    # --- Volatility (slow, replaces the original single 'volatility') ---
    df["vol_slow"] = df["return"].rolling(slow_vol).std()

    # --- Vol ratio: fast / slow ---
    vol_fast = df["return"].rolling(fast_vol).std()
    df["vol_ratio"] = vol_fast / df["vol_slow"].replace(0, np.nan)

    # --- Drawdown: how far below rolling max ---
    rolling_max = prices.rolling(drawdown_window, min_periods=1).max()
    df["drawdown"] = (prices / rolling_max) - 1.0   # always <= 0

    # --- Momentum: distance from moving average ---
    ma = prices.rolling(momentum_window).mean()
    df["momentum"] = (prices / ma) - 1.0            # positive = above MA

    # --- Vol of vol: stability of the volatility regime ---
    df["vol_of_vol"] = df["vol_slow"].rolling(vov_window).std()

    df = df.dropna()
    return df


# --------------------------------------------------------------------------
# Limitations
# --------------------------------------------------------------------------
"""
KNOWN LIMITATIONS:

1. vol_ratio can be NaN or infinite if vol_slow goes to zero (flat price
   period). Already handled with .replace(0, np.nan) -> dropna() removes
   those rows. For very recently listed or very thinly traded stocks this
   can eat significant history.

2. momentum and drawdown both depend on the price level, not returns.
   This is intentional (they carry regime info the return series misses)
   but means the HMM's Gaussian assumption fits them less cleanly than
   the return/vol pair. If you see regime labels that look nonsensical,
   try removing momentum first and see if it cleans up.

3. vol_of_vol has two rolling windows stacked (vol_slow then vov_window),
   so it needs slow_vol + vov_window days before producing a value.
   With defaults that's 31 days -- not a problem in practice but worth
   knowing if you're using a short history.

4. Adding 6 features instead of 2 makes the HMM harder to fit reliably
   with small training windows. The existing MIN_VIABLE_TRAIN_ROWS = 60
   should be raised to at least 120 when using this full feature set.
   Alternatively use a subset: (return, vol_slow, vol_ratio, drawdown)
   is a good 4-feature compromise.
"""


# --------------------------------------------------------------------------
# Model selection / labeling
# --------------------------------------------------------------------------

def select_n_states(features: np.ndarray, max_states: int = 4,
                    n_iter: int = 200):
    """
    Fits HMMs for k=2..max_states and selects the best by BIC.
    Lower BIC = better fit penalised for complexity.
    max_states defaults to 4 (not 5) because with 5 features the parameter
    count per state is larger, so BIC penalises complexity more — k=4 is
    almost always selected anyway and k=5 rarely adds interpretable value.
    """
    results = []
    for k in range(2, max_states + 1):
        model = GaussianHMM(n_components=k, covariance_type="diag",
                            n_iter=n_iter, random_state=42)
        model.fit(features)
        log_likelihood = model.score(features)
        # n_params: transition matrix (k*(k-1)) + means and variances per state
        n_params = k * (k - 1) + k * features.shape[1] * 2
        bic = -2 * log_likelihood + n_params * np.log(len(features))
        results.append((k, bic, model))
        print(f"  k={k}: log-likelihood={log_likelihood:.1f}, BIC={bic:.1f}")
    best = min(results, key=lambda x: x[1])
    print(f"-> Selected k={best[0]} states by BIC")
    return best[2], best[0]


def label_regimes(model: GaussianHMM, features: pd.DataFrame) -> pd.Series:
    """
    Runs Viterbi to get the most likely state sequence, then relabels
    states 0..k-1 by mean return so state 0 = worst, k-1 = best.
    This makes labels human-interpretable and comparable across tickers.
    """
    hidden_states = model.predict(features.values)
    state_returns = [features["return"][hidden_states == s].mean()
                     for s in range(model.n_components)]
    order = np.argsort(state_returns)
    remap = {old: new for new, old in enumerate(order)}
    return pd.Series([remap[s] for s in hidden_states], index=features.index)


def describe_regimes(features: pd.DataFrame, regimes: pd.Series):
    """Prints per-regime mean return / vol for a sanity check."""
    if len(regimes) == 0:
        raise ValueError(
            "describe_regimes: regimes is empty — walk_forward_regimes() "
            "produced no labeled days. Check the NOTE/error printed above."
        )
    # Use vol_slow as the representative volatility column
    vol_col = "vol_slow" if "vol_slow" in features.columns else features.columns[1]
    summary = features.groupby(regimes).agg(
        mean_return=("return", "mean"),
        volatility=(vol_col, "mean"),
        days=("return", "count"),
    )
    summary["ann_return_%"] = summary["mean_return"] * 252 * 100
    print(summary)
    return summary


# --------------------------------------------------------------------------
# FIX 2: Adaptive walk-forward with vol-spike-triggered early refit
# --------------------------------------------------------------------------

def transition_detector(previous_regime: int, current_regime: int,
                        n_states: int, major_jump: int = 2):
    """
    Detect major regime transitions that should reset the walk-forward window.

    Major transitions are:
      - Entering crash regime: non-crash -> 0
      - Sharp recovery from crash: 0 -> top regimes
      - Any large regime jump (abs delta >= major_jump)
    """
    if previous_regime is None or current_regime is None:
        return False, ""
    if previous_regime == current_regime:
        return False, ""

    if current_regime == 0 and previous_regime > 0:
        return True, f"entered crash regime ({previous_regime} -> 0)"

    recovery_floor = max(1, n_states - 2)
    if previous_regime == 0 and current_regime >= recovery_floor:
        return True, f"sharp recovery from crash ({previous_regime} -> {current_regime})"

    if abs(current_regime - previous_regime) >= major_jump:
        return True, f"major regime jump ({previous_regime} -> {current_regime})"

    return False, ""


def walk_forward_regimes(features: pd.DataFrame, n_states: int,
                          min_train: int = DEFAULT_MIN_TRAIN,
                          refit_every: int = 21,
                          n_iter: int = 100,
                          vol_spike_threshold: float = VOL_SPIKE_THRESHOLD,
                          min_days_before_reset: int = DEFAULT_MIN_DAYS_BEFORE_RESET
                          ) -> pd.Series:
    """
    Causal walk-forward HMM labeling with adaptive refitting.

    Standard behaviour (inherited from previous version):
      - Refit the HMM every `refit_every` trading days.
      - Validate each fit before accepting (checks for NaN params).
      - If a fit fails, keep the previous valid model.

    New (FIX 2) — adaptive early refit on volatility spikes:
      - After each day, check the vol_ratio feature value.
      - If vol_ratio > vol_spike_threshold, trigger an immediate refit
        regardless of how recently the last refit happened.
      - This means during a crash onset (e.g. COVID) the model updates
        its regime boundaries within 1-2 days instead of waiting up to
        21 days for the next scheduled refit.
      - vol_ratio column must exist in features (built by build_features).
        If it doesn't (e.g. custom feature set), adaptive refitting is
        skipped silently and the fixed schedule applies.

    New (event-anchored walk-forward):
      - Track major transitions between mapped regimes (crash entry/recovery).
      - On major transition, reset the training start index to that day.
      - Future fits use only data from this reset point onward.
      - `min_days_before_reset` enforces cooldown between resets.

    Parameters
    ----------
    features              : DataFrame from build_features()
    n_states              : number of HMM hidden states (from select_n_states)
    min_train             : trading days of history before labeling starts
                            (FIX 3: default reduced to 252 from 500)
    refit_every           : normal refit interval in trading days (default 21)
    n_iter                : HMM EM iterations per fit (default 100)
    vol_spike_threshold   : vol_ratio value above which early refit fires
                            (default 2.0 — fast vol > 2x slow vol)
    min_days_before_reset : minimum days between event-anchored resets
                            to avoid frequent whipsaw resets (default 252)
    """
    values = features.values
    n      = len(values)

    has_vol_ratio = "vol_ratio" in features.columns
    if has_vol_ratio:
        vol_ratio_idx = features.columns.get_loc("vol_ratio")
    else:
        vol_ratio_idx = None
        print("  NOTE: vol_ratio not found in features — adaptive refitting "
              "disabled, using fixed schedule only.")

    if n < MIN_VIABLE_TRAIN_ROWS + 5:
        raise ValueError(
            f"walk_forward_regimes: only {n} feature rows available — "
            f"too little history for a meaningful HMM fit "
            f"(need at least {MIN_VIABLE_TRAIN_ROWS + 5} trading days). "
            f"Try a longer date range or a different ticker."
        )

    if min_train >= n:
        adjusted = max(MIN_VIABLE_TRAIN_ROWS, int(n * 0.6))
        adjusted = min(adjusted, n - 5)
        print(f"NOTE: min_train={min_train} but only {n} feature rows "
              f"available. Shrinking min_train to {adjusted}.")
        min_train = adjusted

    regimes    = pd.Series(index=features.index, dtype=float)
    model      = None
    last_refit = -1
    remap      = None
    return_col = features.columns.get_loc("return")

    early_refits = 0  # track how many vol-spike refits fired (for reporting)
    reset_count = 0
    blocked_resets = 0
    train_start_idx = 0
    last_reset_idx = 0
    previous_regime = None

    for t in range(min_train, n):

        if model is not None and remap is not None:
            try:
                cycle_window = values[train_start_idx:t + 1]
                cycle_state_seq = model.predict(cycle_window)
                current_regime = remap[cycle_state_seq[-1]]
            except Exception:
                current_regime = None

            major_shift, shift_reason = transition_detector(
                previous_regime, current_regime, n_states
            )
            if major_shift:
                days_since_reset = t - last_reset_idx
                if days_since_reset >= min_days_before_reset:
                    reset_count += 1
                    train_start_idx = t
                    last_reset_idx = t
                    model = None
                    remap = None
                    last_refit = -1
                    previous_regime = None
                    print(f"  [walk_forward] Reset #{reset_count} at {features.index[t]} "
                          f"due to {shift_reason}. "
                          f"New train window starts at index {t}.")
                    continue
                else:
                    blocked_resets += 1
                    print(f"  [walk_forward] Transition detected at {features.index[t]} "
                          f"({shift_reason}) but reset skipped "
                          f"(cooldown: {days_since_reset}/{min_days_before_reset} days).")

        # ---- Decide whether to refit ----
        vol_spike = (
            has_vol_ratio and
            vol_ratio_idx is not None and
            np.isfinite(values[t, vol_ratio_idx]) and
            values[t, vol_ratio_idx] > vol_spike_threshold
        )
        scheduled = (model is None or (t - last_refit) >= refit_every)

        if scheduled or vol_spike:
            if vol_spike and not scheduled:
                early_refits += 1

            train_data = values[train_start_idx:t]
            if len(train_data) < MIN_VIABLE_TRAIN_ROWS:
                continue
            candidate  = GaussianHMM(n_components=n_states,
                                     covariance_type="diag",
                                     n_iter=n_iter, random_state=42)
            fit_ok = False
            try:
                candidate.fit(train_data)
                if (np.isfinite(candidate.startprob_).all() and
                        np.isfinite(candidate.transmat_).all() and
                        np.isfinite(candidate.means_).all()):
                    fit_ok = True
            except Exception as e:
                print(f"  [walk_forward] HMM fit failed at t={t}: {e} "
                      f"— keeping previous model.")

            if fit_ok:
                model      = candidate
                last_refit = t

                # Rebuild remap: sort raw states by mean return
                train_states = model.predict(train_data)
                state_means  = [
                    train_data[train_states == s, return_col].mean()
                    if (train_states == s).any() else 0.0
                    for s in range(n_states)
                ]
                order = np.argsort(state_means)
                remap = {old: new for new, old in enumerate(order)}

        if model is None or remap is None:
            continue

        try:
            window    = values[train_start_idx:t + 1]
            state_seq = model.predict(window)
            mapped_state = remap[state_seq[-1]]
            regimes.iloc[t] = mapped_state
            previous_regime = mapped_state
        except Exception:
            pass  # leave NaN, dropped below

    regimes = regimes.dropna().astype(int)
    print(f"  Walk-forward complete: {len(regimes)} days labeled, "
          f"{early_refits} early refit(s) triggered by vol spikes "
          f"(threshold: vol_ratio > {vol_spike_threshold}), "
          f"{reset_count} event-anchored reset(s), "
          f"{blocked_resets} blocked by cooldown")
    return regimes


# --------------------------------------------------------------------------
# Backtesting
# --------------------------------------------------------------------------

def _sharpe(r: pd.Series) -> float:
    return (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan


def backtest_regime_strategy(prices: pd.Series, features: pd.DataFrame,
                              regimes: pd.Series, avoid_regime: int = 0):
    """
    Toy strategy: hold unless in worst regime (avoid_regime=0), then cash.
    Compared against buy-and-hold. NOT investment advice — tests whether
    regime labels carry any signal at all.
    """
    if len(features) == 0:
        raise ValueError(
            "backtest_regime_strategy: features is empty — walk_forward produced "
            "no labeled days. Check the NOTE/error printed above."
        )
    aligned_prices = prices.loc[features.index]
    daily_ret      = aligned_prices.pct_change().fillna(0)
    position       = (regimes != avoid_regime).astype(int)
    strategy_ret   = daily_ret * position.shift(1).fillna(0)
    bh_equity      = (1 + daily_ret).cumprod()
    strat_equity   = (1 + strategy_ret).cumprod()

    print(f"Buy & Hold   -> total return: {(bh_equity.iloc[-1]-1)*100:.1f}%, "
          f"Sharpe: {_sharpe(daily_ret):.2f}")
    print(f"Regime Strat -> total return: {(strat_equity.iloc[-1]-1)*100:.1f}%, "
          f"Sharpe: {_sharpe(strategy_ret):.2f}")

    return bh_equity, strat_equity, daily_ret, strategy_ret


def monthly_backtest_breakdown(prices: pd.Series, features: pd.DataFrame,
                                regimes: pd.Series, avoid_regime: int = 0,
                                label: str = "",
                                save_path: str = None) -> pd.DataFrame:
    """
    Month-by-month backtest: buy&hold return, strategy return, difference,
    rough monthly Sharpe for each, days in cash, dominant regime.
    Answers "did avoiding the worst regime help in March 2020 specifically,
    or only on average?" — a single all-time number hides this detail.
    """
    aligned_prices = prices.loc[features.index]
    daily_ret      = aligned_prices.pct_change().fillna(0)
    position       = (regimes != avoid_regime).astype(int)
    strategy_ret   = daily_ret * position.shift(1).fillna(0)

    df = pd.DataFrame({
        "bh_ret":    daily_ret,
        "strat_ret": strategy_ret,
        "regime":    regimes,
        "in_cash":   (position == 0).astype(int),
    })
    df["YearMonth"] = df.index.to_period("M")

    rows = []
    for period, g in df.groupby("YearMonth"):
        bh_m    = (1 + g["bh_ret"]).prod() - 1
        strat_m = (1 + g["strat_ret"]).prod() - 1
        rows.append({
            "month":              str(period),
            "trading_days":       len(g),
            "buy_hold_return_%":  bh_m * 100,
            "strategy_return_%":  strat_m * 100,
            "difference_%":       (strat_m - bh_m) * 100,
            "buy_hold_sharpe":    _sharpe(g["bh_ret"]),
            "strategy_sharpe":    _sharpe(g["strat_ret"]),
            "days_in_cash":       int(g["in_cash"].sum()),
            "dominant_regime":    int(g["regime"].mode().iloc[0]),
        })

    monthly = pd.DataFrame(rows).set_index("month")

    print(f"\n=== {label}: Month-by-Month Backtest Breakdown "
          f"({len(monthly)} months) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(monthly.round(2))

    won = (monthly["difference_%"] > 0).sum()
    print(f"\nStrategy beat buy&hold in {won}/{len(monthly)} months "
          f"({won/len(monthly)*100:.0f}%).")
    print(f"Average monthly outperformance: "
          f"{monthly['difference_%'].mean():.2f} pp "
          f"(std: {monthly['difference_%'].std():.2f}).")
    print("(Monthly Sharpe is computed on ~21 daily returns — treat as "
          "directional signal, not a precise estimate.)")

    if save_path:
        monthly.to_csv(save_path)
        print(f"Saved monthly breakdown -> {save_path}")

    return monthly


# --------------------------------------------------------------------------
# Price loading
# --------------------------------------------------------------------------

def _download_prices(ticker: str) -> pd.Series:
    """Downloads close prices via yfinance. Returns empty Series (not an
    exception) if the ticker is unknown or delisted — callers check emptiness."""
    import yfinance as yf
    raw    = yf.download(ticker, start="2015-01-01", progress=False)["Close"]
    prices = raw.dropna()
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    return prices


def load_prices_with_fallback(ticker: str,
                               fallback_ticker: str) -> tuple:
    """
    Tries `ticker` first (NSE candidate, e.g. "AAPL.NS").
    If that returns no data, retries with `fallback_ticker` (raw input,
    e.g. "AAPL") so US and other-exchange tickers work without manual suffixes.
    Raises a clear ValueError if neither candidate has data.

    Returns (prices, ticker_actually_used).
    """
    prices = _download_prices(ticker)
    if not prices.empty:
        return prices, ticker

    if fallback_ticker != ticker:
        print(f"NOTE: no data for '{ticker}' — retrying as '{fallback_ticker}' "
              f"(not an NSE-listed ticker).")
        prices = _download_prices(fallback_ticker)
        if not prices.empty:
            return prices, fallback_ticker

    tried = (ticker if fallback_ticker == ticker
             else f"{ticker} and {fallback_ticker}")
    raise ValueError(
        f"No price data found for '{ticker}' (tried: {tried}). "
        f"For NSE stocks use the plain name (e.g. 'RELIANCE') or '.NS' suffix. "
        f"For US stocks use the plain ticker (e.g. 'AAPL')."
    )


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run(stock_input: str, data_loader=None,
        min_train: int = None, refit_every: int = 21,
        vol_spike_threshold: float = VOL_SPIKE_THRESHOLD,
        min_days_before_reset: int = DEFAULT_MIN_DAYS_BEFORE_RESET):
    """
    Full pipeline: download → features → model selection → walk-forward
    → backtest → plots → verdict → save report.

    Parameters
    ----------
    stock_input         : ticker string (e.g. "RELIANCE", "NSEI", "AAPL")
    data_loader         : optional callable(ticker) -> pd.Series of close prices.
                          Defaults to yfinance. Inject for testing or custom sources.
    min_train           : walk-forward initial training window (trading days).
                          None = auto-scale: uses DEFAULT_MIN_TRAIN (252) if the
                          ticker has enough history, otherwise shrinks to
                          MIN_TRAIN_FLOOR (100).
    refit_every         : normal refit interval in trading days (default 21).
    vol_spike_threshold : vol_ratio above this triggers an immediate early refit
                          (default 2.0). Increase to make early refits rarer,
                          decrease to make them more aggressive.
    min_days_before_reset : minimum days between event-anchored reset events
                            (default 252).
    """
    ticker, label, fallback_ticker = resolve_ticker(stock_input)
    paths = make_output_dirs(label)

    tee        = Tee(paths["report_txt"])
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Ticker candidate: {ticker}  |  Label: {label}")
        print(f"Config: min_train={min_train or DEFAULT_MIN_TRAIN}, "
              f"refit_every={refit_every}, "
              f"vol_spike_threshold={vol_spike_threshold}, "
              f"min_days_before_reset={min_days_before_reset}")

        # ---- Load prices ----
        if data_loader is None:
            prices, ticker_used = load_prices_with_fallback(ticker, fallback_ticker)
            print(f"Ticker used: {ticker_used}  |  {len(prices)} trading days loaded")
        else:
            prices = data_loader(ticker)

        if prices.empty:
            raise ValueError(
                f"No price data available for '{stock_input}' — cannot continue."
            )

        # ---- Feature engineering (FIX 1: 5 features) ----
        features = build_features(prices)
        print(f"Features built: {list(features.columns)} | {len(features)} rows")

        # ---- Global fit (lookahead bias, for comparison only) ----
        print(f"\n=== 1. Global fit for {label} "
              f"(lookahead bias — for comparison only) ===")
        global_model, k = select_n_states(features.values, max_states=4)
        global_regimes  = label_regimes(global_model, features)
        describe_regimes(features, global_regimes)
        print("\nBacktest (has lookahead bias — inflated result expected):")
        backtest_regime_strategy(prices, features, global_regimes, avoid_regime=0)
        plot_regimes(
            prices, global_regimes, k, label,
            subtitle="Global fit — has lookahead bias, for comparison only",
            save_path=os.path.join(paths["plots"],
                                   f"{label}_regimes_lookahead.png"),
        )

        # ---- Walk-forward fit (FIX 2 + FIX 3: adaptive refit, min_train=252) ----
        print(f"\n\n=== 2. Walk-forward causal fit for {label} "
              f"(honest version — no lookahead) ===")
        requested_min_train = DEFAULT_MIN_TRAIN if min_train is None else min_train
        wf_regimes = walk_forward_regimes(
            features, n_states=k,
            min_train=requested_min_train,
            refit_every=refit_every,
            vol_spike_threshold=vol_spike_threshold,
            min_days_before_reset=min_days_before_reset,
        )

        wf_features = features.loc[wf_regimes.index]
        held_out    = len(features) - len(wf_regimes)
        print(f"Labeled {len(wf_regimes)} days "
              f"(first {held_out} held out as initial training window)")
        wf_stats = describe_regimes(wf_features, wf_regimes)

        print("\nBacktest (causal, no lookahead):")
        backtest_regime_strategy(prices, wf_features, wf_regimes, avoid_regime=0)
        plot_regimes(
            prices, wf_regimes, k, label,
            subtitle="Walk-forward causal fit — no lookahead bias",
            save_path=os.path.join(paths["plots"],
                                   f"{label}_regimes_walkforward.png"),
        )

        # ---- Month-by-month breakdown ----
        print(f"\n\n=== 3. Month-by-month breakdown for {label} ===")
        monthly_backtest_breakdown(
            prices, wf_features, wf_regimes, avoid_regime=0,
            label=label, save_path=paths["monthly_csv"],
        )

        # ---- Zoom plots ----
        print(f"\n\n=== 4. Zoom plots for {label} ===")
        plot_random_months(
            prices, wf_regimes, k, label, num_months=3,
            save_path=os.path.join(paths["plots"],
                                   f"{label}_regimes_monthly_zoom.png"),
        )
        plot_recent_window(
            prices, wf_regimes, k, n_days=5, window_label="week",
            ticker_label=label,
            save_path=os.path.join(paths["plots"],
                                   f"{label}_regimes_latest_week.png"),
        )
        plot_recent_window(
            prices, wf_regimes, k, n_days=21, window_label="month",
            ticker_label=label,
            save_path=os.path.join(paths["plots"],
                                   f"{label}_regimes_latest_month.png"),
        )

        # ---- Current verdict ----
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