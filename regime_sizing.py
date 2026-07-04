"""
Regime-Conditioned Position Sizing
------------------------------------
Replaces the binary invested/cash signal with a continuous position size
derived from the HMM's posterior regime probabilities.

Instead of:
    position = 1 if regime != 0 else 0

We compute:
    position = sum of (regime_weight * P(regime_i | data up to t))

where regime_weight is how "good" that regime is (based on its historical
mean return), so the position scales smoothly between 0 and 1 depending
on how confidently the model thinks you're in a good vs bad regime.

Drop-in additions to regime_hmm.py:
  - walk_forward_proba()       -> like walk_forward_regimes() but returns
                                  a DataFrame of per-regime probabilities
  - regime_weights()           -> maps regime index to a [0,1] weight
  - compute_position_size()    -> turns proba DataFrame into a position Series
  - backtest_sized_strategy()  -> backtests the sized strategy vs binary vs BH
                                  NOW WITH: Sharpe, Sortino, CAGR, Max Drawdown,
                                  Calmar, monthly win-rate for all three strategies
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

MIN_VIABLE_TRAIN_ROWS = 60


# --------------------------------------------------------------------------
# Metric helpers  (Part 1 addition)
# --------------------------------------------------------------------------

def _max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown as a negative fraction."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def _cagr(equity: pd.Series, trading_days: int = 252) -> float:
    """Compound Annual Growth Rate. equity should start at 1.0."""
    n = len(equity)
    if n < 2:
        return np.nan
    years = n / trading_days
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def _sharpe(r: pd.Series, trading_days: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    std = r.std()
    return float((r.mean() / std) * np.sqrt(trading_days)) if std > 0 else np.nan


def _sortino(r: pd.Series, trading_days: int = 252) -> float:
    """
    Sortino ratio: penalises only downside volatility.
    Better than Sharpe for strategies that cut losses asymmetrically.
    """
    downside = r[r < 0]
    ds = downside.std()
    if ds == 0 or np.isnan(ds):
        return np.nan
    return float((r.mean() / ds) * np.sqrt(trading_days))


def _calmar(r: pd.Series) -> float:
    """CAGR / |max drawdown|. Higher is better."""
    equity = (1 + r).cumprod()
    mdd = _max_drawdown(equity)
    ann = _cagr(equity)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(ann / abs(mdd))


def _monthly_win(r: pd.Series) -> float:
    """Fraction of calendar months with positive return."""
    monthly = (1 + r).resample("ME").prod() - 1
    return float((monthly > 0).mean()) if len(monthly) > 0 else np.nan


def _metrics_block(r: pd.Series, pos: pd.Series) -> dict:
    """Returns all six metrics for one return series as a dict."""
    equity = (1 + r).cumprod()
    return {
        "total_%":    round((equity.iloc[-1] - 1) * 100, 2),
        "cagr_%":     round(_cagr(equity) * 100, 2),
        "sharpe":     round(_sharpe(r), 3),
        "sortino":    round(_sortino(r), 3),
        "calmar":     round(_calmar(r), 3),
        "max_dd_%":   round(_max_drawdown(equity) * 100, 2),
        "win_rate_%": round(_monthly_win(r) * 100, 1),
        "avg_pos_%":  round(pos.mean() * 100, 1),
    }


# --------------------------------------------------------------------------
# 1. Walk-forward with probabilities instead of hard labels
# --------------------------------------------------------------------------

def walk_forward_proba(features: pd.DataFrame, n_states: int,
                        min_train: int = 500, refit_every: int = 21,
                        n_iter: int = 100) -> pd.DataFrame:
    """
    Causal walk-forward fit that returns a DataFrame of shape
    (n_labeled_days, n_states) where each row is the posterior probability
    vector P(regime_0 | data), P(regime_1 | data), ..., P(regime_k | data)
    for that day.

    Columns are named by their *relabeled* regime index (0 = worst return,
    k-1 = best return) so the numbers are human-interpretable across refits.

    This is the key upgrade over walk_forward_regimes() which only returned
    the hard argmax label -- here you get the full probability distribution,
    which carries much more information for position sizing.
    """
    values = features.values
    n = len(values)

    if n < MIN_VIABLE_TRAIN_ROWS + 5:
        raise ValueError(
            f"walk_forward_proba: only {n} feature rows available -- too few "
            f"for a meaningful HMM fit. Need at least {MIN_VIABLE_TRAIN_ROWS + 5}."
        )

    if min_train >= n:
        adjusted = max(MIN_VIABLE_TRAIN_ROWS, int(n * 0.6))
        adjusted = min(adjusted, n - 5)
        print(f"NOTE: shrinking min_train from {min_train} to {adjusted} "
              f"(only {n} feature rows available).")
        min_train = adjusted

    # Pre-fill with NaN; rows before min_train stay NaN and are dropped later
    proba_df = pd.DataFrame(
        np.nan,
        index=features.index,
        columns=[f"p_regime_{i}" for i in range(n_states)]
    )

    model = None
    last_refit = -1
    remap = None          # maps raw HMM state index -> sorted-by-return index

    for t in range(min_train, n):

        # ---- refit on schedule ----
        if model is None or (t - last_refit) >= refit_every:
            train_data = values[:t]
            candidate = GaussianHMM(n_components=n_states,
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
                print(f"  [walk_forward_proba] fit failed at t={t}: {e} -- "
                      f"keeping previous model.")

            if fit_ok:
                model = candidate
                last_refit = t

                # Build the remap: sort states by mean return so 0 = worst
                train_states = model.predict(train_data)
                return_col = features.columns.get_loc("return")
                state_means = [
                    train_data[train_states == s, return_col].mean()
                    if (train_states == s).any() else 0.0
                    for s in range(n_states)
                ]
                order = np.argsort(state_means)
                remap = {old: new for new, old in enumerate(order)}

        if model is None or remap is None:
            continue

        # ---- posterior probabilities for today ----
        try:
            window = values[:t + 1]
            # predict_proba returns shape (T, n_states) -- take last row
            raw_proba = model.predict_proba(window)[-1]   # shape (n_states,)

            # Reorder columns so index 0 = worst regime, k-1 = best
            reordered = np.zeros(n_states)
            for raw_state, new_state in remap.items():
                reordered[new_state] = raw_proba[raw_state]

            proba_df.iloc[t] = reordered

        except Exception:
            pass   # leave NaN for this day, dropped below

    proba_df = proba_df.dropna()
    return proba_df


# --------------------------------------------------------------------------
# 2. Turn probabilities into a position size
# --------------------------------------------------------------------------

def regime_weights(n_states: int, shape: str = "linear") -> np.ndarray:
    """
    Maps regime index (0 = worst, k-1 = best) to a weight in [0, 1].

    shape="linear"      -> [0, 0.33, 0.67, 1.0] for k=4
        Simplest. Each step up in regime adds equal weight.

    shape="convex"      -> weights curve upward (be more cautious)
        Penalizes bad regimes more heavily; good for risk-averse sizing.

    shape="concave"     -> weights curve downward (be more aggressive)
        Ramps up exposure faster as regime improves.

    These are deliberate design choices, not magic numbers -- you should
    backtest all three and pick the one that matches your risk appetite.
    """
    indices = np.arange(n_states)
    if shape == "linear":
        return indices / (n_states - 1)
    elif shape == "convex":
        return (indices / (n_states - 1)) ** 2
    elif shape == "concave":
        return (indices / (n_states - 1)) ** 0.5
    else:
        raise ValueError(f"Unknown shape '{shape}'. Use 'linear', 'convex', or 'concave'.")


def compute_position_size(proba_df: pd.DataFrame,
                           shape: str = "linear",
                           min_position: float = 0.0,
                           max_position: float = 1.0) -> pd.Series:
    """
    Converts the probability DataFrame into a scalar position size for each
    day, in [min_position, max_position].

    position_t = sum_i( P(regime_i | data_t) * weight_i )

    Interpretation:
      - If the model is 100% confident you're in regime 0 (worst):
            position = min_position   (your floor -- never below this)
      - If 100% confident in regime k-1 (best):
            position = max_position   (fully invested)
      - If uncertain (e.g. 33%/33%/33% across 3 regimes):
            position = somewhere in the middle

    min_position: floor -- never go below this exposure even in worst regime.
        Useful if you don't want to ever be fully in cash (e.g. set to 0.2
        to always keep at least 20% invested).
    max_position: ceiling -- never exceed this exposure even in best regime.
        Useful for leverage limits (keep at 1.0 for no leverage).
    """
    n_states = proba_df.shape[1]
    weights = regime_weights(n_states, shape=shape)    # shape (n_states,)

    # Dot product of each day's probability vector with the weight vector
    raw_position = proba_df.values @ weights           # shape (n_days,)

    # Clip to [min_position, max_position]
    clipped = np.clip(raw_position, min_position, max_position)

    position = pd.Series(clipped, index=proba_df.index, name="position_size")

    # --- Worst-regime exposure report ---
    hard_regime = proba_df.idxmax(axis=1).str.extract(r"(\d+)$")[0].astype(int)
    worst_days_mask = (hard_regime == 0)
    n_worst = worst_days_mask.sum()

    if n_worst > 0:
        avg_exposure_worst = position[worst_days_mask].mean()
        min_exposure_worst = position[worst_days_mask].min()
        print(f"\n=== Worst-Regime Exposure ===")
        print(f"Days classified as worst regime (regime 0): {n_worst} "
              f"({n_worst / len(position) * 100:.1f}% of labeled history)")
        print(f"Your exposure on those days:")
        print(f"  Average : {avg_exposure_worst * 100:.1f}%  "
              f"(min_position floor = {min_position * 100:.0f}%)")
        print(f"  Minimum : {min_exposure_worst * 100:.1f}%  "
              f"(this is always >= min_position)")
        print(f"  Meaning : even in the worst detected regime, you were never "
              f"less than {min_exposure_worst * 100:.1f}% invested.")
    else:
        print("\n=== Worst-Regime Exposure ===")
        print("No days were classified as worst regime (regime 0) in the "
              "labeled history.")

    return position


# --------------------------------------------------------------------------
# 3. Backtest the sized strategy  (Part 1 addition: full metrics table)
# --------------------------------------------------------------------------

def backtest_sized_strategy(prices: pd.Series,
                             proba_df: pd.DataFrame,
                             shape: str = "linear",
                             min_position: float = 0.0,
                             avoid_regime: int = 0) -> pd.DataFrame:
    """
    Compares three strategies on the same history:

    1. Buy & Hold          -- always 100% invested, baseline
    2. Binary (original)   -- 100% invested unless in regime 0, then cash
    3. Sized (new)         -- position = P(good regimes) weighted sum,
                              continuous between 0 and 1

    Now prints a full metrics table for all three strategies:
      Total Return, CAGR, Sharpe, Sortino, Calmar, Max Drawdown,
      Monthly Win Rate, Avg Position Size

    Returns a DataFrame with daily returns and cumulative equity for all
    three, so you can plot or summarize however you like.
    """
    aligned_prices = prices.loc[proba_df.index]
    daily_ret = aligned_prices.pct_change().fillna(0)

    # Hard regime label = argmax of probability vector each day
    hard_regime = proba_df.idxmax(axis=1).str.extract(r"(\d+)$")[0].astype(int)

    # --- Binary position (original behavior) ---
    binary_pos = (hard_regime != avoid_regime).astype(float)
    binary_ret = daily_ret * binary_pos.shift(1).fillna(0)

    # --- Sized position (new) ---
    sized_pos = compute_position_size(proba_df, shape=shape,
                                       min_position=min_position)
    sized_ret = daily_ret * sized_pos.shift(1).fillna(0)

    # --- Assemble results DataFrame ---
    results = pd.DataFrame({
        "bh_return":        daily_ret,
        "binary_return":    binary_ret,
        "sized_return":     sized_ret,
        "binary_position":  binary_pos,
        "sized_position":   sized_pos,
        "bh_equity":        (1 + daily_ret).cumprod(),
        "binary_equity":    (1 + binary_ret).cumprod(),
        "sized_equity":     (1 + sized_ret).cumprod(),
    })

    # ---- Part 1: Full metrics table ----
    bh_m      = _metrics_block(daily_ret, pd.Series(1.0, index=daily_ret.index))
    binary_m  = _metrics_block(binary_ret, binary_pos)
    sized_m   = _metrics_block(sized_ret, sized_pos)

    sized_label = f"Sized ({shape}, floor={min_position*100:.0f}%)"

    header = (f"\n{'='*80}\n"
              f"{'Strategy Comparison':^80}\n"
              f"{'='*80}")
    print(header)

    col_w = 18
    metrics = ["total_%", "cagr_%", "sharpe", "sortino", "calmar",
               "max_dd_%", "win_rate_%", "avg_pos_%"]
    metric_labels = {
        "total_%":    "Total Return %",
        "cagr_%":     "CAGR %",
        "sharpe":     "Sharpe",
        "sortino":    "Sortino",
        "calmar":     "Calmar",
        "max_dd_%":   "Max Drawdown %",
        "win_rate_%": "Monthly Win %",
        "avg_pos_%":  "Avg Position %",
    }

    # Header row
    print(f"\n{'Metric':<22} {'Buy & Hold':>{col_w}} {'Binary':>{col_w}} {sized_label:>{col_w}}")
    print("-" * (22 + col_w * 3 + 6))

    for m in metrics:
        bh_val     = bh_m[m]
        binary_val = binary_m[m]
        sized_val  = sized_m[m]
        # Highlight if strategy beats buy-and-hold (mark with *)
        binary_flag = " *" if _is_better(m, binary_val, bh_val) else "  "
        sized_flag  = " *" if _is_better(m, sized_val,  bh_val) else "  "
        print(f"  {metric_labels[m]:<20} {bh_val:>{col_w}.2f} "
              f"{binary_val:>{col_w}.2f}{binary_flag}"
              f"{sized_val:>{col_w}.2f}{sized_flag}")

    print(f"\n* = beats Buy & Hold on this metric")

    # Edge summary
    print(f"\n--- Sharpe Edge vs Buy & Hold ---")
    print(f"  Binary : {binary_m['sharpe'] - bh_m['sharpe']:+.3f}")
    print(f"  Sized  : {sized_m['sharpe']  - bh_m['sharpe']:+.3f}")
    print(f"\n--- Max Drawdown Improvement (pp, positive = less bad) ---")
    print(f"  Binary : {bh_m['max_dd_%'] - binary_m['max_dd_%']:+.2f} pp")
    print(f"  Sized  : {bh_m['max_dd_%'] - sized_m['max_dd_%']:+.2f} pp")

    print(f"\nNote: 1-day signal lag (yesterday's regime -> today's position).")
    print(f"No lookahead. min_position={min_position*100:.0f}% means you are "
          f"always at least {min_position*100:.0f}% invested.")

    return results


def _is_better(metric: str, val: float, baseline: float) -> bool:
    """True if val is better than baseline for this metric."""
    if pd.isna(val) or pd.isna(baseline):
        return False
    # For max drawdown, less negative is better
    if metric == "max_dd_%":
        return val > baseline
    # For everything else, higher is better
    return val > baseline


# --------------------------------------------------------------------------
# 4. Visualise position size over time
# --------------------------------------------------------------------------

def plot_position_size(results: pd.DataFrame, ticker_label: str,
                        save_path: str = None):
    """
    Three-panel plot:
      Top:    Cumulative equity of all three strategies
      Middle: Daily position size (sized vs binary)
      Bottom: Regime probability stack (how uncertain the model is each day)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # --- Panel 1: Equity curves ---
    ax = axes[0]
    ax.plot(results.index, results["bh_equity"],     label="Buy & Hold",  color="gray",      lw=1.2)
    ax.plot(results.index, results["binary_equity"], label="Binary",       color="tomato",    lw=1.2)
    ax.plot(results.index, results["sized_equity"],  label="Sized",        color="steelblue", lw=1.5)
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"{ticker_label}: Regime-Sized Strategy vs Binary vs Buy & Hold",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    # --- Panel 2: Position size over time ---
    min_pos = results["sized_position"].min()
    ax = axes[1]
    ax.plot(results.index, results["sized_position"],  label="Sized position",  color="steelblue", lw=1)
    ax.step(results.index, results["binary_position"], label="Binary position", color="tomato",    lw=0.8, alpha=0.7)
    ax.axhline(0.5, color="black",     lw=0.5, ls="--", alpha=0.4)
    ax.axhline(min_pos, color="steelblue", lw=1.2, ls=":", alpha=0.8,
               label=f"Floor (min position = {min_pos*100:.0f}%)")
    ax.set_ylabel("Position Size (0=cash, 1=full)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    # --- Panel 3: Probability stack ---
    ax = axes[2]
    prob_cols = [c for c in results.columns if c.startswith("p_regime_")]
    if prob_cols:
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(prob_cols)))
        bottom = np.zeros(len(results))
        for col, color in zip(prob_cols, colors):
            ax.fill_between(results.index, bottom, bottom + results[col].values,
                             color=color, alpha=0.7, label=col)
            bottom += results[col].values
        ax.set_ylabel("Regime Probability")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8, title="0=worst")
    else:
        ax.text(0.5, 0.5, "Attach proba_df columns to results to see this panel",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Date")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
        print(f"Saved -> {save_path}")
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Quick runner (mirrors the structure of run() in regime_hmm.py)
# --------------------------------------------------------------------------

def run_sized(stock_input: str, shape: str = "linear", min_position: float = 0.0):
    """
    Drop-in companion to run() in regime_hmm.py.
    Runs the full pipeline and adds the sized position output.

    shape: "linear" | "convex" | "concave"  (see regime_weights docstring)
    min_position: never go below this exposure (0.0 = allow full cash)
    """
    import os
    # Reuse helpers from your existing module
    from regime_hmm import (resolve_ticker, load_prices_with_fallback,
                              build_features, select_n_states, make_output_dirs)

    ticker, label, fallback = resolve_ticker(stock_input)
    paths = make_output_dirs(label)

    prices, _ = load_prices_with_fallback(ticker, fallback)
    features = build_features(prices)

    print(f"\n=== Selecting number of states for {label} ===")
    _, k = select_n_states(features.values, max_states=4)

    print(f"\n=== Walk-forward probability fit (k={k}) ===")
    proba_df = walk_forward_proba(features, n_states=k)

    print(f"\n=== Backtesting sized strategy (shape='{shape}', "
          f"min_position={min_position}) ===")
    results = backtest_sized_strategy(prices, proba_df, shape=shape,
                                       min_position=min_position)

    # Attach probability columns to results for the plot
    for col in proba_df.columns:
        results[col] = proba_df[col]

    plot_position_size(
        results, label,
        save_path=os.path.join(paths["plots"], f"{label}_sized_position.png")
    )

    return results, proba_df


if __name__ == "__main__":
    stock = input("Enter stock (or NSEI for Nifty): ")
    shape = input("Weight shape [linear/convex/concave] (default: linear): ").strip() or "linear"

    _min_input = input(
        "Minimum position floor — how much to stay invested even in the worst "
        "regime? (e.g. 0.2 = always at least 20% in, 0 = allow full cash, "
        "default: 0): "
    ).strip()
    try:
        min_position = float(_min_input) if _min_input else 0.0
        if not (0.0 <= min_position <= 1.0):
            raise ValueError
    except ValueError:
        print("Invalid input -- defaulting min_position to 0.0")
        min_position = 0.0

    print(f"\nRunning with: shape='{shape}', min_position={min_position*100:.0f}%")
    run_sized(stock, shape=shape, min_position=min_position)