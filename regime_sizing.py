"""
Regime-Conditioned Position Sizing
------------------------------------
Replaces the binary invested/cash signal with a continuous position size
derived from the HMM's posterior regime probabilities.

Drop-in additions to regime_hmm.py:
  - walk_forward_proba()       -> returns per-regime probability DataFrame
  - regime_weights()           -> maps regime index to a [0,1] weight
  - compute_position_size()    -> turns proba DataFrame into a position Series
  - backtest_sized_strategy()  -> full metrics table for all three strategies
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

MIN_VIABLE_TRAIN_ROWS = 60


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------

def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def _cagr(equity: pd.Series, trading_days: int = 252) -> float:
    n = len(equity)
    if n < 2:
        return np.nan
    years = n / trading_days
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def _sharpe(r: pd.Series, trading_days: int = 252) -> float:
    std = r.std()
    return float((r.mean() / std) * np.sqrt(trading_days)) if std > 0 else np.nan


def _sortino(r: pd.Series, trading_days: int = 252) -> float:
    downside = r[r < 0]
    ds = downside.std()
    if ds == 0 or np.isnan(ds):
        return np.nan
    return float((r.mean() / ds) * np.sqrt(trading_days))


def _calmar(r: pd.Series) -> float:
    equity = (1 + r).cumprod()
    mdd = _max_drawdown(equity)
    ann = _cagr(equity)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(ann / abs(mdd))


def _monthly_win(r: pd.Series) -> float:
    # FIX: "ME" is pandas>=2.2 only and broken on older versions.
    # "MS" (month-start anchor) is stable across pandas 1.x-2.x.
    monthly = (1 + r).resample("MS").prod() - 1
    return float((monthly > 0).mean()) if len(monthly) > 0 else np.nan


def _metrics_block(r: pd.Series, pos: pd.Series) -> dict:
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
    Causal walk-forward fit returning a DataFrame of shape
    (n_labeled_days, n_states) with posterior probability vectors.
    Columns are named by relabeled regime index (0 = worst return, k-1 = best).
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

    proba_df = pd.DataFrame(
        np.nan,
        index=features.index,
        columns=[f"p_regime_{i}" for i in range(n_states)]
    )

    model = None
    last_refit = -1
    remap = None

    for t in range(min_train, n):

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

        try:
            window = values[:t + 1]
            raw_proba = model.predict_proba(window)[-1]
            reordered = np.zeros(n_states)
            for raw_state, new_state in remap.items():
                reordered[new_state] = raw_proba[raw_state]
            proba_df.iloc[t] = reordered
        except Exception:
            pass

    proba_df = proba_df.dropna()
    return proba_df


# --------------------------------------------------------------------------
# 2. Turn probabilities into a position size
# --------------------------------------------------------------------------

def regime_weights(n_states: int, shape: str = "linear") -> np.ndarray:
    """
    Maps regime index (0=worst, k-1=best) to a weight in [0,1].
    shape: "linear" | "convex" | "concave"
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
    position_t = sum_i( P(regime_i | data_t) * weight_i )
    Clipped to [min_position, max_position].
    """
    n_states = proba_df.shape[1]
    weights = regime_weights(n_states, shape=shape)
    raw_position = proba_df.values @ weights
    clipped = np.clip(raw_position, min_position, max_position)
    position = pd.Series(clipped, index=proba_df.index, name="position_size")

    hard_regime = proba_df.idxmax(axis=1).str.extract(r"(\d+)$")[0].astype(int)
    worst_days_mask = (hard_regime == 0)
    n_worst = worst_days_mask.sum()

    if n_worst > 0:
        avg_exposure_worst = position[worst_days_mask].mean()
        min_exposure_worst = position[worst_days_mask].min()
        print(f"\n=== Worst-Regime Exposure ===")
        print(f"Days classified as worst regime: {n_worst} "
              f"({n_worst / len(position) * 100:.1f}% of labeled history)")
        print(f"  Average : {avg_exposure_worst * 100:.1f}%")
        print(f"  Minimum : {min_exposure_worst * 100:.1f}%")
    else:
        print("\n=== Worst-Regime Exposure ===")
        print("No days classified as worst regime in the labeled history.")

    return position


# --------------------------------------------------------------------------
# 3. Backtest the sized strategy
# --------------------------------------------------------------------------

def backtest_sized_strategy(prices: pd.Series,
                             proba_df: pd.DataFrame,
                             shape: str = "linear",
                             min_position: float = 0.0,
                             avoid_regime: int = 0) -> pd.DataFrame:
    """
    Compares Buy & Hold, Binary (original), and Sized (new) strategies.
    Prints a full metrics table. Returns DataFrame of daily returns + equity.
    """
    aligned_prices = prices.loc[proba_df.index]
    daily_ret = aligned_prices.pct_change().fillna(0)

    hard_regime = proba_df.idxmax(axis=1).str.extract(r"(\d+)$")[0].astype(int)

    binary_pos = (hard_regime != avoid_regime).astype(float)
    binary_ret = daily_ret * binary_pos.shift(1).fillna(0)

    sized_pos = compute_position_size(proba_df, shape=shape,
                                       min_position=min_position)
    sized_ret = daily_ret * sized_pos.shift(1).fillna(0)

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

    bh_m      = _metrics_block(daily_ret, pd.Series(1.0, index=daily_ret.index))
    binary_m  = _metrics_block(binary_ret, binary_pos)
    sized_m   = _metrics_block(sized_ret, sized_pos)

    sized_label = f"Sized ({shape}, floor={min_position*100:.0f}%)"

    print(f"\n{'='*80}\n{'Strategy Comparison':^80}\n{'='*80}")

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

    print(f"\n{'Metric':<22} {'Buy & Hold':>{col_w}} {'Binary':>{col_w}} {sized_label:>{col_w}}")
    print("-" * (22 + col_w * 3 + 6))

    for m in metrics:
        bh_val     = bh_m[m]
        binary_val = binary_m[m]
        sized_val  = sized_m[m]
        binary_flag = " *" if _is_better(m, binary_val, bh_val) else "  "
        sized_flag  = " *" if _is_better(m, sized_val,  bh_val) else "  "
        print(f"  {metric_labels[m]:<20} {bh_val:>{col_w}.2f} "
              f"{binary_val:>{col_w}.2f}{binary_flag}"
              f"{sized_val:>{col_w}.2f}{sized_flag}")

    print(f"\n* = beats Buy & Hold on this metric")
    print(f"\n--- Sharpe Edge vs Buy & Hold ---")
    print(f"  Binary : {binary_m['sharpe'] - bh_m['sharpe']:+.3f}")
    print(f"  Sized  : {sized_m['sharpe']  - bh_m['sharpe']:+.3f}")
    print(f"\n--- Max Drawdown Improvement (pp) ---")
    print(f"  Binary : {bh_m['max_dd_%'] - binary_m['max_dd_%']:+.2f} pp")
    print(f"  Sized  : {bh_m['max_dd_%'] - sized_m['max_dd_%']:+.2f} pp")
    print(f"\nNote: 1-day signal lag. min_position={min_position*100:.0f}%.")

    return results


def _is_better(metric: str, val: float, baseline: float) -> bool:
    if pd.isna(val) or pd.isna(baseline):
        return False
    if metric == "max_dd_%":
        return val > baseline
    return val > baseline


# --------------------------------------------------------------------------
# 4. Visualise position size over time
# --------------------------------------------------------------------------

def plot_position_size(results: pd.DataFrame, ticker_label: str,
                        save_path: str = None):
    """
    Three-panel: equity curves / position size over time / regime prob stack.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    ax = axes[0]
    ax.plot(results.index, results["bh_equity"],     label="Buy & Hold",  color="gray",      lw=1.2)
    ax.plot(results.index, results["binary_equity"], label="Binary",       color="tomato",    lw=1.2)
    ax.plot(results.index, results["sized_equity"],  label="Sized",        color="steelblue", lw=1.5)
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"{ticker_label}: Regime-Sized vs Binary vs Buy & Hold",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    min_pos = results["sized_position"].min()
    ax = axes[1]
    ax.plot(results.index, results["sized_position"],  label="Sized position",  color="steelblue", lw=1)
    ax.step(results.index, results["binary_position"], label="Binary position", color="tomato",    lw=0.8, alpha=0.7)
    ax.axhline(0.5, color="black",     lw=0.5, ls="--", alpha=0.4)
    ax.axhline(min_pos, color="steelblue", lw=1.2, ls=":", alpha=0.8,
               label=f"Floor ({min_pos*100:.0f}%)")
    ax.set_ylabel("Position Size (0=cash, 1=full)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

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
# 5. Quick runner
# --------------------------------------------------------------------------

def run_sized(stock_input: str, shape: str = "linear", min_position: float = 0.0):
    """
    Full pipeline: download -> features -> model selection -> walk-forward proba
    -> sized backtest -> plot.
    """
    import os
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
        "Minimum position floor (e.g. 0.2 = always 20% invested, 0 = allow full cash, "
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