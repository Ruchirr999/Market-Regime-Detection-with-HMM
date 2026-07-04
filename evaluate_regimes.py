"""
evaluate_regimes.py
--------------------
Part 1 upgrade for regime_hmm.py: proper evaluation metrics across multiple
tickers so you can answer "does my model actually work?" with numbers.

What this adds:
  - max_drawdown()          -> worst peak-to-trough loss
  - compute_metrics()       -> Sharpe, Sortino, max drawdown, CAGR, win-rate,
                               calmar ratio -- all in one dict
  - evaluate_ticker()       -> runs the full regime_hmm pipeline on one ticker
                               and returns a clean metrics row
  - run_evaluation()        -> loops over a list of tickers, collects metrics,
                               prints a summary table, saves to CSV
  - plot_equity_curves()    -> hero chart: regime-colored price + equity curves

How to use:
    from evaluate_regimes import run_evaluation

    tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "WIPRO",
               "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    summary = run_evaluation(tickers, output_dir="outputs/evaluation")

The summary DataFrame is also saved as outputs/evaluation/summary.csv.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


# ============================================================
# Metric helpers
# ============================================================

def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a negative fraction.
    e.g. -0.35 means the strategy fell 35% from its peak at worst.
    """
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def cagr(equity: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.
    equity should start at 1.0 (i.e. (1 + daily_returns).cumprod()).
    """
    n_days = len(equity)
    if n_days < 2:
        return np.nan
    total_return = equity.iloc[-1] / equity.iloc[0]
    years = n_days / trading_days_per_year
    return float(total_return ** (1 / years) - 1)


def sharpe(daily_returns: pd.Series, rf: float = 0.0,
           trading_days: int = 252) -> float:
    """Annualised Sharpe ratio. rf is the daily risk-free rate (default 0)."""
    excess = daily_returns - rf
    std = excess.std()
    return float((excess.mean() / std) * np.sqrt(trading_days)) if std > 0 else np.nan


def sortino(daily_returns: pd.Series, rf: float = 0.0,
            trading_days: int = 252) -> float:
    """
    Sortino ratio: like Sharpe but only penalises downside volatility.
    A better measure for strategies that cut losses (like this one) because
    it doesn't penalise upside variance.
    """
    excess = daily_returns - rf
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return float((excess.mean() / downside_std) * np.sqrt(trading_days))


def calmar(daily_returns: pd.Series) -> float:
    """
    Calmar ratio = CAGR / |max drawdown|.
    Higher is better. Measures return per unit of worst-case drawdown risk.
    """
    equity = (1 + daily_returns).cumprod()
    ann_return = cagr(equity)
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(ann_return / abs(mdd))


def win_rate_monthly(daily_returns: pd.Series) -> float:
    """
    Fraction of calendar months where the strategy had a positive return.
    More interpretable than daily win-rate.
    """
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    if len(monthly) == 0:
        return np.nan
    return float((monthly > 0).mean())


def compute_metrics(daily_returns: pd.Series, label: str = "") -> dict:
    """
    All metrics in one call. Returns a dict ready to go into a DataFrame row.
    """
    equity = (1 + daily_returns).cumprod()
    total_ret = float(equity.iloc[-1] - 1)
    mdd = max_drawdown(equity)

    return {
        "label":          label,
        "total_return_%": round(total_ret * 100, 2),
        "cagr_%":         round(cagr(equity) * 100, 2),
        "sharpe":         round(sharpe(daily_returns), 3),
        "sortino":        round(sortino(daily_returns), 3),
        "calmar":         round(calmar(daily_returns), 3),
        "max_drawdown_%": round(mdd * 100, 2),
        "monthly_win_%":  round(win_rate_monthly(daily_returns) * 100, 1),
        "n_days":         len(daily_returns),
    }


# ============================================================
# Per-ticker pipeline
# ============================================================

def evaluate_ticker(stock_input: str,
                    plots_dir: str = None,
                    min_train: int = 500,
                    refit_every: int = 21) -> dict | None:
    """
    Runs the full regime_hmm walk-forward pipeline on one ticker and
    returns a dict with metrics for both buy-and-hold and the regime strategy.

    Returns None if the ticker fails (too little data, download error, etc.)
    so run_evaluation() can skip it cleanly instead of crashing.
    """
    # Import from your existing module — must be on the Python path
    from regime_hmm import (
        resolve_ticker,
        load_prices_with_fallback,
        build_features,
        select_n_states,
        walk_forward_regimes,
    )

    ticker, label, fallback = resolve_ticker(stock_input)

    try:
        prices, _ = load_prices_with_fallback(ticker, fallback)
    except ValueError as e:
        print(f"  [{label}] SKIP: {e}")
        return None

    if prices.empty or len(prices) < 200:
        print(f"  [{label}] SKIP: not enough price history ({len(prices)} days)")
        return None

    try:
        features = build_features(prices)

        # Use BIC-selected k (same logic as your existing run())
        print(f"  [{label}] Selecting states...")
        _, k = select_n_states(features.values, max_states=4)

        print(f"  [{label}] Walk-forward fit (k={k})...")
        wf_regimes = walk_forward_regimes(
            features, n_states=k,
            min_train=min_train, refit_every=refit_every
        )

        wf_features = features.loc[wf_regimes.index]
        aligned_prices = prices.loc[wf_features.index]
        daily_ret = aligned_prices.pct_change().fillna(0)

        # Binary regime strategy: avoid regime 0
        position = (wf_regimes != 0).astype(int)
        strat_ret = daily_ret * position.shift(1).fillna(0)

        # --- Metrics ---
        bh_m   = compute_metrics(daily_ret, label="buy_hold")
        strat_m = compute_metrics(strat_ret, label="regime_strat")

        row = {
            "ticker": label,
            "n_days": bh_m["n_days"],
            "n_regimes": k,
            # Buy & hold
            "bh_total_%":     bh_m["total_return_%"],
            "bh_cagr_%":      bh_m["cagr_%"],
            "bh_sharpe":      bh_m["sharpe"],
            "bh_sortino":     bh_m["sortino"],
            "bh_maxdd_%":     bh_m["max_drawdown_%"],
            "bh_monthly_win": bh_m["monthly_win_%"],
            # Regime strategy
            "reg_total_%":     strat_m["total_return_%"],
            "reg_cagr_%":      strat_m["cagr_%"],
            "reg_sharpe":      strat_m["sharpe"],
            "reg_sortino":     strat_m["sortino"],
            "reg_maxdd_%":     strat_m["max_drawdown_%"],
            "reg_monthly_win": strat_m["monthly_win_%"],
            # Edge
            "sharpe_edge":     round(strat_m["sharpe"]   - bh_m["sharpe"],   3),
            "sortino_edge":    round(strat_m["sortino"]  - bh_m["sortino"],  3),
            "maxdd_improvement_%": round(bh_m["max_drawdown_%"] - strat_m["max_drawdown_%"], 2),
            "pct_time_in_market": round(position.mean() * 100, 1),
        }

        # --- Optional hero chart per ticker ---
        if plots_dir:
            _plot_hero(
                prices=aligned_prices,
                regimes=wf_regimes,
                daily_ret=daily_ret,
                strat_ret=strat_ret,
                k=k,
                label=label,
                save_path=os.path.join(plots_dir, f"{label}_evaluation.png"),
            )

        print(f"  [{label}] Done. Sharpe edge: {row['sharpe_edge']:+.3f}, "
              f"Max DD improved by: {row['maxdd_improvement_%']:+.1f}pp")
        return row

    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return None


# ============================================================
# Hero chart
# ============================================================

def _plot_hero(prices: pd.Series, regimes: pd.Series,
               daily_ret: pd.Series, strat_ret: pd.Series,
               k: int, label: str, save_path: str):
    """
    Two-panel hero chart:
      Top:    Price history with regime-colored background bands
              (green = best regime, red = worst, yellow = middle)
      Bottom: Equity curves — Buy & Hold vs Regime Strategy

    This is the chart that makes someone understand your model in 10 seconds.
    """
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, k))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 2]})

    # ---- Top panel: price + regime background ----
    ax1.plot(prices.index, prices.values, color="black", lw=0.8, zorder=3)

    # Draw colored vertical bands for each regime change
    regime_arr = regimes.reindex(prices.index).ffill()
    dates = prices.index
    prev_regime = None
    band_start = None

    for i, (date, regime) in enumerate(zip(dates, regime_arr)):
        if pd.isna(regime):
            continue
        regime = int(regime)
        if regime != prev_regime:
            if prev_regime is not None and band_start is not None:
                ax1.axvspan(band_start, date,
                            color=colors[prev_regime], alpha=0.25, zorder=1)
            band_start = date
            prev_regime = regime
    # Close the last band
    if prev_regime is not None and band_start is not None:
        ax1.axvspan(band_start, dates[-1],
                    color=colors[prev_regime], alpha=0.25, zorder=1)

    ax1.set_ylabel("Price", fontsize=11)
    ax1.set_title(f"{label}: Regime Detection — Walk-Forward HMM",
                  fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.2)

    # Legend for regime colors
    patches = [mpatches.Patch(color=colors[i], alpha=0.6,
                               label=f"Regime {i}" + (" (worst)" if i == 0 else
                                                        " (best)" if i == k-1 else ""))
               for i in range(k)]
    ax1.legend(handles=patches, loc="upper left", fontsize=9)

    # ---- Bottom panel: equity curves ----
    bh_equity   = (1 + daily_ret).cumprod()
    strat_equity = (1 + strat_ret).cumprod()

    ax2.plot(bh_equity.index,   bh_equity.values,   color="gray",      lw=1.5,
             label=f"Buy & Hold  (Sharpe: {sharpe(daily_ret):.2f}, "
                   f"MaxDD: {max_drawdown(bh_equity)*100:.1f}%)")
    ax2.plot(strat_equity.index, strat_equity.values, color="steelblue", lw=1.8,
             label=f"Regime Strat (Sharpe: {sharpe(strat_ret):.2f}, "
                   f"MaxDD: {max_drawdown(strat_equity)*100:.1f}%)")
    ax2.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax2.set_ylabel("Cumulative Return (1 = start)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved hero chart -> {save_path}")


# ============================================================
# Summary chart across all tickers
# ============================================================

def plot_summary_comparison(summary: pd.DataFrame, save_path: str):
    """
    Bar chart comparing Sharpe ratio and Max Drawdown across all evaluated
    tickers — buy-and-hold vs regime strategy side by side.
    Gives you the "does this work in general?" view at a glance.
    """
    tickers = summary["ticker"].tolist()
    x = np.arange(len(tickers))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(tickers) * 1.2), 9))

    # Sharpe comparison
    ax1.bar(x - width/2, summary["bh_sharpe"],   width, label="Buy & Hold",    color="gray",      alpha=0.8)
    ax1.bar(x + width/2, summary["reg_sharpe"],  width, label="Regime Strat",  color="steelblue", alpha=0.8)
    ax1.axhline(0, color="black", lw=0.7, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Sharpe Ratio: Buy & Hold vs Regime Strategy (per ticker)",
                  fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.2, axis="y")

    # Max drawdown comparison (both negative — regime strat should be less negative)
    ax2.bar(x - width/2, summary["bh_maxdd_%"],  width, label="Buy & Hold",    color="gray",      alpha=0.8)
    ax2.bar(x + width/2, summary["reg_maxdd_%"], width, label="Regime Strat",  color="steelblue", alpha=0.8)
    ax2.axhline(0, color="black", lw=0.7, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.set_title("Max Drawdown: Buy & Hold vs Regime Strategy — closer to 0 is better",
                  fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved summary chart -> {save_path}")


# ============================================================
# Main entry point
# ============================================================

def run_evaluation(tickers: list[str],
                   output_dir: str = "outputs/evaluation",
                   min_train: int = 500,
                   refit_every: int = 21,
                   save_plots: bool = True) -> pd.DataFrame:
    """
    Runs the full evaluation pipeline across all tickers and returns a
    summary DataFrame with all metrics.

    Parameters
    ----------
    tickers      : list of ticker strings (same format as regime_hmm.run())
    output_dir   : where to save summary.csv and all charts
    min_train    : walk-forward training window (passed to regime_hmm)
    refit_every  : how often to refit the HMM in walk-forward (in trading days)
    save_plots   : if False, skips all chart generation (faster for quick tests)

    Returns
    -------
    pd.DataFrame  with one row per ticker, all metrics columns
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    rows = []
    total = len(tickers)

    print(f"\n{'='*60}")
    print(f"Evaluation: {total} tickers")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] {ticker}")
        row = evaluate_ticker(
            ticker,
            plots_dir=plots_dir if save_plots else None,
            min_train=min_train,
            refit_every=refit_every,
        )
        if row is not None:
            rows.append(row)
        print()

    if not rows:
        print("No tickers succeeded. Check ticker symbols and internet connection.")
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    # ---- Print summary table ----
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    display_cols = [
        "ticker", "n_days", "n_regimes",
        "bh_sharpe", "reg_sharpe", "sharpe_edge",
        "bh_maxdd_%", "reg_maxdd_%", "maxdd_improvement_%",
        "bh_cagr_%", "reg_cagr_%",
        "reg_monthly_win", "pct_time_in_market",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 160,
                            "display.float_format", "{:.2f}".format):
        print(summary[display_cols].to_string(index=False))

    # ---- Aggregate stats ----
    succeeded = len(summary)
    strategy_better_sharpe = (summary["sharpe_edge"] > 0).sum()
    strategy_better_dd     = (summary["maxdd_improvement_%"] > 0).sum()

    print(f"\n--- Aggregate across {succeeded} tickers ---")
    print(f"Strategy beat buy-and-hold Sharpe:    "
          f"{strategy_better_sharpe}/{succeeded} tickers "
          f"({strategy_better_sharpe/succeeded*100:.0f}%)")
    print(f"Strategy reduced max drawdown:         "
          f"{strategy_better_dd}/{succeeded} tickers "
          f"({strategy_better_dd/succeeded*100:.0f}%)")
    print(f"Avg Sharpe edge (strat - B&H):         "
          f"{summary['sharpe_edge'].mean():+.3f}")
    print(f"Avg Max DD improvement (pp):           "
          f"{summary['maxdd_improvement_%'].mean():+.2f}pp")
    print(f"Avg % time in market (strat):          "
          f"{summary['pct_time_in_market'].mean():.1f}%")
    print(f"\nKey question to ask yourself:")
    print(f"  Does the strategy consistently improve Sharpe AND reduce drawdown,")
    print(f"  or does it only do so on some tickers? If it's inconsistent,")
    print(f"  that's worth stating honestly in your README — it's a sign the")
    print(f"  model is picking up some signal but isn't universally reliable.")

    # ---- Save outputs ----
    csv_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary saved -> {csv_path}")

    if save_plots and len(summary) > 1:
        plot_summary_comparison(
            summary,
            save_path=os.path.join(plots_dir, "summary_comparison.png")
        )

    return summary


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":
    # Default test list — mix of NSE and US tickers to show it works cross-market.
    # Edit this list to match the stocks you actually care about.
    DEFAULT_TICKERS = [
        # NSE stocks (just the name, regime_hmm adds .NS automatically)
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "WIPRO",
        # US stocks (plain ticker, regime_hmm detects they aren't NSE and uses fallback)
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
    ]

    print("Tickers to evaluate:")
    for t in DEFAULT_TICKERS:
        print(f"  {t}")
    print()

    custom = input(
        "Press Enter to use the defaults above, or type a comma-separated "
        "list of tickers to override: "
    ).strip()

    if custom:
        tickers = [t.strip() for t in custom.split(",") if t.strip()]
    else:
        tickers = DEFAULT_TICKERS

    results = run_evaluation(tickers, output_dir="outputs/evaluation")