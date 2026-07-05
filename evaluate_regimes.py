"""
evaluate_regimes.py
--------------------
Part 1 upgrade for regime_hmm.py: proper evaluation metrics across multiple
tickers so you can answer "does my model actually work?" with numbers.

Run directly:
    python evaluate_regimes.py
    # prompts you interactively for tickers, output dir, and options

Or import and call from another script:
    from evaluate_regimes import run_evaluation
    summary = run_evaluation(["RELIANCE", "TCS", "AAPL"], output_dir="outputs/eval")

No tickers are hardcoded. The script always asks you at runtime.
"""

from __future__ import annotations  # X | Y union hints on Python 3.9+

import argparse
import os
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# Metric helpers
# ============================================================

def max_drawdown(equity: pd.Series) -> float:
    """
    Worst peak-to-trough loss as a negative fraction.
    e.g. -0.35 means the strategy fell 35% from its peak at the worst point.
    """
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def cagr(equity: pd.Series, trading_days_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.
    equity must be a cumulative product series starting at 1.0,
    i.e. (1 + daily_returns).cumprod().
    """
    n = len(equity)
    if n < 2:
        return np.nan
    years = n / trading_days_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1)


def sharpe(daily_returns: pd.Series, rf: float = 0.0,
           trading_days: int = 252) -> float:
    """Annualised Sharpe ratio. rf is the daily risk-free rate (default 0)."""
    excess = daily_returns - rf
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float((excess.mean() / std) * np.sqrt(trading_days))


def sortino(daily_returns: pd.Series, rf: float = 0.0,
            trading_days: int = 252) -> float:
    """
    Sortino ratio: penalises only downside volatility, not upside.
    More appropriate than Sharpe for strategies that asymmetrically cut losses.
    """
    excess = daily_returns - rf
    downside = excess[excess < 0]
    ds_std = downside.std()
    if ds_std == 0 or np.isnan(ds_std):
        return np.nan
    return float((excess.mean() / ds_std) * np.sqrt(trading_days))


def calmar(daily_returns: pd.Series) -> float:
    """
    Calmar ratio = CAGR / |max drawdown|.
    Measures annualised return per unit of worst-case loss. Higher is better.
    """
    equity = (1 + daily_returns).cumprod()
    ann = cagr(equity)
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(mdd) or np.isnan(ann):
        return np.nan
    return float(ann / abs(mdd))


def win_rate_monthly(daily_returns: pd.Series) -> float:
    """
    Fraction of calendar months where the strategy had a positive return.
    More interpretable than a daily win rate (~21 observations per month
    is enough to be directionally meaningful).
    Uses 'MS' (month-start) offset which is stable across pandas versions.
    """
    # 'MS' = month-start anchor — avoids the deprecated 'M' and the
    # pandas-version-dependent 'ME' alias, so this works on pandas 1.x–2.x.
    monthly = (1 + daily_returns).resample("MS").prod() - 1
    if len(monthly) == 0:
        return np.nan
    return float((monthly > 0).mean())


def compute_metrics(daily_returns: pd.Series, label: str = "") -> dict:
    """
    Computes all six metrics in one call.
    Returns a flat dict ready to be appended as a DataFrame row.
    """
    equity = (1 + daily_returns).cumprod()
    mdd = max_drawdown(equity)
    total_ret = float(equity.iloc[-1] - 1)

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
# Per-ticker evaluation pipeline
# ============================================================

def evaluate_ticker(stock_input: str,
                    plots_dir: str | None = None,
                    min_train: int = 500,
                    refit_every: int = 21) -> dict | None:
    """
    Runs the full walk-forward HMM pipeline on one ticker and returns a dict
    of metrics for both buy-and-hold and the regime strategy.

    Returns None — and prints why — if the ticker fails for any reason
    (bad symbol, too little history, download error, HMM fit collapse).
    The caller can then skip it cleanly without crashing the whole run.

    Parameters
    ----------
    stock_input  : ticker string in the same format as regime_hmm.run()
                   e.g. "RELIANCE", "NSEI", "AAPL", "RELIANCE.NS"
    plots_dir    : if given, saves a hero chart for this ticker here
    min_train    : walk-forward initial training window in trading days
    refit_every  : how many days between HMM refits in the walk-forward loop
    """
    # regime_hmm.py must be on the Python path (same directory is fine)
    try:
        from regime_hmm import (
            resolve_ticker,
            load_prices_with_fallback,
            build_features,
            select_n_states,
            walk_forward_regimes,
        )
    except ImportError as e:
        print(f"  ERROR: cannot import regime_hmm — {e}")
        print(f"  Make sure regime_hmm.py is in the same directory as this script.")
        return None

    ticker, label, fallback = resolve_ticker(stock_input)

    # ---- Download prices ----
    try:
        prices, _ = load_prices_with_fallback(ticker, fallback)
    except ValueError as e:
        print(f"  [{label}] SKIP (download failed): {e}")
        return None

    if prices is None or prices.empty:
        print(f"  [{label}] SKIP: yfinance returned no data for '{ticker}'")
        return None

    MIN_REQUIRED_DAYS = 200
    if len(prices) < MIN_REQUIRED_DAYS:
        print(f"  [{label}] SKIP: only {len(prices)} trading days of history "
              f"(need at least {MIN_REQUIRED_DAYS})")
        return None

    # ---- Feature engineering ----
    try:
        features = build_features(prices)
    except Exception as e:
        print(f"  [{label}] SKIP (feature build failed): {e}")
        return None

    if features.empty:
        print(f"  [{label}] SKIP: feature DataFrame is empty after dropna")
        return None

    # ---- Model selection + walk-forward ----
    try:
        print(f"  [{label}] Selecting number of HMM states...")
        _, k = select_n_states(features.values, max_states=4)

        print(f"  [{label}] Walk-forward fit with k={k} states...")
        wf_regimes = walk_forward_regimes(
            features, n_states=k,
            min_train=min_train,
            refit_every=refit_every,
        )
    except Exception as e:
        print(f"  [{label}] SKIP (model fit failed): {e}")
        return None

    if wf_regimes is None or len(wf_regimes) == 0:
        print(f"  [{label}] SKIP: walk-forward produced no labeled days")
        return None

    # ---- Align prices / build returns ----
    try:
        wf_features    = features.loc[wf_regimes.index]
        aligned_prices = prices.loc[wf_features.index]
        daily_ret      = aligned_prices.pct_change().fillna(0)

        # Sanity check: index alignment
        if not daily_ret.index.equals(wf_regimes.index):
            wf_regimes = wf_regimes.reindex(daily_ret.index).dropna().astype(int)
            daily_ret  = daily_ret.loc[wf_regimes.index]

        # Binary strategy: fully invested unless in worst regime (0), then cash
        position  = (wf_regimes != 0).astype(int)
        strat_ret = daily_ret * position.shift(1).fillna(0)

    except Exception as e:
        print(f"  [{label}] SKIP (return computation failed): {e}")
        return None

    # ---- Metrics ----
    try:
        bh_m    = compute_metrics(daily_ret,  label="buy_hold")
        strat_m = compute_metrics(strat_ret, label="regime_strat")
    except Exception as e:
        print(f"  [{label}] SKIP (metrics computation failed): {e}")
        return None

    row = {
        "ticker":     label,
        "n_days":     bh_m["n_days"],
        "n_regimes":  k,
        # Buy & hold
        "bh_total_%":      bh_m["total_return_%"],
        "bh_cagr_%":       bh_m["cagr_%"],
        "bh_sharpe":       bh_m["sharpe"],
        "bh_sortino":      bh_m["sortino"],
        "bh_maxdd_%":      bh_m["max_drawdown_%"],
        "bh_monthly_win":  bh_m["monthly_win_%"],
        # Regime strategy
        "reg_total_%":     strat_m["total_return_%"],
        "reg_cagr_%":      strat_m["cagr_%"],
        "reg_sharpe":      strat_m["sharpe"],
        "reg_sortino":     strat_m["sortino"],
        "reg_maxdd_%":     strat_m["max_drawdown_%"],
        "reg_monthly_win": strat_m["monthly_win_%"],
        # Edge: strategy minus buy-and-hold
        "sharpe_edge":          round(strat_m["sharpe"]        - bh_m["sharpe"],        3),
        "sortino_edge":         round(strat_m["sortino"]       - bh_m["sortino"],       3),
        "maxdd_improvement_%":  round(bh_m["max_drawdown_%"]   - strat_m["max_drawdown_%"], 2),
        "pct_time_in_market":   round(float(position.mean()) * 100, 1),
    }

    # ---- Hero chart ----
    if plots_dir:
        try:
            _plot_hero(
                prices=aligned_prices,
                regimes=wf_regimes,
                daily_ret=daily_ret,
                strat_ret=strat_ret,
                k=k,
                label=label,
                save_path=os.path.join(plots_dir, f"{label}_evaluation.png"),
            )
        except Exception as e:
            print(f"  [{label}] WARNING: hero chart failed (non-fatal): {e}")

    print(f"  [{label}] Done — Sharpe edge: {row['sharpe_edge']:+.3f}, "
          f"Max DD improved: {row['maxdd_improvement_%']:+.1f}pp, "
          f"Time in market: {row['pct_time_in_market']:.1f}%")
    return row


# ============================================================
# Hero chart (per ticker)
# ============================================================

def _plot_hero(prices: pd.Series, regimes: pd.Series,
               daily_ret: pd.Series, strat_ret: pd.Series,
               k: int, label: str, save_path: str) -> None:
    """
    Two-panel hero chart — the single image that explains the model to anyone.

    Top panel:    Price history with regime-colored background bands
                  (red = worst regime, green = best, yellow = middle).
    Bottom panel: Cumulative equity curves — Buy & Hold vs Regime Strategy,
                  with Sharpe and Max Drawdown in the legend.
    """
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, k))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    # ---- Top: price + regime background bands ----
    ax1.plot(prices.index, prices.values, color="black", lw=0.8, zorder=3)

    regime_arr = regimes.reindex(prices.index).ffill()
    prev_regime, band_start = None, None

    for date, regime in zip(prices.index, regime_arr):
        if pd.isna(regime):
            continue
        regime = int(regime)
        if regime != prev_regime:
            if prev_regime is not None and band_start is not None:
                ax1.axvspan(band_start, date,
                            color=colors[prev_regime], alpha=0.25, zorder=1)
            band_start, prev_regime = date, regime

    if prev_regime is not None and band_start is not None:
        ax1.axvspan(band_start, prices.index[-1],
                    color=colors[prev_regime], alpha=0.25, zorder=1)

    patches = [
        mpatches.Patch(
            color=colors[i], alpha=0.6,
            label=(f"Regime {i} (worst)" if i == 0
                   else f"Regime {i} (best)" if i == k - 1
                   else f"Regime {i}")
        )
        for i in range(k)
    ]
    ax1.legend(handles=patches, loc="upper left", fontsize=9)
    ax1.set_ylabel("Price", fontsize=11)
    ax1.set_title(f"{label}: Walk-Forward HMM Regime Detection", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.2)

    # ---- Bottom: equity curves ----
    bh_equity    = (1 + daily_ret).cumprod()
    strat_equity = (1 + strat_ret).cumprod()

    bh_sharpe_val    = sharpe(daily_ret)
    strat_sharpe_val = sharpe(strat_ret)
    bh_mdd           = max_drawdown(bh_equity)
    strat_mdd        = max_drawdown(strat_equity)

    ax2.plot(bh_equity.index,    bh_equity.values,    color="gray",      lw=1.5,
             label=f"Buy & Hold   (Sharpe: {bh_sharpe_val:.2f}, MaxDD: {bh_mdd*100:.1f}%)")
    ax2.plot(strat_equity.index, strat_equity.values, color="steelblue", lw=1.8,
             label=f"Regime Strat (Sharpe: {strat_sharpe_val:.2f}, MaxDD: {strat_mdd*100:.1f}%)")
    ax2.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax2.set_ylabel("Cumulative Return (start = 1.0)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  Saved hero chart -> {save_path}")


# ============================================================
# Cross-ticker summary chart
# ============================================================

def plot_summary_comparison(summary: pd.DataFrame, save_path: str) -> None:
    """
    Two-panel bar chart: Sharpe ratio and Max Drawdown for every evaluated
    ticker, buy-and-hold vs regime strategy side by side.
    The "does it work in general?" view at a glance.
    """
    tickers = summary["ticker"].tolist()
    n = len(tickers)
    x = np.arange(n)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 1.4), 9))

    ax1.bar(x - width / 2, summary["bh_sharpe"],  width, label="Buy & Hold",   color="gray",      alpha=0.8)
    ax1.bar(x + width / 2, summary["reg_sharpe"], width, label="Regime Strat", color="steelblue", alpha=0.8)
    ax1.axhline(0, color="black", lw=0.7, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Sharpe Ratio: Buy & Hold vs Regime Strategy", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.2, axis="y")

    ax2.bar(x - width / 2, summary["bh_maxdd_%"],  width, label="Buy & Hold",   color="gray",      alpha=0.8)
    ax2.bar(x + width / 2, summary["reg_maxdd_%"], width, label="Regime Strat", color="steelblue", alpha=0.8)
    ax2.axhline(0, color="black", lw=0.7, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.set_title("Max Drawdown: closer to 0 = better", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"Saved summary chart -> {save_path}")


# ============================================================
# Main evaluation runner
# ============================================================

def run_evaluation(tickers: list[str],
                   output_dir: str = "outputs/evaluation",
                   min_train: int = 500,
                   refit_every: int = 21,
                   save_plots: bool = True) -> pd.DataFrame:
    """
    Runs the full evaluation pipeline across a list of tickers.

    Parameters
    ----------
    tickers      : list of ticker strings, e.g. ["RELIANCE", "TCS", "AAPL"]
    output_dir   : directory for summary.csv and all charts
    min_train    : initial walk-forward training window (trading days)
    refit_every  : HMM refit interval in the walk-forward loop (trading days)
    save_plots   : set False to skip charts (useful for quick metric-only runs)

    Returns
    -------
    pd.DataFrame with one row per successful ticker, all metric columns.
    Also saved to <output_dir>/summary.csv.
    """
    if not tickers:
        raise ValueError("run_evaluation: tickers list is empty.")

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    total = len(tickers)
    rows  = []

    print(f"\n{'='*62}")
    print(f"  Evaluation: {total} ticker(s)")
    print(f"  Output dir: {output_dir}")
    print(f"  min_train={min_train}, refit_every={refit_every}, plots={save_plots}")
    print(f"{'='*62}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] {ticker}")
        row = evaluate_ticker(
            stock_input=ticker,
            plots_dir=plots_dir if save_plots else None,
            min_train=min_train,
            refit_every=refit_every,
        )
        if row is not None:
            rows.append(row)
        print()

    if not rows:
        print("No tickers succeeded — check symbols and internet connection.")
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    # ---- Summary table ----
    display_cols = [
        "ticker", "n_days", "n_regimes",
        "bh_sharpe",  "reg_sharpe",  "sharpe_edge",
        "bh_maxdd_%", "reg_maxdd_%", "maxdd_improvement_%",
        "bh_cagr_%",  "reg_cagr_%",
        "reg_monthly_win", "pct_time_in_market",
    ]
    # Only show columns that actually exist (guards against partial failures)
    display_cols = [c for c in display_cols if c in summary.columns]

    print(f"\n{'='*62}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*62}")
    with pd.option_context("display.max_rows", None, "display.width", 160,
                           "display.float_format", "{:.2f}".format):
        print(summary[display_cols].to_string(index=False))

    # ---- Aggregate stats ----
    succeeded = len(summary)
    n_beat_sharpe = int((summary["sharpe_edge"] > 0).sum())
    n_beat_dd     = int((summary["maxdd_improvement_%"] > 0).sum())

    print(f"\n--- Aggregate across {succeeded} ticker(s) ---")
    print(f"  Beat buy-and-hold Sharpe  : {n_beat_sharpe}/{succeeded} "
          f"({n_beat_sharpe/succeeded*100:.0f}%)")
    print(f"  Reduced max drawdown      : {n_beat_dd}/{succeeded} "
          f"({n_beat_dd/succeeded*100:.0f}%)")
    print(f"  Avg Sharpe edge           : {summary['sharpe_edge'].mean():+.3f}")
    print(f"  Avg Max DD improvement    : {summary['maxdd_improvement_%'].mean():+.2f} pp")
    print(f"  Avg time in market        : {summary['pct_time_in_market'].mean():.1f}%")
    print()
    print("  Interpret: if strategy beats Sharpe AND reduces drawdown on 7+/10")
    print("  tickers, the model carries real signal. If it's 5/10 or less, state")
    print("  that honestly in your README — inconsistency is still a finding.")

    # ---- Save CSV ----
    csv_path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV -> {csv_path}")

    # ---- Summary chart (only if more than one ticker succeeded) ----
    if save_plots and succeeded > 1:
        try:
            plot_summary_comparison(
                summary,
                save_path=os.path.join(plots_dir, "summary_comparison.png"),
            )
        except Exception as e:
            print(f"  WARNING: summary chart failed (non-fatal): {e}")

    return summary


# ============================================================
# CLI entry point — no hardcoded tickers
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the walk-forward HMM regime strategy across multiple tickers.\n"
            "Tickers are always provided by you at runtime — nothing is hardcoded."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate_regimes.py\n"
            "      (interactive: prompts for tickers)\n\n"
            "  python evaluate_regimes.py --tickers RELIANCE TCS INFY AAPL MSFT\n"
            "      (non-interactive: tickers passed on command line)\n\n"
            "  python evaluate_regimes.py --tickers RELIANCE TCS --no-plots\n"
            "      (skip chart generation — faster for a quick metrics check)\n\n"
            "  python evaluate_regimes.py --tickers HDFCBANK --output outputs/hdfc_test\n"
            "      (custom output directory)"
        )
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        metavar="TICKER",
        help=(
            "Space-separated list of tickers to evaluate. "
            "Use the same format as regime_hmm.run(): plain NSE name (RELIANCE), "
            "NSEI for the Nifty index, or any Yahoo Finance ticker (AAPL, MSFT). "
            "If omitted, the script prompts you interactively."
        ),
    )
    parser.add_argument(
        "--output", default="outputs/evaluation",
        metavar="DIR",
        help="Directory for summary.csv and charts. Created if it doesn't exist. "
             "(default: outputs/evaluation)",
    )
    parser.add_argument(
        "--min-train", type=int, default=500,
        metavar="N",
        help="Walk-forward initial training window in trading days. "
             "(default: 500 — about 2 years)",
    )
    parser.add_argument(
        "--refit-every", type=int, default=21,
        metavar="N",
        help="How many trading days between HMM refits in the walk-forward loop. "
             "(default: 21 — roughly monthly)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip all chart generation. Useful for a fast metrics-only run.",
    )
    return parser.parse_args()


def _prompt_tickers() -> list[str]:
    """Interactive ticker input when --tickers is not passed on the CLI."""
    print("=" * 62)
    print("  evaluate_regimes.py — interactive mode")
    print("=" * 62)
    print()
    print("Enter the tickers you want to evaluate.")
    print("  NSE stocks  : just the name, e.g. RELIANCE  TCS  INFY")
    print("  Nifty index : NSEI")
    print("  US / other  : plain Yahoo Finance ticker, e.g. AAPL  MSFT")
    print()

    raw = input("Tickers (space or comma separated): ").strip()
    if not raw:
        print("No tickers entered — exiting.")
        sys.exit(0)

    # Accept both space-separated and comma-separated input
    tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    if not tickers:
        print("Could not parse any tickers — exiting.")
        sys.exit(0)

    print(f"\nWill evaluate: {', '.join(tickers)}")
    return tickers


if __name__ == "__main__":
    args = _parse_args()

    # Resolve tickers: CLI flag takes priority, otherwise prompt interactively
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    else:
        tickers = _prompt_tickers()

    run_evaluation(
        tickers=tickers,
        output_dir=args.output,
        min_train=args.min_train,
        refit_every=args.refit_every,
        save_plots=not args.no_plots,
    )