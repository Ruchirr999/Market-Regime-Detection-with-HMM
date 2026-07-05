"""
Regime Filters and Severity Scoring
--------------------------------------
Three additions to the regime pipeline:

  1. apply_persistence_filter()  -- suppress flickering regime changes
  2. compute_severity_score()    -- single 0-1 "how bearish is today" number
  3. compute_position_size_v2()  -- graded sizing using severity score,
                                    replaces the version in regime_sizing.py

These are designed to plug into the existing walk_forward_proba() output
from regime_sizing.py and the features DataFrame from build_features().
"""

import numpy as np
import pandas as pd


# ==========================================================================
# 1. PERSISTENCE FILTER
# ==========================================================================

def apply_persistence_filter(regimes: pd.Series, min_days: int = 3) -> pd.Series:
    """
    Suppresses flickering regime changes by only confirming a new regime
    after it has persisted for `min_days` consecutive trading days.

    Until confirmation, the previous confirmed regime is held. This means
    the filtered series always lags the raw series by up to min_days,
    but trades far less and avoids reacting to single-day HMM noise.

    WHY THIS HELPS:
    The HMM's Viterbi decode can flip the regime label for a single day
    when the posterior probabilities are close (e.g. 52% regime 1 vs
    48% regime 0). That single-day flip triggers a position change in the
    backtest but carries no real information -- the market didn't change,
    the model just had a moment of uncertainty. The persistence filter
    treats those as noise and holds the previous regime until the new
    one has proven itself over multiple days.

    Parameters
    ----------
    regimes  : pd.Series of integer regime labels (raw walk-forward output)
    min_days : how many consecutive days in a new regime before confirming

    Returns
    -------
    pd.Series of filtered regime labels, same index as input

    LIMITATION:
    The filter introduces a lag of up to min_days. On a fast-moving crash
    (e.g. COVID week 1) this means you're still "confirmed" in regime 2
    for 3 days after the HMM first sees regime 0. This is the fundamental
    tradeoff -- less noise vs less responsiveness. min_days=3 is a
    reasonable default; go to 5 for more stability, 2 for more speed.
    """
    values = regimes.values
    n = len(values)
    filtered = values.copy()

    confirmed_regime = values[0]
    candidate_regime = values[0]
    candidate_streak = 1

    for i in range(1, n):
        current = values[i]

        if current == candidate_regime:
            candidate_streak += 1
        else:
            candidate_regime = current
            candidate_streak = 1

        if candidate_streak >= min_days:
            confirmed_regime = candidate_regime

        filtered[i] = confirmed_regime

    return pd.Series(filtered, index=regimes.index, name="regime_filtered")


def compare_filter_impact(prices: pd.Series, raw_regimes: pd.Series,
                           min_days_options: list = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Backtests with and without the persistence filter for multiple min_days
    values so you can measure the noise-reduction vs lag tradeoff directly.

    Returns a summary DataFrame with total return and Sharpe for each setting.
    """
    aligned = prices.loc[raw_regimes.index]
    daily_ret = aligned.pct_change().fillna(0)

    rows = []
    for min_days in min_days_options:
        filtered = apply_persistence_filter(raw_regimes, min_days=min_days)
        position = (filtered != 0).astype(float)
        strat_ret = daily_ret * position.shift(1).fillna(0)

        total = (1 + strat_ret).prod() - 1
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) \
                  if strat_ret.std() > 0 else np.nan
        n_switches = (filtered.diff() != 0).sum()
        pct_in_market = position.mean()

        rows.append({
            "min_days": min_days,
            "total_return_%": round(total * 100, 2),
            "sharpe": round(sharpe, 3),
            "regime_switches": int(n_switches),
            "pct_in_market_%": round(pct_in_market * 100, 1),
        })

    bh_total = (1 + daily_ret).prod() - 1
    bh_sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    rows.append({
        "min_days": "buy_hold",
        "total_return_%": round(bh_total * 100, 2),
        "sharpe": round(bh_sharpe, 3),
        "regime_switches": 0,
        "pct_in_market_%": 100.0,
    })

    return pd.DataFrame(rows).set_index("min_days")


# ==========================================================================
# 2. REGIME SEVERITY SCORE
# ==========================================================================

def compute_severity_score(proba_df: pd.DataFrame,
                            features: pd.DataFrame,
                            momentum_col: str = "momentum",
                            drawdown_col: str = "drawdown",
                            weights: dict = None) -> pd.Series:
    """
    Computes a single score in [0, 1] for each day representing
    "how bullish is today" -- 1 = extremely bullish, 0 = extremely bearish.

    Combines four signals:

      regime_score   : weighted average of regime index by posterior proba.
      confidence     : max(posterior probabilities).
      drawdown_score : 1 + drawdown (drawdown is <= 0, so this is in [0,1]).
      momentum_score : sigmoid of the momentum feature (price vs MA).

    Final score = weighted average of the four. Default weights:
      regime_score   : 0.40
      confidence     : 0.20
      drawdown_score : 0.25
      momentum_score : 0.15
    """
    if weights is None:
        weights = {
            "regime":     0.40,
            "confidence": 0.20,
            "drawdown":   0.25,
            "momentum":   0.15,
        }
    assert abs(sum(weights.values()) - 1.0) < 1e-6, \
        "weights must sum to 1.0"

    common = proba_df.index.intersection(features.index)
    proba  = proba_df.loc[common]
    feat   = features.loc[common]

    n_states = proba.shape[1]

    linear_w = np.linspace(0, 1, n_states)
    regime_score = pd.Series(proba.values @ linear_w, index=common)

    raw_conf = proba.max(axis=1)
    min_conf = 1.0 / n_states
    confidence = (raw_conf - min_conf) / (1.0 - min_conf)
    confidence_component = regime_score + confidence * (regime_score - 0.5)
    confidence_component = confidence_component.clip(0, 1)

    if drawdown_col in feat.columns:
        drawdown_score = (1.0 + feat[drawdown_col]).clip(0, 1)
    else:
        print(f"WARNING: '{drawdown_col}' not in features -- "
              f"using 0.5 (neutral) for drawdown component.")
        drawdown_score = pd.Series(0.5, index=common)

    if momentum_col in feat.columns:
        momentum_score = 1.0 / (1.0 + np.exp(-feat[momentum_col] * 20))
    else:
        print(f"WARNING: '{momentum_col}' not in features -- "
              f"using 0.5 (neutral) for momentum component.")
        momentum_score = pd.Series(0.5, index=common)

    severity = (
        weights["regime"]     * regime_score +
        weights["confidence"] * confidence_component +
        weights["drawdown"]   * drawdown_score +
        weights["momentum"]   * momentum_score
    ).clip(0, 1)

    severity.name = "severity_score"
    return severity


# ==========================================================================
# 3. GRADED POSITION SIZING V2 (uses severity score)
# ==========================================================================

def compute_position_size_v2(severity: pd.Series,
                              variant: str = "moderate",
                              min_position: float = 0.0,
                              max_position: float = 1.0) -> pd.Series:
    """
    Converts the severity score (0=bearish, 1=bullish) into a position
    size using one of three variants.

    variant="conservative" : position = severity ** 2  (convex, cautious)
    variant="moderate"     : position = severity       (linear)
    variant="aggressive"   : position = severity ** 0.5 (concave, bold)
    """
    s = severity.clip(0, 1)

    if variant == "conservative":
        raw = s ** 2
    elif variant == "moderate":
        raw = s
    elif variant == "aggressive":
        raw = s ** 0.5
    else:
        raise ValueError(f"variant must be conservative/moderate/aggressive, "
                         f"got '{variant}'")

    position = min_position + raw * (max_position - min_position)
    position = position.clip(min_position, max_position)
    position.name = f"position_{variant}"
    return position


def compare_variants(prices: pd.Series, severity: pd.Series,
                     min_position: float = 0.0) -> pd.DataFrame:
    """
    Backtests all three position sizing variants side by side.
    """
    aligned = prices.loc[severity.index]
    daily_ret = aligned.pct_change().fillna(0)

    rows = []
    for variant in ["conservative", "moderate", "aggressive"]:
        pos = compute_position_size_v2(severity, variant=variant,
                                        min_position=min_position)
        strat_ret = daily_ret * pos.shift(1).fillna(0)
        total = (1 + strat_ret).prod() - 1
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) \
                  if strat_ret.std() > 0 else np.nan
        max_dd = (strat_ret + 1).cumprod()
        max_dd = ((max_dd / max_dd.cummax()) - 1).min()
        rows.append({
            "variant":         variant,
            "total_return_%":  round(total * 100, 2),
            "sharpe":          round(sharpe, 3),
            "max_drawdown_%":  round(max_dd * 100, 2),
            "avg_position_%":  round(pos.mean() * 100, 1),
        })

    bh_total = (1 + daily_ret).prod() - 1
    bh_sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    bh_dd = ((1 + daily_ret).cumprod() /
              (1 + daily_ret).cumprod().cummax() - 1).min()
    rows.append({
        "variant":        "buy_hold",
        "total_return_%": round(bh_total * 100, 2),
        "sharpe":         round(bh_sharpe, 3),
        "max_drawdown_%": round(bh_dd * 100, 2),
        "avg_position_%": 100.0,
    })

    return pd.DataFrame(rows).set_index("variant")


# ==========================================================================
# Example usage
# ==========================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from regime_hmm import (resolve_ticker, load_prices_with_fallback,
                             make_output_dirs, build_features, select_n_states,
                             walk_forward_regimes)
    from regime_sizing import walk_forward_proba

    stock = input("Enter stock: ")
    ticker, label, fallback = resolve_ticker(stock)
    paths = make_output_dirs(label)

    prices, _ = load_prices_with_fallback(ticker, fallback)
    features = build_features(prices)

    # FIX: was max_steps=4 (wrong kwarg), now max_states=4
    # FIX: pass all feature columns, not a 2-column slice
    _, k = select_n_states(features.values, max_states=4)
    proba_df = walk_forward_proba(features, n_states=k)

    # Persistence filter comparison
    raw_regimes = walk_forward_regimes(features, n_states=k)
    print("\n=== Persistence Filter Impact ===")
    print(compare_filter_impact(prices, raw_regimes, [1, 2, 3, 5]))

    # Severity score
    severity = compute_severity_score(proba_df, features)

    # Position sizing variant comparison
    min_pos = float(input("min_position (e.g. 0.1): ") or 0.0)
    print("\n=== Position Sizing Variant Comparison ===")
    print(compare_variants(prices, severity, min_position=min_pos))