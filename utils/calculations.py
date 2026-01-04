"""
Calculation utilities for Expected Value, metrics, and rolling calculations.
"""

import pandas as pd
import numpy as np
import streamlit as st
from config import MIN_TRADES_FOR_STATS


def get_quarter(date):
    """
    Assign calendar quarter based on month.
    """
    month = date.month
    if month <= 3:
        return 'Q1'
    elif month <= 6:
        return 'Q2'
    elif month <= 9:
        return 'Q3'
    else:
        return 'Q4'


def categorize_vix(vix_value):
    """
    Categorize VIX into regime buckets.
    """
    if pd.isna(vix_value):
        return 'Unknown'
    if vix_value < 15:
        return 'Low (<15)'
    elif vix_value < 20:
        return 'Normal (15-20)'
    elif vix_value < 25:
        return 'Elevated (20-25)'
    elif vix_value < 30:
        return 'High (25-30)'
    else:
        return 'Extreme (>30)'


def calculate_expected_value(trades_df):
    """
    Calculate Expected Value for a set of trades.
    EV = (Win Rate × Average Win) - (Loss Rate × Average Loss)

    Returns EV in dollars.
    """
    if len(trades_df) == 0:
        return 0

    wins = trades_df[trades_df['P/L'] > 0]
    losses = trades_df[trades_df['P/L'] <= 0]

    total_trades = len(trades_df)

    win_rate = len(wins) / total_trades
    loss_rate = len(losses) / total_trades

    avg_win = wins['P/L'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['P/L'].mean()) if len(losses) > 0 else 0

    expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)
    return expected_value


def calculate_rolling_ev(series):
    """
    Calculate EV for a rolling window of P/L values.
    """
    if len(series) == 0:
        return 0

    wins = series[series > 0]
    losses = series[series <= 0]

    total = len(series)
    win_rate = len(wins) / total if total > 0 else 0
    loss_rate = len(losses) / total if total > 0 else 0

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    return (win_rate * avg_win) - (loss_rate * avg_loss)


@st.cache_data
def calculate_strategy_metrics(df):
    """
    Calculate comprehensive metrics for each strategy.
    """
    metrics = []
    for strategy in df['Strategy'].unique():
        strategy_df = df[df['Strategy'] == strategy]

        wins = strategy_df[strategy_df['P/L'] > 0]
        losses = strategy_df[strategy_df['P/L'] <= 0]

        metrics.append({
            'Strategy': strategy,
            'Trades': len(strategy_df),
            'Win Rate': (len(wins) / len(strategy_df) * 100) if len(strategy_df) > 0 else 0,
            'Avg Win': wins['P/L'].mean() if len(wins) > 0 else 0,
            'Avg Loss': losses['P/L'].mean() if len(losses) > 0 else 0,
            'Total P/L': strategy_df['P/L'].sum(),
            'Expected Value': calculate_expected_value(strategy_df),
        })

    return pd.DataFrame(metrics).sort_values('Expected Value', ascending=False)


def calculate_quarterly_ev(df, strategy, quarter):
    """
    Calculate EV for a specific strategy in a specific quarter.
    """
    filtered = df[(df['Strategy'] == strategy) & (df['Quarter'] == quarter)]
    if len(filtered) < MIN_TRADES_FOR_STATS:
        return np.nan
    return calculate_expected_value(filtered)


@st.cache_data
def calculate_quarterly_matrix(df):
    """
    Create a pivot table of Strategy × Quarter with EV values.
    """
    strategies = df['Strategy'].unique()
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']

    matrix_data = []
    for strategy in strategies:
        row = {'Strategy': strategy}
        for quarter in quarters:
            row[quarter] = calculate_quarterly_ev(df, strategy, quarter)
        matrix_data.append(row)

    return pd.DataFrame(matrix_data).set_index('Strategy')


def calculate_vix_regime_ev(df, strategy, regime):
    """
    Calculate EV for a specific strategy in a specific VIX regime.
    """
    filtered = df[(df['Strategy'] == strategy) & (df['VIX_Regime'] == regime)]
    if len(filtered) < MIN_TRADES_FOR_STATS:
        return np.nan
    return calculate_expected_value(filtered)


@st.cache_data
def calculate_vix_regime_matrix(df):
    """
    Create a pivot table of Strategy × VIX Regime with EV values.
    """
    strategies = df['Strategy'].unique()
    regimes = ['Low (<15)', 'Normal (15-20)', 'Elevated (20-25)', 'High (25-30)', 'Extreme (>30)']

    matrix_data = []
    for strategy in strategies:
        row = {'Strategy': strategy}
        for regime in regimes:
            row[regime] = calculate_vix_regime_ev(df, strategy, regime)
        matrix_data.append(row)

    return pd.DataFrame(matrix_data).set_index('Strategy')


def calculate_correlation_metrics(df, strategy):
    """
    Calculate correlation between market factors and P/L for a strategy.
    """
    strategy_df = df[df['Strategy'] == strategy].copy()

    if len(strategy_df) < MIN_TRADES_FOR_STATS:
        return {
            'VIX Correlation': np.nan,
            'Gap Correlation': np.nan,
            'Movement Correlation': np.nan,
        }

    return {
        'VIX Correlation': strategy_df['Opening VIX'].corr(strategy_df['P/L']),
        'Gap Correlation': strategy_df['Gap'].corr(strategy_df['P/L']),
        'Movement Correlation': strategy_df['Movement'].corr(strategy_df['P/L']),
    }


def calculate_sharpe_like_ratio(df):
    """
    Calculate a Sharpe-like ratio for strategy comparison.
    Uses EV as return and standard deviation of P/L as risk.
    """
    if len(df) < MIN_TRADES_FOR_STATS:
        return np.nan

    ev = calculate_expected_value(df)
    std = df['P/L'].std()

    if std == 0:
        return np.nan

    return ev / std


def calculate_trend_metrics(df, strategy, lookback_days=90):
    """
    Calculate trend direction and strength for a strategy.
    """
    strategy_df = df[df['Strategy'] == strategy].copy()
    strategy_df = strategy_df.sort_values('Date Opened')

    if len(strategy_df) < MIN_TRADES_FOR_STATS:
        return {
            'trend_direction': 'neutral',
            'trend_strength': 0,
            'volatility': np.nan,
        }

    # Get recent data
    cutoff_date = strategy_df['Date Opened'].max() - pd.Timedelta(days=lookback_days)
    recent = strategy_df[strategy_df['Date Opened'] >= cutoff_date]

    if len(recent) < MIN_TRADES_FOR_STATS:
        return {
            'trend_direction': 'neutral',
            'trend_strength': 0,
            'volatility': np.nan,
        }

    # Calculate rolling EV for trend
    rolling_ev = recent['P/L'].expanding(min_periods=5).apply(calculate_rolling_ev)

    if len(rolling_ev) < 2:
        return {
            'trend_direction': 'neutral',
            'trend_strength': 0,
            'volatility': np.nan,
        }

    # Trend direction based on slope
    slope = (rolling_ev.iloc[-1] - rolling_ev.iloc[0]) / len(rolling_ev)

    if slope > 0.5:
        trend_direction = 'up'
    elif slope < -0.5:
        trend_direction = 'down'
    else:
        trend_direction = 'neutral'

    # Trend strength (percentage change)
    if rolling_ev.iloc[0] != 0:
        trend_strength = ((rolling_ev.iloc[-1] - rolling_ev.iloc[0]) / abs(rolling_ev.iloc[0])) * 100
    else:
        trend_strength = 0

    return {
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'volatility': rolling_ev.std(),
    }


def calculate_max_drawdown(cumulative_pnl):
    """
    Calculate maximum drawdown from cumulative P/L series.
    """
    if len(cumulative_pnl) == 0:
        return 0

    peak = cumulative_pnl.expanding(min_periods=1).max()
    drawdown = cumulative_pnl - peak
    return drawdown.min()


def calculate_portfolio_metrics(df, selected_strategies):
    """
    Calculate combined metrics for a portfolio of strategies.
    """
    portfolio_df = df[df['Strategy'].isin(selected_strategies)].copy()

    if len(portfolio_df) == 0:
        return {
            'total_trades': 0,
            'combined_ev': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
        }

    portfolio_df = portfolio_df.sort_values('Date Closed')
    portfolio_df['Cumulative P/L'] = portfolio_df['P/L'].cumsum()

    return {
        'total_trades': len(portfolio_df),
        'combined_ev': calculate_expected_value(portfolio_df),
        'win_rate': (portfolio_df['P/L'] > 0).mean() * 100,
        'total_pnl': portfolio_df['P/L'].sum(),
        'max_drawdown': calculate_max_drawdown(portfolio_df['Cumulative P/L']),
    }


def calculate_strategy_correlation_matrix(df, strategies):
    """
    Calculate correlation matrix between strategies based on daily P/L.
    """
    # Aggregate daily P/L for each strategy
    daily_pnl = df.pivot_table(
        index='Date Closed',
        columns='Strategy',
        values='P/L',
        aggfunc='sum',
        fill_value=0
    )

    # Filter to selected strategies
    available = [s for s in strategies if s in daily_pnl.columns]
    if len(available) < 2:
        return pd.DataFrame()

    return daily_pnl[available].corr()


def get_best_strategies_by_quarter(df, top_n=3):
    """
    Get the best performing strategies for each quarter.
    """
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    recommendations = {}

    for quarter in quarters:
        quarter_df = df[df['Quarter'] == quarter]
        strategy_evs = []

        for strategy in df['Strategy'].unique():
            strategy_quarter_df = quarter_df[quarter_df['Strategy'] == strategy]
            if len(strategy_quarter_df) >= MIN_TRADES_FOR_STATS:
                ev = calculate_expected_value(strategy_quarter_df)
                strategy_evs.append({
                    'strategy': strategy,
                    'ev': ev,
                    'trades': len(strategy_quarter_df),
                })

        # Sort by EV and get top N
        strategy_evs.sort(key=lambda x: x['ev'], reverse=True)
        recommendations[quarter] = strategy_evs[:top_n]

    return recommendations


def calculate_trading_days(start_date, end_date):
    """
    Calculate the number of trading days between two dates.
    Uses pandas business day calendar (excludes weekends).
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
    return len(pd.bdate_range(start=start_date, end=end_date))


def calculate_ev_confidence_interval(trades_df, confidence=0.95):
    """
    Calculate confidence interval for Expected Value using bootstrap-like approach.
    Uses standard error of the mean P/L as approximation.

    Returns: (ev, margin_of_error, lower_bound, upper_bound)
    """
    from scipy import stats

    if len(trades_df) < MIN_TRADES_FOR_STATS:
        return (0, np.nan, np.nan, np.nan)

    ev = calculate_expected_value(trades_df)
    pnl = trades_df['P/L'].values

    # Standard error of the mean
    std_error = stats.sem(pnl)

    # t-critical value for confidence level
    t_critical = stats.t.ppf((1 + confidence) / 2, len(pnl) - 1)

    margin_of_error = t_critical * std_error

    return (ev, margin_of_error, ev - margin_of_error, ev + margin_of_error)


def calculate_coefficient_of_variation(trades_df):
    """
    Calculate Coefficient of Variation (CV) for P/L.
    CV = Standard Deviation / |Mean|

    Returns CV as a positive number (or nan if mean is 0).
    """
    if len(trades_df) < MIN_TRADES_FOR_STATS:
        return np.nan

    pnl = trades_df['P/L']
    mean_pnl = pnl.mean()
    std_pnl = pnl.std()

    if mean_pnl == 0:
        return np.nan

    return abs(std_pnl / mean_pnl)


def interpret_cv(cv):
    """
    Interpret Coefficient of Variation for trading context.

    Returns: (label, color, description)
    """
    if pd.isna(cv):
        return ('Unknown', 'gray', 'Insufficient data')

    if cv < 0.5:
        return ('Consistent', 'green', 'EV is reliable - outcomes cluster tightly')
    elif cv < 1.0:
        return ('Moderate', 'blue', 'Some variability - EV is reasonable estimate')
    elif cv < 2.0:
        return ('Variable', 'orange', 'Wide spread - EV is a rough guide')
    else:
        return ('Highly Variable', 'red', 'Very unpredictable - treat EV with caution')


def calculate_ev_reliability_metrics(trades_df, start_date=None, end_date=None):
    """
    Calculate comprehensive EV reliability metrics for a set of trades.

    Returns dict with:
    - trade_count: number of trades
    - trading_days: available trading days in period
    - trade_density: trades as % of trading days
    - ev: expected value
    - std_dev: standard deviation of P/L
    - cv: coefficient of variation
    - cv_interpretation: (label, color, description)
    - confidence_interval: (ev, margin, lower, upper)
    - is_reliable: boolean indicating if sample is statistically reliable
    """
    n_trades = len(trades_df)

    # Determine date range
    if start_date is None and len(trades_df) > 0:
        start_date = trades_df['Date Opened'].min()
    if end_date is None and len(trades_df) > 0:
        end_date = trades_df['Date Opened'].max()

    trading_days = calculate_trading_days(start_date, end_date) if start_date and end_date else 0

    # Trade density (% of trading days with trades)
    trade_density = (n_trades / trading_days * 100) if trading_days > 0 else 0

    if n_trades < MIN_TRADES_FOR_STATS:
        return {
            'trade_count': n_trades,
            'trading_days': trading_days,
            'trade_density': trade_density,
            'ev': 0,
            'std_dev': np.nan,
            'cv': np.nan,
            'cv_interpretation': interpret_cv(np.nan),
            'confidence_interval': (0, np.nan, np.nan, np.nan),
            'is_reliable': False,
        }

    ev = calculate_expected_value(trades_df)
    std_dev = trades_df['P/L'].std()
    cv = calculate_coefficient_of_variation(trades_df)
    confidence_interval = calculate_ev_confidence_interval(trades_df)

    # Reliability check: sufficient trades AND reasonable CV
    is_reliable = n_trades >= 30 and (pd.isna(cv) or cv < 2.0)

    return {
        'trade_count': n_trades,
        'trading_days': trading_days,
        'trade_density': trade_density,
        'ev': ev,
        'std_dev': std_dev,
        'cv': cv,
        'cv_interpretation': interpret_cv(cv),
        'confidence_interval': confidence_interval,
        'is_reliable': is_reliable,
    }


def calculate_ev_sensitivity(trades_df, exclude_n=1):
    """
    Calculate how sensitive EV is to outliers by removing best/worst trades.

    Returns: (base_ev, ev_without_best, ev_without_worst, sensitivity_score)
    """
    if len(trades_df) < MIN_TRADES_FOR_STATS + (2 * exclude_n):
        return (np.nan, np.nan, np.nan, np.nan)

    base_ev = calculate_expected_value(trades_df)

    # Sort by P/L
    sorted_pnl = trades_df.sort_values('P/L')

    # Remove worst N trades
    without_worst = sorted_pnl.iloc[exclude_n:]
    ev_without_worst = calculate_expected_value(without_worst)

    # Remove best N trades
    without_best = sorted_pnl.iloc[:-exclude_n]
    ev_without_best = calculate_expected_value(without_best)

    # Sensitivity score: average absolute change
    if base_ev != 0:
        sensitivity = (abs(ev_without_best - base_ev) + abs(ev_without_worst - base_ev)) / (2 * abs(base_ev)) * 100
    else:
        sensitivity = 0

    return (base_ev, ev_without_best, ev_without_worst, sensitivity)


def interpret_ev_reliability(cv, sensitivity_score, trade_count):
    """
    Generate a synthesized interpretation of EV reliability based on
    the combination of CV, sensitivity, and sample size.

    Returns: dict with:
    - verdict: overall reliability verdict
    - icon: emoji indicator
    - color: color for display (green/blue/orange/red)
    - summary: one-line summary
    - explanation: detailed explanation
    - implications: list of practical implications
    - confidence_level: 1-5 scale
    """

    # Handle missing data
    if pd.isna(cv) or pd.isna(sensitivity_score):
        return {
            'verdict': 'Insufficient Data',
            'icon': '❓',
            'color': 'gray',
            'summary': 'Not enough data to assess reliability.',
            'explanation': 'Need more trades to calculate meaningful statistics.',
            'implications': ['Gather more trade data before relying on EV'],
            'confidence_level': 0,
        }

    # Categorize CV
    if cv < 0.5:
        cv_level = 'low'
    elif cv < 1.0:
        cv_level = 'moderate'
    elif cv < 2.0:
        cv_level = 'high'
    else:
        cv_level = 'very_high'

    # Categorize sensitivity
    if sensitivity_score < 10:
        sens_level = 'low'
    elif sensitivity_score < 25:
        sens_level = 'moderate'
    else:
        sens_level = 'high'

    # Sample size factor
    if trade_count >= 100:
        sample_quality = 'excellent'
    elif trade_count >= 50:
        sample_quality = 'good'
    elif trade_count >= 30:
        sample_quality = 'adequate'
    else:
        sample_quality = 'limited'

    # Matrix of interpretations based on CV and Sensitivity combinations
    interpretations = {
        # Low CV combinations
        ('low', 'low'): {
            'verdict': 'Highly Reliable',
            'icon': '✅',
            'color': 'green',
            'summary': 'EV is highly trustworthy. Consistent outcomes with no outlier influence.',
            'explanation': 'This strategy produces tight, predictable outcomes. The low variability (CV) means trades cluster around the expected value, and low sensitivity confirms no single trade is skewing the results.',
            'implications': [
                'EV is a strong predictor of future performance',
                'Can size positions with higher confidence',
                'Strategy behaves consistently across trades',
            ],
            'confidence_level': 5,
        },
        ('low', 'moderate'): {
            'verdict': 'Reliable',
            'icon': '✅',
            'color': 'green',
            'summary': 'EV is reliable, though a few trades have moderate impact.',
            'explanation': 'Outcomes are generally consistent, but some individual trades have noticeable influence on the average. This is normal for strategies with occasional larger wins or losses.',
            'implications': [
                'EV is a good estimate of expected performance',
                'Monitor for trades that deviate significantly',
                'Consider if outlier trades are repeatable or flukes',
            ],
            'confidence_level': 4,
        },
        ('low', 'high'): {
            'verdict': 'Caution: Outlier-Dependent',
            'icon': '⚠️',
            'color': 'orange',
            'summary': 'Normally consistent, but EV is heavily influenced by specific trades.',
            'explanation': 'Most trades are consistent, but one or a few extreme trades are significantly impacting the EV. This could be a few big wins masking a mediocre strategy, or big losses dragging down an otherwise good one.',
            'implications': [
                'Investigate the outlier trades - are they repeatable?',
                'EV may not reflect typical trade outcomes',
                'Consider the median P/L as an alternative measure',
            ],
            'confidence_level': 2,
        },

        # Moderate CV combinations
        ('moderate', 'low'): {
            'verdict': 'Reliable with Normal Variance',
            'icon': '👍',
            'color': 'blue',
            'summary': 'EV is trustworthy. Variance is evenly distributed across trades.',
            'explanation': 'There is meaningful variation in trade outcomes, but it is spread evenly - no single trade dominates. This is healthy variance that should average out over time.',
            'implications': [
                'EV is a reasonable long-term expectation',
                'Expect moderate swings in short-term results',
                'Larger sample sizes will converge to EV',
            ],
            'confidence_level': 4,
        },
        ('moderate', 'moderate'): {
            'verdict': 'Moderately Reliable',
            'icon': '👍',
            'color': 'blue',
            'summary': 'EV is a reasonable estimate with typical trading variance.',
            'explanation': 'Both variance and outlier influence are at moderate levels. This is common for most trading strategies. The EV gives a reasonable expectation, but individual results will vary.',
            'implications': [
                'Use EV as a guide, not a guarantee',
                'Results will vary - plan for drawdowns',
                'More trades will improve confidence',
            ],
            'confidence_level': 3,
        },
        ('moderate', 'high'): {
            'verdict': 'Outlier-Influenced',
            'icon': '⚠️',
            'color': 'orange',
            'summary': 'EV is skewed by specific trades. Dig deeper before trusting it.',
            'explanation': 'A few trades are disproportionately affecting the EV. The headline number may not represent typical performance. Examine what drove those outlier trades.',
            'implications': [
                'Review the best and worst trades individually',
                'Ask: would those trades happen again?',
                'Consider reporting EV with and without outliers',
            ],
            'confidence_level': 2,
        },

        # High CV combinations
        ('high', 'low'): {
            'verdict': 'Distributed Variance',
            'icon': '📊',
            'color': 'blue',
            'summary': 'EV is trustworthy, but expect a volatile ride. Variance is real but evenly spread.',
            'explanation': 'Individual trades vary widely from the average, but this variance is distributed across many trades - not caused by outliers. The EV is a fair long-term estimate, but short-term results will swing significantly.',
            'implications': [
                'Trust the EV as a long-term average',
                'Prepare for significant trade-to-trade swings',
                'Position sizing and risk management are critical',
                'Need many trades to realize the expected value',
            ],
            'confidence_level': 3,
        },
        ('high', 'moderate'): {
            'verdict': 'High Variance Strategy',
            'icon': '📊',
            'color': 'orange',
            'summary': 'Volatile strategy with some outlier influence. Use EV cautiously.',
            'explanation': 'This strategy has high variance, and some of that is driven by specific outlier trades. The EV is a rough guide at best. Performance will be choppy and somewhat unpredictable.',
            'implications': [
                'EV is a rough estimate only',
                'High variance + outlier influence = less predictable',
                'Consider smaller position sizes',
                'May need to evaluate if variance is acceptable',
            ],
            'confidence_level': 2,
        },
        ('high', 'high'): {
            'verdict': 'Unreliable: Outlier-Driven',
            'icon': '🚨',
            'color': 'red',
            'summary': 'EV is not reliable. Results are volatile AND driven by outliers.',
            'explanation': 'High variance combined with high sensitivity to outliers means the EV is heavily influenced by a few extreme trades. The number you see may not represent what to expect going forward.',
            'implications': [
                'Do not rely on the EV for decision-making',
                'Investigate the outlier trades thoroughly',
                'Consider if this strategy has too much randomness',
                'May need fundamental changes or more data',
            ],
            'confidence_level': 1,
        },

        # Very high CV combinations
        ('very_high', 'low'): {
            'verdict': 'Extreme but Distributed Variance',
            'icon': '📊',
            'color': 'orange',
            'summary': 'Very high variance, but evenly distributed. EV is valid but highly uncertain.',
            'explanation': 'Trade outcomes vary extremely widely, but no single trade dominates. This could be a legitimate high-variance strategy (like lottery-style trades). The EV may be accurate long-term, but convergence will be very slow.',
            'implications': [
                'EV may be correct but will take many trades to realize',
                'Expect extreme swings in performance',
                'Only allocate capital you can afford to see swing wildly',
                'Consider if this variance level is acceptable for you',
            ],
            'confidence_level': 2,
        },
        ('very_high', 'moderate'): {
            'verdict': 'Very High Variance',
            'icon': '🚨',
            'color': 'red',
            'summary': 'Extreme variance with outlier influence. EV is unreliable.',
            'explanation': 'The strategy produces wildly different outcomes, and some outliers are influencing the average. This level of unpredictability makes the EV difficult to trust.',
            'implications': [
                'EV should not be primary decision factor',
                'Evaluate if the strategy mechanics are sound',
                'Consider if market conditions caused the variance',
                'May need to revisit strategy rules',
            ],
            'confidence_level': 1,
        },
        ('very_high', 'high'): {
            'verdict': 'Highly Unreliable',
            'icon': '🚨',
            'color': 'red',
            'summary': 'Do not trust this EV. Extreme variance and outlier-driven.',
            'explanation': 'This is the least reliable combination. Extreme variance means outcomes are all over the map, and high sensitivity means a few trades are dominating the average. The EV number is essentially meaningless for prediction.',
            'implications': [
                'This EV should not inform trading decisions',
                'Fundamental review of strategy is needed',
                'Consider if there is a systematic issue',
                'More data unlikely to help if strategy is flawed',
            ],
            'confidence_level': 0,
        },
    }

    # Get the interpretation
    key = (cv_level, sens_level)
    result = interpretations.get(key, interpretations[('moderate', 'moderate')])

    # Adjust confidence based on sample size
    sample_adjustments = {
        'excellent': 0,
        'good': 0,
        'adequate': -1,
        'limited': -1,
    }

    result = result.copy()  # Don't modify the original
    result['confidence_level'] = max(0, result['confidence_level'] + sample_adjustments[sample_quality])

    # Add sample size note if limited
    if sample_quality in ('adequate', 'limited'):
        result['implications'] = result['implications'] + [
            f'Sample size ({trade_count} trades) is limited - gather more data for higher confidence'
        ]

    return result
