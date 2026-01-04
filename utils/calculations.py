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
