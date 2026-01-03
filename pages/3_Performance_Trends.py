"""
Performance Trends Page - Rolling performance analysis and trend identification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, ROLLING_WINDOWS, COLORS, MIN_TRADES_FOR_STATS
from utils.calculations import (
    calculate_rolling_ev,
    calculate_trend_metrics,
    calculate_expected_value,
)
from utils.visualizations import (
    create_rolling_ev_chart,
    create_equity_curve,
    create_drawdown_chart,
)
from utils.filters import render_sidebar_filters, apply_filters

st.set_page_config(
    page_title=f"Performance Trends - {PAGE_CONFIG['page_title']}",
    page_icon=PAGE_CONFIG['page_icon'],
    layout=PAGE_CONFIG['layout'],
)

st.title("Performance Trends")

# Check if data is loaded
if 'df' not in st.session_state:
    st.warning("No data loaded. Please go to the Dashboard to upload your trading data.")
    st.stop()

df = st.session_state.df

# Sidebar filters
filters = render_sidebar_filters(df)
filtered_df = apply_filters(df, filters)

if len(filtered_df) == 0:
    st.warning("No trades match the current filters.")
    st.stop()

# Rolling window selector
st.header("Rolling Expected Value Analysis")

col1, col2 = st.columns([1, 3])

with col1:
    window_size = st.radio(
        "Rolling Window",
        options=ROLLING_WINDOWS,
        format_func=lambda x: f"{x} Days",
        horizontal=False,
    )

# Strategy selector
all_strategies = sorted(filtered_df['Strategy'].unique())
selected_strategies = st.multiselect(
    "Select strategies to display (max 7 for readability)",
    options=all_strategies,
    default=all_strategies[:5],
    max_selections=7,
)

if not selected_strategies:
    st.info("Please select at least one strategy to view trends.")
    st.stop()

# Rolling EV line chart
rolling_chart = create_rolling_ev_chart(
    filtered_df,
    selected_strategies,
    window_size=window_size,
    title=f"Rolling {window_size}-Day Expected Value",
)
st.plotly_chart(rolling_chart, use_container_width=True)

# Trend Indicators
st.header("Trend Indicators")

st.markdown("""
Trend analysis based on the last 90 days of trading data:
- **Direction**: ↑ Up, ↓ Down, → Neutral
- **Strength**: Percentage change from start to end of period
- **Volatility**: Standard deviation of rolling EV
""")

trend_data = []
for strategy in selected_strategies:
    metrics = calculate_trend_metrics(filtered_df, strategy, lookback_days=90)
    strategy_df = filtered_df[filtered_df['Strategy'] == strategy]

    trend_data.append({
        'Strategy': strategy,
        'Trades (90d)': len(strategy_df[
            strategy_df['Date Opened'] >= (strategy_df['Date Opened'].max() - pd.Timedelta(days=90))
        ]),
        'Direction': '↑' if metrics['trend_direction'] == 'up' else ('↓' if metrics['trend_direction'] == 'down' else '→'),
        'Strength': f"{metrics['trend_strength']:.1f}%",
        'Volatility': f"${metrics['volatility']:.2f}" if not np.isnan(metrics['volatility']) else "N/A",
        'Current EV': f"${calculate_expected_value(strategy_df):.2f}",
    })

trend_df = pd.DataFrame(trend_data)

# Apply color coding to direction
def color_direction(val):
    if val == '↑':
        return f'color: {COLORS["positive"]}'
    elif val == '↓':
        return f'color: {COLORS["negative"]}'
    return f'color: {COLORS["neutral"]}'

styled_trend_df = trend_df.style.applymap(
    color_direction,
    subset=['Direction']
)

st.dataframe(
    trend_df,
    use_container_width=True,
    hide_index=True,
)

# Performance Regime Analysis
st.header("Performance Regime Analysis")

st.markdown("""
Identify periods of consistent high performance, drawdowns, and recovery.
""")

# Select a single strategy for detailed analysis
detail_strategy = st.selectbox(
    "Select strategy for detailed analysis",
    options=selected_strategies,
)

strategy_df = filtered_df[filtered_df['Strategy'] == detail_strategy].copy()
strategy_df = strategy_df.sort_values('Date Closed')

if len(strategy_df) < MIN_TRADES_FOR_STATS:
    st.info(f"Not enough trades for {detail_strategy} to analyze.")
else:
    # Equity curve
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Equity Curve")
        equity_curve = create_equity_curve(strategy_df, title=f"{detail_strategy} Equity Curve")
        st.plotly_chart(equity_curve, use_container_width=True)

    with col2:
        st.subheader("Drawdown Analysis")
        drawdown_chart = create_drawdown_chart(strategy_df, title=f"{detail_strategy} Drawdown")
        st.plotly_chart(drawdown_chart, use_container_width=True)

    # Calculate regime periods
    strategy_df['Cumulative P/L'] = strategy_df['P/L'].cumsum()
    strategy_df['Rolling EV'] = strategy_df['P/L'].rolling(window=20, min_periods=10).apply(calculate_rolling_ev)

    # Identify regime periods
    high_threshold = strategy_df['Rolling EV'].quantile(0.75) if len(strategy_df) > 20 else 0
    low_threshold = 0

    strategy_df['Regime'] = 'Normal'
    strategy_df.loc[strategy_df['Rolling EV'] > high_threshold, 'Regime'] = 'High Performance'
    strategy_df.loc[strategy_df['Rolling EV'] < low_threshold, 'Regime'] = 'Drawdown'

    # Recovery detection
    in_recovery = False
    for i in range(1, len(strategy_df)):
        if (strategy_df.iloc[i-1]['Rolling EV'] < low_threshold and
            strategy_df.iloc[i]['Rolling EV'] >= low_threshold and
            strategy_df.iloc[i]['Rolling EV'] < high_threshold):
            strategy_df.iloc[i, strategy_df.columns.get_loc('Regime')] = 'Recovery'

    # Display regime summary
    st.subheader("Regime Summary")

    regime_counts = strategy_df['Regime'].value_counts()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        high_perf = regime_counts.get('High Performance', 0)
        st.metric("High Performance", f"{high_perf} trades")

    with col2:
        normal = regime_counts.get('Normal', 0)
        st.metric("Normal", f"{normal} trades")

    with col3:
        drawdown = regime_counts.get('Drawdown', 0)
        st.metric("Drawdown", f"{drawdown} trades")

    with col4:
        recovery = regime_counts.get('Recovery', 0)
        st.metric("Recovery", f"{recovery} trades")

# Strategy Comparison Over Time
st.header("Strategy Performance Timeline")

st.markdown("""
Compare how selected strategies have performed over different time periods.
""")

# Time period selector
time_periods = {
    'Last 30 Days': 30,
    'Last 90 Days': 90,
    'Last 180 Days': 180,
    'Last 365 Days': 365,
    'All Time': None,
}

selected_period = st.selectbox(
    "Select time period",
    options=list(time_periods.keys()),
)

period_days = time_periods[selected_period]

if period_days:
    cutoff_date = filtered_df['Date Opened'].max() - pd.Timedelta(days=period_days)
    period_df = filtered_df[filtered_df['Date Opened'] >= cutoff_date]
else:
    period_df = filtered_df

# Calculate metrics for each strategy in the period
period_metrics = []
for strategy in selected_strategies:
    strategy_period_df = period_df[period_df['Strategy'] == strategy]
    if len(strategy_period_df) >= MIN_TRADES_FOR_STATS:
        ev = calculate_expected_value(strategy_period_df)
        win_rate = (strategy_period_df['P/L'] > 0).mean() * 100
        total_pnl = strategy_period_df['P/L'].sum()

        period_metrics.append({
            'Strategy': strategy,
            'Trades': len(strategy_period_df),
            'Win Rate': f"{win_rate:.1f}%",
            'Total P/L': f"${total_pnl:,.2f}",
            'Expected Value': ev,
        })

if period_metrics:
    period_metrics_df = pd.DataFrame(period_metrics)
    period_metrics_df = period_metrics_df.sort_values('Expected Value', ascending=False)
    period_metrics_df['Expected Value'] = period_metrics_df['Expected Value'].apply(lambda x: f"${x:.2f}")

    st.dataframe(
        period_metrics_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(f"Not enough data for selected strategies in the {selected_period} period.")

# Improvement/Degradation Detection
st.header("Strategy Health Check")

st.markdown("""
Comparing recent performance (last 90 days) vs historical performance to identify
strategies that are improving or degrading.
""")

health_data = []
for strategy in all_strategies:
    strategy_df = filtered_df[filtered_df['Strategy'] == strategy]

    if len(strategy_df) < MIN_TRADES_FOR_STATS * 2:
        continue

    cutoff = strategy_df['Date Opened'].max() - pd.Timedelta(days=90)
    recent = strategy_df[strategy_df['Date Opened'] >= cutoff]
    historical = strategy_df[strategy_df['Date Opened'] < cutoff]

    if len(recent) >= MIN_TRADES_FOR_STATS and len(historical) >= MIN_TRADES_FOR_STATS:
        recent_ev = calculate_expected_value(recent)
        historical_ev = calculate_expected_value(historical)

        if historical_ev != 0:
            change_pct = ((recent_ev - historical_ev) / abs(historical_ev)) * 100
        else:
            change_pct = 0 if recent_ev == 0 else (100 if recent_ev > 0 else -100)

        status = 'Improving' if change_pct > 20 else ('Degrading' if change_pct < -20 else 'Stable')

        health_data.append({
            'Strategy': strategy,
            'Historical EV': f"${historical_ev:.2f}",
            'Recent EV': f"${recent_ev:.2f}",
            'Change': f"{change_pct:+.1f}%",
            'Status': status,
        })

if health_data:
    health_df = pd.DataFrame(health_data)

    # Sort by change percentage
    health_df['Change_Num'] = health_df['Change'].str.rstrip('%').astype(float)
    health_df = health_df.sort_values('Change_Num', ascending=False)
    health_df = health_df.drop('Change_Num', axis=1)

    st.dataframe(
        health_df,
        use_container_width=True,
        hide_index=True,
    )

    # Summary
    improving = len([h for h in health_data if h['Status'] == 'Improving'])
    stable = len([h for h in health_data if h['Status'] == 'Stable'])
    degrading = len([h for h in health_data if h['Status'] == 'Degrading'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Improving", improving, delta=None)
    with col2:
        st.metric("Stable", stable, delta=None)
    with col3:
        st.metric("Degrading", degrading, delta=None)
else:
    st.info("Not enough historical data to perform health check analysis.")
