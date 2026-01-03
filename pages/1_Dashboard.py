"""
Dashboard Page - File upload, summary metrics, and quick insights.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG
from utils.data_processing import load_and_process_data, detect_file_format
from utils.calculations import calculate_expected_value, calculate_strategy_metrics
from utils.visualizations import create_ev_bar_chart, create_pie_chart
from utils.filters import initialize_session_state, render_sidebar_filters, apply_filters

st.set_page_config(
    page_title=f"Dashboard - {PAGE_CONFIG['page_title']}",
    page_icon=PAGE_CONFIG['page_icon'],
    layout=PAGE_CONFIG['layout'],
)

st.title("Dashboard")

# File upload section
st.header("Data Upload")

uploaded_file = st.file_uploader(
    "Upload Options Omega CSV",
    type=['csv'],
    help="Export your trades from Options Omega in CSV format"
)

if uploaded_file is not None:
    # Detect file format
    file_format = detect_file_format(uploaded_file)

    if file_format == 'unknown':
        st.error("Unrecognized file format. Please upload an Options Omega CSV export.")
    else:
        # Load and process data
        with st.spinner('Loading and processing data...'):
            df = load_and_process_data(uploaded_file)
            st.session_state.df = df
            initialize_session_state(df)

        st.success(f"Loaded {len(df):,} trades from {df['Date Opened'].min().date()} to {df['Date Opened'].max().date()}")

# Check if data is loaded
if 'df' not in st.session_state:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

df = st.session_state.df

# Sidebar filters
filters = render_sidebar_filters(df)
filtered_df = apply_filters(df, filters)

if len(filtered_df) == 0:
    st.warning("No trades match the current filters. Please adjust your filter settings.")
    st.stop()

# Summary metrics
st.header("Summary Metrics")

# Calculate overall metrics
total_trades = len(filtered_df)
win_rate = (filtered_df['P/L'] > 0).mean() * 100
total_pnl = filtered_df['P/L'].sum()
overall_ev = calculate_expected_value(filtered_df)

# Get best and worst strategies
metrics_df = calculate_strategy_metrics(filtered_df)
best_strategy = metrics_df.iloc[0]['Strategy'] if len(metrics_df) > 0 else "N/A"
best_ev = metrics_df.iloc[0]['Expected Value'] if len(metrics_df) > 0 else 0
worst_strategy = metrics_df.iloc[-1]['Strategy'] if len(metrics_df) > 0 else "N/A"
worst_ev = metrics_df.iloc[-1]['Expected Value'] if len(metrics_df) > 0 else 0

# Display metrics in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Trades", f"{total_trades:,}")
    st.metric("Overall Win Rate", f"{win_rate:.1f}%")

with col2:
    st.metric("Total P/L", f"${total_pnl:,.2f}", delta=None)
    st.metric("Overall Expected Value", f"${overall_ev:.2f}")

with col3:
    st.metric(
        "Best Strategy",
        best_strategy,
        delta=f"${best_ev:.2f} EV"
    )
    st.metric(
        "Worst Strategy",
        worst_strategy,
        delta=f"${worst_ev:.2f} EV"
    )

# Strategy performance table
st.header("Strategy Performance")

# Format the metrics dataframe for display
display_df = metrics_df.copy()
display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
display_df['Avg Win'] = display_df['Avg Win'].apply(lambda x: f"${x:.2f}")
display_df['Avg Loss'] = display_df['Avg Loss'].apply(lambda x: f"${x:.2f}")
display_df['Total P/L'] = display_df['Total P/L'].apply(lambda x: f"${x:,.2f}")
display_df['Expected Value'] = display_df['Expected Value'].apply(lambda x: f"${x:.2f}")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

# Quick visualizations
st.header("Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Trade Distribution")
    pie_chart = create_pie_chart(filtered_df)
    st.plotly_chart(pie_chart, use_container_width=True)

with col2:
    st.subheader("Expected Value by Strategy")
    bar_chart = create_ev_bar_chart(metrics_df)
    st.plotly_chart(bar_chart, use_container_width=True)

# Additional insights
st.header("Quick Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Trade Outcomes")
    outcome_counts = filtered_df['Reason For Close'].value_counts()
    st.dataframe(
        outcome_counts.reset_index().rename(columns={'index': 'Reason', 'Reason For Close': 'Count'}),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.subheader("VIX Regime Distribution")
    vix_counts = filtered_df['VIX_Regime'].value_counts()
    st.dataframe(
        vix_counts.reset_index().rename(columns={'index': 'VIX Regime', 'VIX_Regime': 'Count'}),
        use_container_width=True,
        hide_index=True,
    )

# Date range summary
st.subheader("Data Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Strategies", len(filtered_df['Strategy'].unique()))

with col2:
    st.metric("Date Range", f"{(filtered_df['Date Opened'].max() - filtered_df['Date Opened'].min()).days} days")

with col3:
    avg_trades_per_day = len(filtered_df) / max((filtered_df['Date Opened'].max() - filtered_df['Date Opened'].min()).days, 1)
    st.metric("Avg Trades/Day", f"{avg_trades_per_day:.1f}")

with col4:
    avg_duration = filtered_df['Duration_Days'].mean()
    st.metric("Avg Trade Duration", f"{avg_duration:.1f} days")
