"""
Portfolio Builder Page - Construct and analyze strategy portfolios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, QUARTERS, MIN_TRADES_FOR_STATS
from utils.calculations import (
    calculate_expected_value,
    calculate_strategy_metrics,
    calculate_portfolio_metrics,
    calculate_strategy_correlation_matrix,
    calculate_sharpe_like_ratio,
    calculate_max_drawdown,
    get_best_strategies_by_quarter,
    calculate_strategy_bp_requirements,
    calculate_portfolio_bp,
    simulate_portfolio_with_allocation,
)
from utils.visualizations import (
    create_equity_curve,
    create_correlation_heatmap,
    create_drawdown_chart,
)
from utils.filters import render_sidebar_filters, apply_filters, filter_by_quarter
from utils.auth import check_authentication, render_user_info_sidebar

st.set_page_config(
    page_title=f"Portfolio Builder - {PAGE_CONFIG['page_title']}",
    page_icon=PAGE_CONFIG['page_icon'],
    layout=PAGE_CONFIG['layout'],
)

# Auth check
if not check_authentication():
    st.stop()

render_user_info_sidebar()

st.title("Portfolio Builder")

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

# Scenario Selector
st.header("Scenario Selection")

# Month names for display
MONTHS = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

col1, col2 = st.columns([1, 2])

with col1:
    period_type = st.radio(
        "Period Type",
        options=['Quarter', 'Date Range'],
        horizontal=True,
        help="Analyze by calendar quarter or custom recurring date range"
    )

with col2:
    if period_type == 'Quarter':
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            target_period = st.selectbox(
                "Target Quarter",
                options=['All Quarters'] + QUARTERS,
                key="target_quarter_select"
            )
        with q_col2:
            year_options = ['All Time', 'Last 12 Months'] + [str(y) for y in sorted(filtered_df['Year'].unique(), reverse=True)]
            year_filter = st.selectbox(
                "Year Range",
                options=year_options,
            )
    else:
        # Custom date range - recurring across all years
        st.markdown("**Recurring Date Range** (applied to all years)")
        range_col1, range_col2, range_col3 = st.columns([2, 2, 1])

        with range_col1:
            start_month = st.selectbox(
                "Start Month",
                options=list(MONTHS.values()),
                index=0,
                key="range_start_month"
            )
            start_day = st.number_input(
                "Start Day",
                min_value=1,
                max_value=31,
                value=1,
                key="range_start_day"
            )

        with range_col2:
            end_month = st.selectbox(
                "End Month",
                options=list(MONTHS.values()),
                index=11,  # December
                key="range_end_month"
            )
            end_day = st.number_input(
                "End Day",
                min_value=1,
                max_value=31,
                value=31,
                key="range_end_day"
            )

        with range_col3:
            year_options = ['All Years'] + [str(y) for y in sorted(filtered_df['Year'].unique(), reverse=True)]
            year_filter = st.selectbox(
                "Year",
                options=year_options,
                key="range_year_filter"
            )

# Apply scenario filters
scenario_df = filtered_df.copy()

# Apply period filter
if period_type == 'Quarter':
    if target_period != 'All Quarters':
        scenario_df = scenario_df[scenario_df['Quarter'] == target_period]
    period_label = target_period

    if year_filter == 'Last 12 Months':
        cutoff = scenario_df['Date Opened'].max() - pd.Timedelta(days=365)
        scenario_df = scenario_df[scenario_df['Date Opened'] >= cutoff]
    elif year_filter != 'All Time':
        scenario_df = scenario_df[scenario_df['Year'] == int(year_filter)]
else:
    # Custom date range filter
    start_month_num = [k for k, v in MONTHS.items() if v == start_month][0]
    end_month_num = [k for k, v in MONTHS.items() if v == end_month][0]

    # Create month-day tuple for comparison
    scenario_df['_month_day'] = list(zip(scenario_df['Month'], scenario_df['Date Opened'].dt.day))

    start_tuple = (start_month_num, start_day)
    end_tuple = (end_month_num, end_day)

    if start_tuple <= end_tuple:
        # Normal range (e.g., Jan 15 - Mar 30)
        mask = (scenario_df['_month_day'] >= start_tuple) & (scenario_df['_month_day'] <= end_tuple)
    else:
        # Range spans year boundary (e.g., Nov 15 - Jan 15)
        mask = (scenario_df['_month_day'] >= start_tuple) | (scenario_df['_month_day'] <= end_tuple)

    scenario_df = scenario_df[mask].drop(columns=['_month_day'])

    period_label = f"{start_month[:3]} {start_day} - {end_month[:3]} {end_day}"

    if year_filter != 'All Years':
        scenario_df = scenario_df[scenario_df['Year'] == int(year_filter)]
        year_filter_display = year_filter
    else:
        year_filter_display = "All Years"

if len(scenario_df) == 0:
    st.warning("No trades match the scenario criteria.")
    st.stop()

# Strategy Performance for Selected Scenario
if period_type == 'Quarter':
    st.header(f"Strategy Performance: {period_label} ({year_filter})")
else:
    st.header(f"Strategy Performance: {period_label} ({year_filter_display})")

# Calculate metrics for scenario
scenario_metrics = []
for strategy in scenario_df['Strategy'].unique():
    strategy_df = scenario_df[scenario_df['Strategy'] == strategy]
    if len(strategy_df) >= MIN_TRADES_FOR_STATS:
        ev = calculate_expected_value(strategy_df)
        sharpe = calculate_sharpe_like_ratio(strategy_df)

        scenario_metrics.append({
            'Strategy': strategy,
            'EV': ev,
            'Win Rate': (strategy_df['P/L'] > 0).mean() * 100,
            'Trades': len(strategy_df),
            'Sharpe-like Ratio': sharpe if not np.isnan(sharpe) else 0,
        })

if not scenario_metrics:
    st.info("Not enough trades for any strategy in this scenario.")
    st.stop()

scenario_metrics_df = pd.DataFrame(scenario_metrics)
scenario_metrics_df['Rank'] = range(1, len(scenario_metrics_df) + 1)
scenario_metrics_df = scenario_metrics_df.sort_values('EV', ascending=False)
scenario_metrics_df['Rank'] = range(1, len(scenario_metrics_df) + 1)

# Format for display
display_metrics = scenario_metrics_df.copy()
display_metrics['EV'] = display_metrics['EV'].apply(lambda x: f"${x:.2f}")
display_metrics['Win Rate'] = display_metrics['Win Rate'].apply(lambda x: f"{x:.1f}%")
display_metrics['Sharpe-like Ratio'] = display_metrics['Sharpe-like Ratio'].apply(lambda x: f"{x:.2f}")

st.dataframe(
    display_metrics[['Rank', 'Strategy', 'EV', 'Win Rate', 'Trades', 'Sharpe-like Ratio']],
    use_container_width=True,
    hide_index=True,
)

# Strategy Selection Interface
st.header("Build Your Portfolio")

# Get available options for current scenario
available_options = sorted(scenario_df['Strategy'].unique())

# Clean up old session state key if it exists (from previous version)
if 'selected_portfolio_strategies' in st.session_state:
    del st.session_state.selected_portfolio_strategies
if 'portfolio_strategies_override' in st.session_state:
    del st.session_state.portfolio_strategies_override

# Initialize the multiselect widget's session state BEFORE the widget is created
# This is the ONLY source of truth - no separate tracking variable
if 'portfolio_multiselect' not in st.session_state:
    # Default to top 5 by EV on first load
    top_5 = scenario_metrics_df.head(5)['Strategy'].tolist()
    # Filter to only available options
    initial_selection = [s for s in top_5 if s in available_options]
    st.session_state.portfolio_multiselect = initial_selection
else:
    # Ensure current selection only contains available strategies
    # (handles case where scenario filters change)
    current = st.session_state.portfolio_multiselect
    valid_current = [s for s in current if s in available_options]
    if valid_current != current:
        st.session_state.portfolio_multiselect = valid_current

# Check if there are filtered strategies from Quarterly Analysis
has_quarterly_filter = 'filtered_strategies_from_quarterly' in st.session_state

if has_quarterly_filter:
    filtered_from_quarterly = st.session_state.filtered_strategies_from_quarterly
    filter_period = st.session_state.get('filtered_strategies_period', 'Unknown')
    filter_year = st.session_state.get('filtered_strategies_year', 'All Years')

    # Only include strategies that exist in current scenario
    available_filtered = [s for s in filtered_from_quarterly if s in available_options]

    if available_filtered:
        st.info(f"💡 **{len(available_filtered)} strategies** from Quarterly Analysis reliability filter ({filter_period}, {filter_year}) are available.")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(
                "Use Filtered Strategies",
                key="use_quarterly_strategies",
                help="Start with strategies that passed the reliability filter in Quarterly Analysis"
            ):
                # Directly update the widget's session state key
                st.session_state.portfolio_multiselect = available_filtered
                st.rerun()

# Render the multiselect - value comes from session state (no default parameter needed)
selected_strategies = st.multiselect(
    "Select strategies to include in portfolio",
    options=available_options,
    key="portfolio_multiselect",
)

if not selected_strategies:
    st.info("Please select at least one strategy to build a portfolio.")
    st.stop()

# Trade Allocation & Buying Power Section
st.header("Trade Allocation & Buying Power")

st.markdown("""
Configure how many concurrent trades to run for each strategy. This affects:
- **Buying Power (BP)** required for the portfolio
- **Equity Curve** simulation (P/L scaled by allocation)
- **Drawdown** analysis
""")

# Check if Margin Req. column exists
has_margin_data = 'Margin Req.' in scenario_df.columns

if not has_margin_data:
    st.warning("⚠️ 'Margin Req.' column not found in data. BP estimates will not be available.")

# Initialize allocation values in session state BEFORE creating widgets
# This avoids the "default value vs session state" conflict
for strategy in selected_strategies:
    alloc_key = f"alloc_{strategy}"
    if alloc_key not in st.session_state:
        st.session_state[alloc_key] = 1

col1, col2 = st.columns([3, 1])

with col2:
    # Quick set buttons
    st.markdown("**Quick Set**")
    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        if st.button("All to 1", key="set_all_1"):
            for strategy in selected_strategies:
                st.session_state[f"alloc_{strategy}"] = 1
            st.rerun()
    with quick_col2:
        if st.button("All to 2", key="set_all_2"):
            for strategy in selected_strategies:
                st.session_state[f"alloc_{strategy}"] = 2
            st.rerun()

with col1:
    st.markdown("**Set trades per strategy:**")

# Build the allocation interface
allocation_data = []
trade_allocations = {}

# Create allocation inputs for each strategy
# Use rows of 4 columns
num_strategies = len(selected_strategies)
strategies_per_row = 4

for row_start in range(0, num_strategies, strategies_per_row):
    row_strategies = selected_strategies[row_start:row_start + strategies_per_row]
    alloc_cols = st.columns(len(row_strategies))

    for col_idx, strategy in enumerate(row_strategies):
        with alloc_cols[col_idx]:
            # Get BP requirements for this strategy
            bp_req = calculate_strategy_bp_requirements(scenario_df, strategy)

            # Truncate long strategy names for display
            display_name = f"{strategy[:25]}..." if len(strategy) > 25 else strategy

            # Create the number input - value comes from session state (initialized above)
            allocation = st.number_input(
                display_name,
                min_value=0,
                max_value=10,
                step=1,
                key=f"alloc_{strategy}",
                help=f"Max BP/trade: ${bp_req['max_bp']:,.0f}" if bp_req['max_bp'] > 0 else "BP data not available"
            )

            trade_allocations[strategy] = allocation

            # Store allocation data for display
            allocation_data.append({
                'Strategy': strategy,
                'Trades Allocated': allocation,
                'Max BP/Trade': bp_req['max_bp'],
                'Avg BP/Trade': bp_req['avg_bp'],
                'Total BP': bp_req['max_bp'] * allocation,
            })

# Calculate total portfolio BP
bp_result = calculate_portfolio_bp(scenario_df, trade_allocations)

# Display BP Summary
st.markdown("---")
st.subheader("Buying Power Summary")

bp_col1, bp_col2, bp_col3 = st.columns(3)

with bp_col1:
    st.metric(
        "Total Buying Power Required",
        f"${bp_result['total_bp']:,.0f}",
        help="Sum of (Max BP per trade × Allocated trades) for all strategies"
    )

with bp_col2:
    active_strategies = sum(1 for a in trade_allocations.values() if a > 0)
    st.metric("Active Strategies", f"{active_strategies} of {len(selected_strategies)}")

with bp_col3:
    total_allocated_trades = sum(trade_allocations.values())
    st.metric("Total Concurrent Trades", total_allocated_trades)

# Detailed BP table
with st.expander("Detailed BP by Strategy", expanded=False):
    bp_df = pd.DataFrame(allocation_data)
    bp_df['Max BP/Trade'] = bp_df['Max BP/Trade'].apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")
    bp_df['Avg BP/Trade'] = bp_df['Avg BP/Trade'].apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")
    bp_df['Total BP'] = bp_df['Total BP'].apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")

    st.dataframe(bp_df, use_container_width=True, hide_index=True)

# Portfolio Analysis Panel
st.header("Portfolio Analysis")

# Calculate portfolio metrics
portfolio_metrics = calculate_portfolio_metrics(scenario_df, selected_strategies)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Trades", f"{portfolio_metrics['total_trades']:,}")

with col2:
    st.metric("Combined EV", f"${portfolio_metrics['combined_ev']:.2f}")

with col3:
    st.metric("Win Rate", f"{portfolio_metrics['win_rate']:.1f}%")

with col4:
    st.metric("Total P/L", f"${portfolio_metrics['total_pnl']:,.2f}")

with col5:
    st.metric("Max Drawdown", f"${portfolio_metrics['max_drawdown']:.2f}")

# Portfolio Equity Curve (with allocation scaling)
st.subheader("Portfolio Equity Curve")

# Check if any trades are allocated
active_allocations = {k: v for k, v in trade_allocations.items() if v > 0}

if not active_allocations:
    st.warning("No trades allocated. Set at least one strategy to 1 or more trades.")
else:
    # Simulate portfolio with allocations
    simulated_portfolio = simulate_portfolio_with_allocation(
        scenario_df,
        active_allocations,
        start_capital=bp_result['total_bp'] if bp_result['total_bp'] > 0 else 10000
    )

    if len(simulated_portfolio) > 0:
        # Create equity curve from simulated data
        import plotly.graph_objects as go
        from config import COLORS

        # Equity curve chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=simulated_portfolio['Date Closed'],
            y=simulated_portfolio['Equity'],
            mode='lines',
            name='Portfolio Equity',
            fill='tozeroy',
            fillcolor='rgba(40, 167, 69, 0.3)' if simulated_portfolio['Equity'].iloc[-1] > simulated_portfolio['Equity'].iloc[0] else 'rgba(220, 53, 69, 0.3)',
            line=dict(color=COLORS['positive'] if simulated_portfolio['Equity'].iloc[-1] > simulated_portfolio['Equity'].iloc[0] else COLORS['negative']),
            hovertemplate='$%{y:,.2f}<extra></extra>',
        ))

        # Add starting capital line
        start_capital = simulated_portfolio['Equity'].iloc[0] - simulated_portfolio['Scaled P/L'].iloc[0]
        fig.add_hline(y=start_capital, line_dash="dash", line_color=COLORS['neutral'],
                      annotation_text=f"Starting BP: ${start_capital:,.0f}")

        fig.update_layout(
            title="Portfolio Equity Curve (Scaled by Allocation)",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            height=500,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics for simulated portfolio
        sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)

        with sim_col1:
            total_pnl = simulated_portfolio['Scaled P/L'].sum()
            st.metric("Total P/L (Scaled)", f"${total_pnl:,.2f}")

        with sim_col2:
            max_dd = simulated_portfolio['Drawdown'].min()
            st.metric("Max Drawdown", f"${max_dd:,.2f}")

        with sim_col3:
            max_dd_pct = simulated_portfolio['Drawdown %'].min()
            st.metric("Max Drawdown %", f"{max_dd_pct:.1f}%")

        with sim_col4:
            if bp_result['total_bp'] > 0:
                roi = (total_pnl / bp_result['total_bp']) * 100
                st.metric("Return on BP", f"{roi:.1f}%")
            else:
                st.metric("Return on BP", "N/A")

        # Drawdown chart
        col1, col2 = st.columns(2)

        with col1:
            # Custom drawdown chart for simulated portfolio
            dd_fig = go.Figure()

            dd_fig.add_trace(go.Scatter(
                x=simulated_portfolio['Date Closed'],
                y=simulated_portfolio['Drawdown'],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(220, 53, 69, 0.3)',
                line=dict(color=COLORS['negative']),
                name='Drawdown',
                hovertemplate='$%{y:,.2f}<extra></extra>',
            ))

            dd_fig.update_layout(
                title="Portfolio Drawdown (Scaled)",
                xaxis_title="Date",
                yaxis_title="Drawdown ($)",
                height=300,
                showlegend=False,
            )

            st.plotly_chart(dd_fig, use_container_width=True)

        with col2:
            # Individual strategy contribution (scaled)
            st.subheader("Strategy Contribution (Scaled)")
            contribution_data = []
            for strategy in active_allocations.keys():
                strat_df = simulated_portfolio[simulated_portfolio['Strategy'] == strategy]
                allocation = active_allocations[strategy]
                scaled_pnl = strat_df['P/L'].sum() * allocation

                contribution_data.append({
                    'Strategy': strategy,
                    'Allocation': f"{allocation}x",
                    'Trades': len(strat_df),
                    'Scaled P/L': scaled_pnl,
                })

            contribution_df = pd.DataFrame(contribution_data)
            contribution_df = contribution_df.sort_values('Scaled P/L', ascending=False)
            contribution_df['Scaled P/L'] = contribution_df['Scaled P/L'].apply(lambda x: f"${x:,.2f}")

            st.dataframe(contribution_df, use_container_width=True, hide_index=True)

# Strategy Correlation Matrix
st.subheader("Strategy Diversification")

if len(selected_strategies) >= 2:
    corr_matrix = calculate_strategy_correlation_matrix(scenario_df, selected_strategies)

    if not corr_matrix.empty:
        corr_heatmap = create_correlation_heatmap(corr_matrix, title="Strategy Correlation Matrix")
        st.plotly_chart(corr_heatmap, use_container_width=True)

        # Diversification score
        avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()
        diversification_score = 1 - avg_corr

        st.metric(
            "Diversification Score",
            f"{diversification_score:.2f}",
            help="1.0 = perfectly uncorrelated (best diversification), 0.0 = perfectly correlated (no diversification)"
        )

        if diversification_score > 0.7:
            st.success("Excellent diversification! Strategies are largely uncorrelated.")
        elif diversification_score > 0.4:
            st.info("Good diversification. Some correlation between strategies.")
        else:
            st.warning("Limited diversification. Consider adding uncorrelated strategies.")
else:
    st.info("Select at least 2 strategies to view diversification analysis.")

# Quarterly Rotation Strategy
st.header("Quarterly Rotation Strategy")

st.markdown("""
Automatically suggested strategy rotation based on historical quarterly performance.
""")

# Get best strategies for each quarter
quarterly_recommendations = get_best_strategies_by_quarter(filtered_df, top_n=3)

rotation_data = []
total_projected_ev = 0

for quarter, strategies in quarterly_recommendations.items():
    if strategies:
        strategy_names = [s['strategy'] for s in strategies]
        combined_ev = sum(s['ev'] for s in strategies)
        total_projected_ev += combined_ev

        rotation_data.append({
            'Quarter': quarter,
            'Recommended Strategies': ', '.join(strategy_names),
            'Combined EV': f"${combined_ev:.2f}",
            'Total Trades': sum(s['trades'] for s in strategies),
        })

if rotation_data:
    st.dataframe(
        pd.DataFrame(rotation_data),
        use_container_width=True,
        hide_index=True,
    )

    st.metric("Full Year Projected EV (Rotation)", f"${total_projected_ev:.2f}")

    # Compare to static portfolio
    static_ev = portfolio_metrics['combined_ev'] * 4  # Approximate annual
    improvement = ((total_projected_ev - static_ev) / abs(static_ev) * 100) if static_ev != 0 else 0

    if improvement > 0:
        st.success(f"Quarterly rotation could improve EV by {improvement:.1f}% compared to static portfolio.")
    else:
        st.info("Current static portfolio may perform similarly to quarterly rotation.")

# Custom Portfolio Comparison
st.header("Portfolio Comparison Tool")

st.markdown("Compare your selected portfolio against alternative configurations.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Portfolio")
    st.write(f"Strategies: {', '.join(selected_strategies)}")
    st.write(f"Combined EV: ${portfolio_metrics['combined_ev']:.2f}")
    st.write(f"Win Rate: {portfolio_metrics['win_rate']:.1f}%")
    st.write(f"Max Drawdown: ${portfolio_metrics['max_drawdown']:.2f}")

with col2:
    st.subheader("Alternative: Top 5 by EV")
    top_5_strategies = scenario_metrics_df.head(5)['Strategy'].tolist()
    top_5_metrics = calculate_portfolio_metrics(scenario_df, top_5_strategies)

    st.write(f"Strategies: {', '.join(top_5_strategies)}")
    st.write(f"Combined EV: ${top_5_metrics['combined_ev']:.2f}")
    st.write(f"Win Rate: {top_5_metrics['win_rate']:.1f}%")
    st.write(f"Max Drawdown: ${top_5_metrics['max_drawdown']:.2f}")

# Export functionality
st.header("Export Portfolio")

if st.button("Generate Portfolio Report"):
    report = f"""
# Portfolio Report

## Configuration
- Target Quarter: {target_quarter}
- Year Range: {year_filter}
- Selected Strategies: {', '.join(selected_strategies)}

## Portfolio Metrics
- Total Trades: {portfolio_metrics['total_trades']}
- Combined Expected Value: ${portfolio_metrics['combined_ev']:.2f}
- Win Rate: {portfolio_metrics['win_rate']:.1f}%
- Total P/L: ${portfolio_metrics['total_pnl']:,.2f}
- Maximum Drawdown: ${portfolio_metrics['max_drawdown']:.2f}

## Individual Strategy Performance
"""
    for strategy in selected_strategies:
        strat_df = scenario_df[scenario_df['Strategy'] == strategy]
        if len(strat_df) >= MIN_TRADES_FOR_STATS:
            ev = calculate_expected_value(strat_df)
            report += f"\n### {strategy}\n"
            report += f"- Trades: {len(strat_df)}\n"
            report += f"- Expected Value: ${ev:.2f}\n"
            report += f"- Win Rate: {(strat_df['P/L'] > 0).mean() * 100:.1f}%\n"

    st.download_button(
        label="Download Report",
        data=report,
        file_name="portfolio_report.md",
        mime="text/markdown",
    )

# Strategy Details Expander
with st.expander("View Detailed Strategy Information"):
    for strategy in selected_strategies:
        strat_df = scenario_df[scenario_df['Strategy'] == strategy]

        if len(strat_df) >= MIN_TRADES_FOR_STATS:
            st.subheader(strategy)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Trades", len(strat_df))

            with col2:
                ev = calculate_expected_value(strat_df)
                st.metric("Expected Value", f"${ev:.2f}")

            with col3:
                win_rate = (strat_df['P/L'] > 0).mean() * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")

            with col4:
                total_pnl = strat_df['P/L'].sum()
                st.metric("Total P/L", f"${total_pnl:,.2f}")

            # Mini equity curve
            strat_sorted = strat_df.sort_values('Date Closed').copy()
            strat_sorted['Cumulative P/L'] = strat_sorted['P/L'].cumsum()

            st.line_chart(strat_sorted.set_index('Date Closed')['Cumulative P/L'])

            st.markdown("---")
