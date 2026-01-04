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

col1, col2 = st.columns(2)

with col1:
    target_quarter = st.selectbox(
        "Target Quarter",
        options=['All Quarters'] + QUARTERS,
    )

with col2:
    year_options = ['All Time', 'Last 12 Months'] + [str(y) for y in sorted(filtered_df['Year'].unique(), reverse=True)]
    year_filter = st.selectbox(
        "Year Range",
        options=year_options,
    )

# Apply scenario filters
scenario_df = filtered_df.copy()

if target_quarter != 'All Quarters':
    scenario_df = scenario_df[scenario_df['Quarter'] == target_quarter]

if year_filter == 'Last 12 Months':
    cutoff = scenario_df['Date Opened'].max() - pd.Timedelta(days=365)
    scenario_df = scenario_df[scenario_df['Date Opened'] >= cutoff]
elif year_filter != 'All Time':
    scenario_df = scenario_df[scenario_df['Year'] == int(year_filter)]

if len(scenario_df) == 0:
    st.warning("No trades match the scenario criteria.")
    st.stop()

# Strategy Performance for Selected Scenario
st.header(f"Strategy Performance: {target_quarter} ({year_filter})")

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

# Check if there are filtered strategies from Quarterly Analysis
has_quarterly_filter = 'filtered_strategies_from_quarterly' in st.session_state

if has_quarterly_filter:
    filtered_from_quarterly = st.session_state.filtered_strategies_from_quarterly
    filter_period = st.session_state.get('filtered_strategies_period', 'Unknown')
    filter_year = st.session_state.get('filtered_strategies_year', 'All Years')

    # Only include strategies that exist in current scenario
    available_filtered = [s for s in filtered_from_quarterly if s in scenario_df['Strategy'].unique()]

    if available_filtered:
        st.info(f"💡 **{len(available_filtered)} strategies** from Quarterly Analysis reliability filter ({filter_period}, {filter_year}) are available.")

        col1, col2 = st.columns([1, 3])
        with col1:
            use_quarterly_filter = st.button(
                "Use Filtered Strategies",
                key="use_quarterly_strategies",
                help="Start with strategies that passed the reliability filter in Quarterly Analysis"
            )

        if use_quarterly_filter:
            st.session_state.portfolio_strategies_override = available_filtered

# Determine default strategies
if 'portfolio_strategies_override' in st.session_state:
    default_strategies = st.session_state.portfolio_strategies_override
    # Clear the override after using it
    del st.session_state.portfolio_strategies_override
elif has_quarterly_filter:
    # If quarterly filter exists but user hasn't clicked button, still use top 5 by EV
    top_5 = scenario_metrics_df.head(5)['Strategy'].tolist()
    default_strategies = top_5
else:
    # Get top 5 by EV as default
    top_5 = scenario_metrics_df.head(5)['Strategy'].tolist()
    default_strategies = top_5

selected_strategies = st.multiselect(
    "Select strategies to include in portfolio",
    options=sorted(scenario_df['Strategy'].unique()),
    default=default_strategies,
)

if not selected_strategies:
    st.info("Please select at least one strategy to build a portfolio.")
    st.stop()

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

# Portfolio Equity Curve
st.subheader("Portfolio Equity Curve")

portfolio_trades = scenario_df[scenario_df['Strategy'].isin(selected_strategies)].copy()
portfolio_trades = portfolio_trades.sort_values('Date Closed')

if len(portfolio_trades) > 0:
    equity_curve = create_equity_curve(portfolio_trades, title="Combined Portfolio Equity Curve")
    st.plotly_chart(equity_curve, use_container_width=True)

    # Drawdown chart
    col1, col2 = st.columns(2)

    with col1:
        drawdown_chart = create_drawdown_chart(portfolio_trades, title="Portfolio Drawdown")
        st.plotly_chart(drawdown_chart, use_container_width=True)

    with col2:
        # Individual strategy contribution
        st.subheader("Strategy Contribution")
        contribution_data = []
        for strategy in selected_strategies:
            strat_df = portfolio_trades[portfolio_trades['Strategy'] == strategy]
            contribution_data.append({
                'Strategy': strategy,
                'Trades': len(strat_df),
                'P/L Contribution': strat_df['P/L'].sum(),
                '% of Total': (len(strat_df) / len(portfolio_trades) * 100) if len(portfolio_trades) > 0 else 0,
            })

        contribution_df = pd.DataFrame(contribution_data)
        contribution_df['P/L Contribution'] = contribution_df['P/L Contribution'].apply(lambda x: f"${x:,.2f}")
        contribution_df['% of Total'] = contribution_df['% of Total'].apply(lambda x: f"{x:.1f}%")

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
