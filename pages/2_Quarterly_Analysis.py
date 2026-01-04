"""
Quarterly Analysis Page - Analyze strategy performance by calendar quarter.
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
    calculate_trading_days,
    calculate_profitability_probability,
)
from utils.visualizations import (
    create_ev_heatmap,
    create_quarterly_comparison_chart,
)
from utils.filters import (
    render_sidebar_filters,
    apply_filters,
    get_year_filter_options,
)
from utils.auth import check_authentication, render_user_info_sidebar


def get_quarter_date_range(df, quarter, year=None):
    """Get the date range for a quarter (or custom period) from the data."""
    quarter_ranges = {
        'Q1': (1, 1, 3, 31),
        'Q2': (4, 1, 6, 30),
        'Q3': (7, 1, 9, 30),
        'Q4': (10, 1, 12, 31),
    }

    if quarter in quarter_ranges:
        sm, sd, em, ed = quarter_ranges[quarter]
        if year:
            start_date = pd.Timestamp(year, sm, sd)
            # Handle end of month correctly
            if em == 2:
                ed = 28  # Simplified - doesn't handle leap years
            end_date = pd.Timestamp(year, em, ed)
        else:
            # Use the data's date range
            years = df['Year'].unique()
            start_date = pd.Timestamp(min(years), sm, sd)
            end_date = pd.Timestamp(max(years), em, ed)
        return start_date, end_date
    return None, None


def calculate_strategy_reliability_for_period(df, strategy, period_filter_func, period_start=None, period_end=None):
    """
    Calculate trade density and probability of profit for a strategy in a specific period.

    Returns: dict with trade_count, trading_days, trade_density, probability_of_profit
    """
    strategy_df = df[df['Strategy'] == strategy].copy()
    period_df = strategy_df[strategy_df['Date Opened'].apply(period_filter_func)]

    if len(period_df) < MIN_TRADES_FOR_STATS:
        return {
            'trade_count': len(period_df),
            'trading_days': 0,
            'trade_density': 0,
            'probability_of_profit': np.nan,
        }

    # Calculate trading days
    if period_start is None:
        period_start = period_df['Date Opened'].min()
    if period_end is None:
        period_end = period_df['Date Opened'].max()

    trading_days = calculate_trading_days(period_start, period_end)
    trade_density = (len(period_df) / trading_days * 100) if trading_days > 0 else 0

    # Calculate probability of profit
    prob_result = calculate_profitability_probability(period_df, simulations=1000)  # Fewer sims for speed
    probability_of_profit = prob_result['probability'] if not np.isnan(prob_result['probability']) else 0

    return {
        'trade_count': len(period_df),
        'trading_days': trading_days,
        'trade_density': trade_density,
        'probability_of_profit': probability_of_profit,
    }

st.set_page_config(
    page_title=f"Quarterly Analysis - {PAGE_CONFIG['page_title']}",
    page_icon=PAGE_CONFIG['page_icon'],
    layout=PAGE_CONFIG['layout'],
)

# Auth check
if not check_authentication():
    st.stop()

render_user_info_sidebar()

st.title("Quarterly Analysis")

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

# Year filter for heatmap
st.sidebar.markdown("---")
st.sidebar.subheader("Year Filter")
year_options = get_year_filter_options(filtered_df)
selected_year = st.sidebar.selectbox(
    "Select Year",
    options=year_options,
    index=0,
)

if selected_year != 'All Years':
    year_filtered_df = filtered_df[filtered_df['Year'] == int(selected_year)]
else:
    year_filtered_df = filtered_df

if len(year_filtered_df) == 0:
    st.warning("No trades for the selected year.")
    st.stop()

# Heat Map - Strategy × Quarter Performance
st.header("Strategy × Quarter Heat Map")

st.markdown("""
This heat map shows the Expected Value (EV) for each strategy across calendar quarters.
**Green** indicates positive EV, **red** indicates negative EV.
""")

# Strategy filter for heatmap
all_strategies = sorted(year_filtered_df['Strategy'].unique())

with st.expander("Heatmap Settings", expanded=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Filter Strategies for Heatmap**")
        st.caption("Exclude high-EV outliers to improve color gradient visibility")

        heatmap_strategies = st.multiselect(
            "Select strategies to include in heatmap",
            options=all_strategies,
            default=all_strategies,
            key="heatmap_strategy_filter",
        )

    with col2:
        st.markdown("**Quick Actions**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select All", key="select_all_heatmap"):
                st.session_state.heatmap_strategy_filter = all_strategies
                st.rerun()
        with col_b:
            if st.button("Clear All", key="clear_all_heatmap"):
                st.session_state.heatmap_strategy_filter = []
                st.rerun()

    # Custom Quarter Definition
    st.markdown("---")
    st.markdown("**Custom Date Range (5th Column)**")
    st.caption("Define a custom date range to appear as an additional column in the heatmap")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        enable_custom = st.checkbox("Enable Custom Period", value=False, key="enable_custom_quarter")

    with col2:
        custom_start_month = st.selectbox(
            "Start Month",
            options=list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(2000, x, 1).strftime('%B'),
            index=10,  # November
            disabled=not enable_custom,
        )
        custom_start_day = st.number_input(
            "Start Day",
            min_value=1,
            max_value=31,
            value=15,
            disabled=not enable_custom,
        )

    with col3:
        custom_end_month = st.selectbox(
            "End Month",
            options=list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(2000, x, 1).strftime('%B'),
            index=11,  # December
            disabled=not enable_custom,
        )
        custom_end_day = st.number_input(
            "End Day",
            min_value=1,
            max_value=31,
            value=31,
            disabled=not enable_custom,
        )

    if enable_custom:
        custom_label = st.text_input(
            "Custom Period Label",
            value=f"{pd.Timestamp(2000, custom_start_month, 1).strftime('%b')} {custom_start_day} - {pd.Timestamp(2000, custom_end_month, 1).strftime('%b')} {custom_end_day}",
            key="custom_quarter_label",
        )

    # Sorting options
    st.markdown("---")
    st.markdown("**Sort Heatmap**")

    sort_options = ['None (Alphabetical)'] + QUARTERS.copy()
    if enable_custom:
        sort_options.append(custom_label)

    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "Sort strategies by",
            options=sort_options,
            index=0,
            key="heatmap_sort_by",
        )
    with col2:
        sort_order = st.radio(
            "Sort order",
            options=['Descending (Best first)', 'Ascending (Worst first)'],
            horizontal=True,
            key="heatmap_sort_order",
        )

    # Reliability-based filtering
    st.markdown("---")
    st.markdown("**Filter by Reliability Metrics**")
    st.caption("Only show strategies that meet minimum thresholds for a selected period")

    enable_reliability_filter = st.checkbox(
        "Enable reliability filtering",
        value=False,
        key="enable_reliability_filter",
    )

    if enable_reliability_filter:
        # Row 1: Period selection
        filter_period_options = QUARTERS.copy()
        if enable_custom:
            filter_period_options.append(custom_label)

        filter_period = st.selectbox(
            "Filter based on period",
            options=filter_period_options,
            index=0,
            key="reliability_filter_period",
            help="Strategies will be filtered based on their metrics in this period"
        )

        # Row 2: Threshold sliders
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            min_trade_count = st.number_input(
                "Min Trade Count",
                min_value=0,
                max_value=500,
                value=10,
                step=5,
                key="min_trade_count",
                help="Minimum number of trades in the period"
            )

        with filter_col2:
            min_trade_density = st.slider(
                "Min Trade Density (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key="min_trade_density",
                help="Minimum percentage of trading days with trades"
            )

        with filter_col3:
            min_profit_prob = st.slider(
                "Min Profit Probability (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                key="min_profit_prob",
                help="Minimum probability of being profitable"
            )

        with filter_col4:
            min_ev = st.number_input(
                "Min Expected Value ($)",
                min_value=-100.0,
                max_value=500.0,
                value=0.0,
                step=5.0,
                key="min_ev_filter",
                help="Minimum Expected Value in dollars"
            )

        # Calculate reliability metrics for all strategies and filter
        st.markdown("**Calculating reliability metrics...**")

        # Create period filter function
        def create_period_filter(period, custom_sm=None, custom_sd=None, custom_em=None, custom_ed=None):
            quarter_ranges = {
                'Q1': (1, 1, 3, 31),
                'Q2': (4, 1, 6, 30),
                'Q3': (7, 1, 9, 30),
                'Q4': (10, 1, 12, 31),
            }

            if period in quarter_ranges:
                sm, sd, em, ed = quarter_ranges[period]
            else:
                # Custom period
                sm, sd, em, ed = custom_sm, custom_sd, custom_em, custom_ed

            def filter_func(date):
                month, day = date.month, date.day
                start = (sm, sd)
                end = (em, ed)
                if start <= end:
                    return start <= (month, day) <= end
                else:
                    return (month, day) >= start or (month, day) <= end

            return filter_func

        if filter_period in QUARTERS:
            period_filter = create_period_filter(filter_period)
        else:
            period_filter = create_period_filter(
                filter_period,
                custom_start_month, int(custom_start_day),
                custom_end_month, int(custom_end_day)
            )

        # Calculate metrics for each strategy
        reliability_data = []
        strategies_meeting_threshold = []

        for strategy in heatmap_strategies:
            metrics = calculate_strategy_reliability_for_period(
                year_filtered_df, strategy, period_filter
            )

            # Calculate EV for this strategy in the period
            strategy_df = year_filtered_df[year_filtered_df['Strategy'] == strategy]
            period_strategy_df = strategy_df[strategy_df['Date Opened'].apply(period_filter)]
            ev = calculate_expected_value(period_strategy_df) if len(period_strategy_df) >= MIN_TRADES_FOR_STATS else np.nan

            # Check all thresholds
            meets_trade_count = metrics['trade_count'] >= min_trade_count
            meets_density = metrics['trade_density'] >= min_trade_density
            meets_prob = (not np.isnan(metrics['probability_of_profit']) and
                         metrics['probability_of_profit'] * 100 >= min_profit_prob)
            meets_ev = (not np.isnan(ev) and ev >= min_ev)

            all_criteria_met = meets_trade_count and meets_density and meets_prob and meets_ev

            reliability_data.append({
                'Strategy': strategy,
                'Trades': metrics['trade_count'],
                'Density': f"{metrics['trade_density']:.1f}%",
                'Profit Prob': f"{metrics['probability_of_profit']*100:.0f}%" if not np.isnan(metrics['probability_of_profit']) else "N/A",
                'EV': f"${ev:.2f}" if not np.isnan(ev) else "N/A",
                'Meets Criteria': '✓' if all_criteria_met else '✗',
            })

            if all_criteria_met:
                strategies_meeting_threshold.append(strategy)

        # Show reliability summary
        with st.expander(f"Reliability Metrics for {filter_period}", expanded=False):
            reliability_df = pd.DataFrame(reliability_data)
            # Sort by whether criteria is met, then by EV
            st.dataframe(reliability_df, use_container_width=True, hide_index=True)

            st.caption(f"Thresholds: Trades ≥ {min_trade_count}, Density ≥ {min_trade_density}%, Profit Prob ≥ {min_profit_prob}%, EV ≥ ${min_ev:.2f}")

        # Update heatmap strategies based on filter
        if strategies_meeting_threshold:
            filtered_count = len(strategies_meeting_threshold)
            total_count = len(heatmap_strategies)
            st.success(f"**{filtered_count} of {total_count} strategies** meet all reliability thresholds for {filter_period}")
            heatmap_strategies = strategies_meeting_threshold

            # Store filtered strategies in session state for Portfolio Builder
            st.session_state.filtered_strategies_from_quarterly = strategies_meeting_threshold
            st.session_state.filtered_strategies_period = filter_period
            st.session_state.filtered_strategies_year = selected_year

            # Option to use in Portfolio Builder
            st.info("💡 These filtered strategies are now available as a starting point in the **Portfolio Builder** page.")
        else:
            st.warning(f"No strategies meet all thresholds. Try lowering the minimums.")
            # Clear the session state if no strategies meet criteria
            if 'filtered_strategies_from_quarterly' in st.session_state:
                del st.session_state.filtered_strategies_from_quarterly
    else:
        # Clear the filtered strategies from session state when filter is disabled
        if 'filtered_strategies_from_quarterly' in st.session_state:
            del st.session_state.filtered_strategies_from_quarterly

if not heatmap_strategies:
    st.warning("Please select at least one strategy for the heatmap.")
    st.stop()

# Filter data for heatmap
heatmap_df = year_filtered_df[year_filtered_df['Strategy'].isin(heatmap_strategies)]


def calculate_quarterly_matrix_with_custom(df, include_custom=False, custom_start_month=None,
                                           custom_start_day=None, custom_end_month=None,
                                           custom_end_day=None, custom_label=None):
    """
    Create a pivot table of Strategy × Quarter with EV values.
    Optionally includes a custom date range column.
    """
    strategies = df['Strategy'].unique()
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']

    if include_custom and custom_label:
        quarters = quarters + [custom_label]

    matrix_data = []
    for strategy in strategies:
        row = {'Strategy': strategy}

        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            quarter_df = df[(df['Strategy'] == strategy) & (df['Quarter'] == quarter)]
            if len(quarter_df) >= MIN_TRADES_FOR_STATS:
                row[quarter] = calculate_expected_value(quarter_df)
            else:
                row[quarter] = np.nan

        # Custom period calculation
        if include_custom and custom_label:
            strategy_df = df[df['Strategy'] == strategy].copy()

            # Filter by month/day range (across all years)
            def in_custom_range(date):
                month, day = date.month, date.day
                start = (custom_start_month, custom_start_day)
                end = (custom_end_month, custom_end_day)

                if start <= end:
                    # Normal range (e.g., Mar 1 - Jun 30)
                    return start <= (month, day) <= end
                else:
                    # Wrapping range (e.g., Nov 15 - Feb 15)
                    return (month, day) >= start or (month, day) <= end

            custom_df = strategy_df[strategy_df['Date Opened'].apply(in_custom_range)]

            if len(custom_df) >= MIN_TRADES_FOR_STATS:
                row[custom_label] = calculate_expected_value(custom_df)
            else:
                row[custom_label] = np.nan

        matrix_data.append(row)

    return pd.DataFrame(matrix_data).set_index('Strategy')


# Calculate quarterly matrix
if enable_custom:
    quarterly_matrix = calculate_quarterly_matrix_with_custom(
        heatmap_df,
        include_custom=True,
        custom_start_month=custom_start_month,
        custom_start_day=int(custom_start_day),
        custom_end_month=custom_end_month,
        custom_end_day=int(custom_end_day),
        custom_label=custom_label,
    )
else:
    quarterly_matrix = calculate_quarterly_matrix_with_custom(heatmap_df)

# Apply sorting
if sort_by != 'None (Alphabetical)':
    ascending = sort_order == 'Ascending (Worst first)'
    if sort_by in quarterly_matrix.columns:
        # Sort by the selected column, putting NaN values at the end
        quarterly_matrix = quarterly_matrix.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position='last'
        )
else:
    # Alphabetical sort by strategy name
    quarterly_matrix = quarterly_matrix.sort_index()

# Create and display heatmap
heatmap_fig = create_ev_heatmap(
    quarterly_matrix,
    title=f"Expected Value by Strategy and Quarter ({selected_year})",
    x_title="Quarter",
    y_title="Strategy",
)
st.plotly_chart(heatmap_fig, use_container_width=True)

# Show excluded strategies info
excluded = set(all_strategies) - set(heatmap_strategies)
if excluded:
    st.caption(f"Excluded from heatmap: {', '.join(sorted(excluded))}")

# Quarter Deep-Dive Panel
st.header("Quarter Deep-Dive")

quarter_options = QUARTERS.copy()
if enable_custom:
    quarter_options = quarter_options + [custom_label]

selected_quarter = st.selectbox(
    "Select Quarter to Analyze",
    options=quarter_options,
    index=0,
)

# Get trades for selected quarter
if selected_quarter in QUARTERS:
    quarter_df = year_filtered_df[year_filtered_df['Quarter'] == selected_quarter]
else:
    # Custom quarter
    def in_custom_range(date):
        month, day = date.month, date.day
        start = (custom_start_month, custom_start_day)
        end = (custom_end_month, custom_end_day)

        if start <= end:
            return start <= (month, day) <= end
        else:
            return (month, day) >= start or (month, day) <= end

    quarter_df = year_filtered_df[year_filtered_df['Date Opened'].apply(in_custom_range)]

if len(quarter_df) < MIN_TRADES_FOR_STATS:
    st.info(f"Not enough trades in {selected_quarter} for meaningful analysis (minimum {MIN_TRADES_FOR_STATS} required).")
else:
    # Calculate metrics for this quarter
    quarter_metrics = calculate_strategy_metrics(quarter_df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top 5 Strategies in {selected_quarter}")
        top_5 = quarter_metrics.head(5).copy()
        top_5['Expected Value'] = top_5['Expected Value'].apply(lambda x: f"${x:.2f}")
        top_5['Win Rate'] = top_5['Win Rate'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(
            top_5[['Strategy', 'Trades', 'Win Rate', 'Expected Value']],
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.subheader(f"Bottom 5 Strategies in {selected_quarter}")
        bottom_5 = quarter_metrics.tail(5).copy()
        bottom_5['Expected Value'] = bottom_5['Expected Value'].apply(lambda x: f"${x:.2f}")
        bottom_5['Win Rate'] = bottom_5['Win Rate'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(
            bottom_5[['Strategy', 'Trades', 'Win Rate', 'Expected Value']],
            use_container_width=True,
            hide_index=True,
        )

    # Quarter performance across years (only for standard quarters)
    if selected_quarter in QUARTERS:
        st.subheader(f"{selected_quarter} Performance Across Years")

        years = sorted(filtered_df['Year'].unique())
        year_ev_data = []

        for year in years:
            year_quarter_df = filtered_df[
                (filtered_df['Year'] == year) &
                (filtered_df['Quarter'] == selected_quarter)
            ]
            if len(year_quarter_df) >= MIN_TRADES_FOR_STATS:
                ev = calculate_expected_value(year_quarter_df)
                year_ev_data.append({
                    'Year': year,
                    'Expected Value': ev,
                    'Trades': len(year_quarter_df),
                    'Win Rate': (year_quarter_df['P/L'] > 0).mean() * 100,
                })

        if year_ev_data:
            year_ev_df = pd.DataFrame(year_ev_data)
            st.dataframe(
                year_ev_df.style.format({
                    'Expected Value': '${:.2f}',
                    'Win Rate': '{:.1f}%',
                }),
                use_container_width=True,
                hide_index=True,
            )

# Quarter Comparison Table
st.header("Quarter Comparison Table")

# Determine columns for comparison
comparison_quarters = QUARTERS.copy()
if enable_custom:
    comparison_quarters = comparison_quarters + [custom_label]

comparison_data = []
for strategy in filtered_df['Strategy'].unique():
    row = {'Strategy': strategy}
    evs = []

    for quarter in QUARTERS:
        quarter_strategy_df = filtered_df[
            (filtered_df['Strategy'] == strategy) &
            (filtered_df['Quarter'] == quarter)
        ]
        if len(quarter_strategy_df) >= MIN_TRADES_FOR_STATS:
            ev = calculate_expected_value(quarter_strategy_df)
            row[f'{quarter} EV'] = ev
            evs.append((quarter, ev))
        else:
            row[f'{quarter} EV'] = np.nan
            evs.append((quarter, np.nan))

    # Custom quarter
    if enable_custom:
        strategy_df = filtered_df[filtered_df['Strategy'] == strategy]

        def in_custom_range(date):
            month, day = date.month, date.day
            start = (custom_start_month, custom_start_day)
            end = (custom_end_month, custom_end_day)
            if start <= end:
                return start <= (month, day) <= end
            else:
                return (month, day) >= start or (month, day) <= end

        custom_strategy_df = strategy_df[strategy_df['Date Opened'].apply(in_custom_range)]

        if len(custom_strategy_df) >= MIN_TRADES_FOR_STATS:
            ev = calculate_expected_value(custom_strategy_df)
            row[f'{custom_label} EV'] = ev
            evs.append((custom_label, ev))
        else:
            row[f'{custom_label} EV'] = np.nan
            evs.append((custom_label, np.nan))

    # Find best and worst quarters
    valid_evs = [(q, e) for q, e in evs if not np.isnan(e)]
    if valid_evs:
        row['Best Quarter'] = max(valid_evs, key=lambda x: x[1])[0]
        row['Worst Quarter'] = min(valid_evs, key=lambda x: x[1])[0]
    else:
        row['Best Quarter'] = 'N/A'
        row['Worst Quarter'] = 'N/A'

    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# Sort by average EV across quarters
ev_cols = [f'{q} EV' for q in comparison_quarters]
comparison_df['Avg EV'] = comparison_df[ev_cols].mean(axis=1)
comparison_df = comparison_df.sort_values('Avg EV', ascending=False)

# Format for display
display_cols = ['Strategy'] + ev_cols + ['Best Quarter', 'Worst Quarter']
display_comparison = comparison_df[display_cols].copy()

for col in ev_cols:
    display_comparison[col] = display_comparison[col].apply(
        lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A"
    )

st.dataframe(
    display_comparison,
    use_container_width=True,
    hide_index=True,
)

# Strategy comparison chart
st.header("Visual Quarter Comparison")

# Select strategies to compare
strategies_to_compare = st.multiselect(
    "Select strategies to compare (max 5)",
    options=sorted(filtered_df['Strategy'].unique()),
    default=sorted(filtered_df['Strategy'].unique())[:5],
    max_selections=5,
)

if strategies_to_compare:
    comparison_chart = create_quarterly_comparison_chart(
        filtered_df,
        strategies_to_compare,
        title="Quarterly Performance Comparison",
    )
    st.plotly_chart(comparison_chart, use_container_width=True)

# Insights section
st.header("Key Insights")

# Find consistently positive strategies
consistently_positive = []
for strategy in filtered_df['Strategy'].unique():
    all_positive = True
    quarters_with_data = 0

    for quarter in QUARTERS:
        quarter_strategy_df = filtered_df[
            (filtered_df['Strategy'] == strategy) &
            (filtered_df['Quarter'] == quarter)
        ]
        if len(quarter_strategy_df) >= MIN_TRADES_FOR_STATS:
            quarters_with_data += 1
            ev = calculate_expected_value(quarter_strategy_df)
            if ev <= 0:
                all_positive = False
                break

    if all_positive and quarters_with_data >= 3:
        consistently_positive.append(strategy)

if consistently_positive:
    st.success(f"**Consistently Positive Strategies** (positive EV in all quarters with data): {', '.join(consistently_positive)}")

# Find strategies with strong seasonal patterns
st.subheader("Seasonal Pattern Detection")

for strategy in filtered_df['Strategy'].unique()[:5]:  # Limit to top 5 for performance
    evs = []
    for quarter in QUARTERS:
        quarter_strategy_df = filtered_df[
            (filtered_df['Strategy'] == strategy) &
            (filtered_df['Quarter'] == quarter)
        ]
        if len(quarter_strategy_df) >= MIN_TRADES_FOR_STATS:
            evs.append(calculate_expected_value(quarter_strategy_df))
        else:
            evs.append(np.nan)

    valid_evs = [e for e in evs if not np.isnan(e)]
    if len(valid_evs) >= 3:
        ev_range = max(valid_evs) - min(valid_evs)
        if ev_range > 50:  # Significant variation
            best_q = QUARTERS[evs.index(max(valid_evs))]
            worst_q = QUARTERS[evs.index(min(valid_evs))]
            st.info(f"**{strategy}** shows seasonal variation: Best in {best_q} (${max(valid_evs):.2f}), Worst in {worst_q} (${min(valid_evs):.2f})")
