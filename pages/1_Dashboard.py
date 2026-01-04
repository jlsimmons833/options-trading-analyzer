"""
Dashboard Page - File upload, summary metrics, and quick insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, MIN_TRADES_FOR_STATS, QUARTERS
from utils.data_processing import load_and_process_data, detect_file_format
from utils.calculations import (
    calculate_expected_value,
    calculate_strategy_metrics,
    calculate_ev_reliability_metrics,
    calculate_ev_sensitivity,
)
from utils.visualizations import (
    create_ev_bar_chart,
    create_pie_chart,
    create_ev_heatmap,
    create_cv_gauge,
    create_pnl_distribution,
    create_pnl_box_plot,
    create_sensitivity_chart,
)
from utils.filters import initialize_session_state, render_sidebar_filters, apply_filters
from utils.auth import check_authentication, render_user_info_sidebar

st.set_page_config(
    page_title=f"Dashboard - {PAGE_CONFIG['page_title']}",
    page_icon=PAGE_CONFIG['page_icon'],
    layout=PAGE_CONFIG['layout'],
)

# Auth check
if not check_authentication():
    st.stop()

render_user_info_sidebar()

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

# Strategy Deep Dive Section
st.header("Strategy Deep Dive")

st.markdown("""
Select strategies to analyze EV reliability. This helps you understand how much to trust the Expected Value calculation.
""")

# Strategy selector for deep dive
all_strategies = sorted(filtered_df['Strategy'].unique())
deep_dive_strategies = st.multiselect(
    "Select strategies to analyze (1-5 recommended)",
    options=all_strategies,
    default=[],
    max_selections=5,
    key="deep_dive_strategies",
)

if deep_dive_strategies:
    # Time period filter for the deep dive
    st.markdown("---")
    period_col1, period_col2 = st.columns(2)

    with period_col1:
        dd_period_type = st.selectbox(
            "Analysis Period",
            options=['All Data', 'Standard Quarter', 'Custom Date Range'],
            key="dd_period_type",
        )

    with period_col2:
        if dd_period_type == 'Standard Quarter':
            dd_quarter = st.selectbox(
                "Select Quarter",
                options=QUARTERS,
                key="dd_quarter",
            )
        elif dd_period_type == 'Custom Date Range':
            dd_date_range = st.date_input(
                "Date Range",
                value=(filtered_df['Date Opened'].min().date(), filtered_df['Date Opened'].max().date()),
                key="dd_date_range",
            )

    # CV Legend
    with st.expander("Understanding Coefficient of Variation (CV)", expanded=False):
        st.markdown("""
        **CV measures how spread out the P/L values are relative to the average.**

        | CV Range | Label | What it Means |
        |----------|-------|---------------|
        | < 0.5 | **Consistent** | Results cluster tightly around EV - high confidence |
        | 0.5 - 1.0 | **Moderate** | Some spread, but EV is still a reasonable estimate |
        | 1.0 - 2.0 | **Variable** | Wide spread - use EV as a rough guide only |
        | > 2.0 | **Highly Variable** | Very unpredictable - treat EV with caution |

        *Lower CV = More reliable EV*
        """)

    st.markdown("---")

    # Analyze each selected strategy
    for strategy in deep_dive_strategies:
        st.subheader(f"{strategy}")

        # Filter data for this strategy
        strategy_df = filtered_df[filtered_df['Strategy'] == strategy].copy()

        # Apply time period filter
        if dd_period_type == 'Standard Quarter':
            strategy_df = strategy_df[strategy_df['Quarter'] == dd_quarter]
            period_start = strategy_df['Date Opened'].min() if len(strategy_df) > 0 else None
            period_end = strategy_df['Date Opened'].max() if len(strategy_df) > 0 else None
        elif dd_period_type == 'Custom Date Range' and len(dd_date_range) == 2:
            strategy_df = strategy_df[
                (strategy_df['Date Opened'].dt.date >= dd_date_range[0]) &
                (strategy_df['Date Opened'].dt.date <= dd_date_range[1])
            ]
            period_start = pd.Timestamp(dd_date_range[0])
            period_end = pd.Timestamp(dd_date_range[1])
        else:
            period_start = strategy_df['Date Opened'].min() if len(strategy_df) > 0 else None
            period_end = strategy_df['Date Opened'].max() if len(strategy_df) > 0 else None

        if len(strategy_df) < MIN_TRADES_FOR_STATS:
            st.warning(f"Insufficient trades ({len(strategy_df)}) for reliable analysis. Need at least {MIN_TRADES_FOR_STATS}.")
            st.markdown("---")
            continue

        # Calculate reliability metrics
        reliability = calculate_ev_reliability_metrics(strategy_df, period_start, period_end)
        sensitivity = calculate_ev_sensitivity(strategy_df)

        # Row 1: Key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            # Trade count with density
            trade_text = f"{reliability['trade_count']} of {reliability['trading_days']} days"
            density_pct = reliability['trade_density']
            st.metric(
                "Trade Coverage",
                trade_text,
                delta=f"{density_pct:.0f}% density",
                delta_color="off",
            )

        with metric_col2:
            # EV with confidence interval
            ev, margin, lower, upper = reliability['confidence_interval']
            if not pd.isna(margin):
                st.metric(
                    "Expected Value",
                    f"${ev:.2f}",
                    delta=f"± ${margin:.2f}",
                    delta_color="off",
                )
                st.caption(f"95% CI: ${lower:.2f} to ${upper:.2f}")
            else:
                st.metric("Expected Value", f"${ev:.2f}")

        with metric_col3:
            # Standard deviation
            std = reliability['std_dev']
            if not pd.isna(std):
                st.metric("Std Deviation", f"${std:.2f}")
            else:
                st.metric("Std Deviation", "N/A")

        with metric_col4:
            # Reliability indicator
            if reliability['is_reliable']:
                st.metric("Reliability", "Good", delta="Sufficient data", delta_color="normal")
            else:
                st.metric("Reliability", "Caution", delta="Limited data", delta_color="inverse")

        # Row 2: CV gauge and interpretation + P/L distribution
        viz_col1, viz_col2 = st.columns([1, 2])

        with viz_col1:
            # CV Gauge
            cv_gauge = create_cv_gauge(reliability['cv'], title="Variability (CV)")
            st.plotly_chart(cv_gauge, use_container_width=True)

            # CV interpretation
            label, color, description = reliability['cv_interpretation']
            st.markdown(f"**{label}**: {description}")

        with viz_col2:
            # P/L Distribution with EV and CI
            dist_chart = create_pnl_distribution(strategy_df, title="P/L Distribution")
            st.plotly_chart(dist_chart, use_container_width=True)

        # Row 3: Sensitivity analysis (collapsible)
        with st.expander("EV Sensitivity Analysis", expanded=False):
            if not pd.isna(sensitivity[0]):
                base_ev, ev_wo_best, ev_wo_worst, sensitivity_score = sensitivity

                sens_col1, sens_col2 = st.columns([2, 1])

                with sens_col1:
                    sens_chart = create_sensitivity_chart(base_ev, ev_wo_best, ev_wo_worst)
                    st.plotly_chart(sens_chart, use_container_width=True)

                with sens_col2:
                    st.metric(
                        "Sensitivity Score",
                        f"{sensitivity_score:.1f}%",
                        help="Average % change in EV when removing best/worst trade"
                    )

                    if sensitivity_score < 10:
                        st.success("Low sensitivity - EV is stable")
                    elif sensitivity_score < 25:
                        st.info("Moderate sensitivity")
                    else:
                        st.warning("High sensitivity - outliers heavily influence EV")

                    st.caption(f"EV without best: ${ev_wo_best:.2f}")
                    st.caption(f"EV without worst: ${ev_wo_worst:.2f}")
            else:
                st.info("Not enough trades for sensitivity analysis.")

        st.markdown("---")

else:
    st.info("Select one or more strategies above to see detailed reliability analysis.")

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

# Year-over-Year Comparison
st.header("Year-over-Year Comparison")

st.markdown("""
Compare strategy performance for the same time period across different years.
Select a date range (e.g., Nov 15 - Dec 31) to see how each strategy performed during that period in each year.
""")

available_years = sorted(filtered_df['Year'].unique())

if len(available_years) < 2:
    st.info("Need at least 2 years of data for year-over-year comparison.")
else:
    with st.expander("Year-over-Year Settings", expanded=True):
        # Period type selection
        period_type = st.radio(
            "Select period type",
            options=['Standard Quarter', 'Custom Date Range'],
            horizontal=True,
            key="yoy_period_type",
        )

        if period_type == 'Standard Quarter':
            selected_period = st.selectbox(
                "Select Quarter",
                options=QUARTERS,
                key="yoy_quarter",
            )
            period_label = selected_period
        else:
            col1, col2 = st.columns(2)

            with col1:
                yoy_start_month = st.selectbox(
                    "Start Month",
                    options=list(range(1, 13)),
                    format_func=lambda x: pd.Timestamp(2000, x, 1).strftime('%B'),
                    index=10,  # November
                    key="yoy_start_month",
                )
                yoy_start_day = st.number_input(
                    "Start Day",
                    min_value=1,
                    max_value=31,
                    value=15,
                    key="yoy_start_day",
                )

            with col2:
                yoy_end_month = st.selectbox(
                    "End Month",
                    options=list(range(1, 13)),
                    format_func=lambda x: pd.Timestamp(2000, x, 1).strftime('%B'),
                    index=11,  # December
                    key="yoy_end_month",
                )
                yoy_end_day = st.number_input(
                    "End Day",
                    min_value=1,
                    max_value=31,
                    value=31,
                    key="yoy_end_day",
                )

            period_label = f"{pd.Timestamp(2000, yoy_start_month, 1).strftime('%b')} {yoy_start_day} - {pd.Timestamp(2000, yoy_end_month, 1).strftime('%b')} {yoy_end_day}"

        # Strategy filter for YoY heatmap
        st.markdown("---")
        all_strategies = sorted(filtered_df['Strategy'].unique())
        yoy_strategies = st.multiselect(
            "Filter strategies for heatmap",
            options=all_strategies,
            default=all_strategies,
            key="yoy_strategy_filter",
        )

        # Sorting options
        col1, col2 = st.columns(2)
        with col1:
            yoy_sort_by = st.selectbox(
                "Sort strategies by year",
                options=['None (Alphabetical)'] + [str(y) for y in available_years],
                key="yoy_sort_by",
            )
        with col2:
            yoy_sort_order = st.radio(
                "Sort order",
                options=['Descending', 'Ascending'],
                horizontal=True,
                key="yoy_sort_order",
            )

    if not yoy_strategies:
        st.warning("Please select at least one strategy.")
    else:
        # Calculate YoY matrix
        def in_date_range(date, start_month, start_day, end_month, end_day):
            """Check if date falls within the month/day range."""
            month, day = date.month, date.day
            start = (start_month, start_day)
            end = (end_month, end_day)

            if start <= end:
                return start <= (month, day) <= end
            else:
                # Wrapping range (e.g., Nov 15 - Feb 15)
                return (month, day) >= start or (month, day) <= end

        def get_quarter_range(quarter):
            """Get month/day range for a standard quarter."""
            ranges = {
                'Q1': (1, 1, 3, 31),
                'Q2': (4, 1, 6, 30),
                'Q3': (7, 1, 9, 30),
                'Q4': (10, 1, 12, 31),
            }
            return ranges[quarter]

        # Build the matrix
        yoy_matrix_data = []

        for strategy in yoy_strategies:
            row = {'Strategy': strategy}

            for year in available_years:
                # Filter by year
                year_df = filtered_df[(filtered_df['Strategy'] == strategy) & (filtered_df['Year'] == year)]

                # Filter by period
                if period_type == 'Standard Quarter':
                    period_df = year_df[year_df['Quarter'] == selected_period]
                else:
                    period_df = year_df[year_df['Date Opened'].apply(
                        lambda d: in_date_range(d, yoy_start_month, int(yoy_start_day), yoy_end_month, int(yoy_end_day))
                    )]

                if len(period_df) >= MIN_TRADES_FOR_STATS:
                    row[str(year)] = calculate_expected_value(period_df)
                else:
                    row[str(year)] = np.nan

            yoy_matrix_data.append(row)

        yoy_matrix = pd.DataFrame(yoy_matrix_data).set_index('Strategy')

        # Apply sorting
        if yoy_sort_by != 'None (Alphabetical)':
            ascending = yoy_sort_order == 'Ascending'
            if yoy_sort_by in yoy_matrix.columns:
                yoy_matrix = yoy_matrix.sort_values(
                    by=yoy_sort_by,
                    ascending=ascending,
                    na_position='last'
                )
        else:
            yoy_matrix = yoy_matrix.sort_index()

        # Create and display heatmap
        yoy_heatmap = create_ev_heatmap(
            yoy_matrix,
            title=f"Year-over-Year Comparison: {period_label}",
            x_title="Year",
            y_title="Strategy",
        )
        st.plotly_chart(yoy_heatmap, use_container_width=True)

        # Summary statistics
        st.subheader("Period Summary by Year")

        year_summary = []
        for year in available_years:
            if period_type == 'Standard Quarter':
                sm, sd, em, ed = get_quarter_range(selected_period)
            else:
                sm, sd, em, ed = yoy_start_month, int(yoy_start_day), yoy_end_month, int(yoy_end_day)

            year_df = filtered_df[filtered_df['Year'] == year]
            if period_type == 'Standard Quarter':
                period_df = year_df[year_df['Quarter'] == selected_period]
            else:
                period_df = year_df[year_df['Date Opened'].apply(
                    lambda d: in_date_range(d, sm, sd, em, ed)
                )]

            if len(period_df) >= MIN_TRADES_FOR_STATS:
                year_summary.append({
                    'Year': year,
                    'Trades': len(period_df),
                    'Win Rate': f"{(period_df['P/L'] > 0).mean() * 100:.1f}%",
                    'Total P/L': f"${period_df['P/L'].sum():,.2f}",
                    'Expected Value': f"${calculate_expected_value(period_df):.2f}",
                })

        if year_summary:
            st.dataframe(
                pd.DataFrame(year_summary),
                use_container_width=True,
                hide_index=True,
            )
