"""
Filter logic and session state management.
"""

import streamlit as st
import pandas as pd


def initialize_session_state(df):
    """
    Initialize session state with default filter values.
    Always resets filters when new data is loaded.
    """
    # Store a hash of the data to detect new uploads
    data_hash = hash(tuple(df['Date Opened'].head(5).astype(str)) + tuple(df['Strategy'].unique()))

    # Reset if this is a new file or first load
    if ('filters_initialized' not in st.session_state or
        st.session_state.get('data_hash') != data_hash):

        st.session_state.filters_initialized = True
        st.session_state.data_hash = data_hash

        # Date range
        st.session_state.date_range = (
            df['Date Opened'].min().date(),
            df['Date Opened'].max().date()
        )

        # Selected strategies
        st.session_state.selected_strategies = df['Strategy'].unique().tolist()

        # Selected outcomes
        st.session_state.selected_outcomes = df['Reason For Close'].unique().tolist()

        # Year filter
        st.session_state.selected_years = ['All Years']

        # Data loaded flag
        st.session_state.data_loaded = True


def reset_filters(df):
    """
    Reset all filters to default values.
    """
    st.session_state.date_range = (
        df['Date Opened'].min().date(),
        df['Date Opened'].max().date()
    )
    st.session_state.selected_strategies = df['Strategy'].unique().tolist()
    st.session_state.selected_outcomes = df['Reason For Close'].unique().tolist()
    st.session_state.selected_years = ['All Years']


def render_sidebar_filters(df):
    """
    Render sidebar filter controls and return the selected values.
    """
    st.sidebar.header("Filters")

    # Date range
    st.sidebar.subheader("Date Range")
    min_date = df['Date Opened'].min().date()
    max_date = df['Date Opened'].max().date()

    date_range = st.sidebar.date_input(
        "Select date range",
        value=st.session_state.get('date_range', (min_date, max_date)),
        min_value=min_date,
        max_value=max_date,
        key='date_range_input'
    )

    # Handle single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range if isinstance(date_range, type(min_date)) else min_date
        end_date = max_date

    # Strategy filter
    st.sidebar.subheader("Strategies")

    all_strategies = sorted(df['Strategy'].unique().tolist())

    # Select/deselect all buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="select_all_strategies"):
            st.session_state.selected_strategies = all_strategies
    with col2:
        if st.button("Clear All", key="clear_all_strategies"):
            st.session_state.selected_strategies = []

    # Get valid defaults (intersection of session state and current options)
    stored_strategies = st.session_state.get('selected_strategies', all_strategies)
    valid_strategy_defaults = [s for s in stored_strategies if s in all_strategies]
    if not valid_strategy_defaults:
        valid_strategy_defaults = all_strategies

    selected_strategies = st.sidebar.multiselect(
        "Select strategies",
        options=all_strategies,
        default=valid_strategy_defaults,
        key='strategy_multiselect'
    )

    # Trade outcome filter
    st.sidebar.subheader("Trade Outcomes")
    all_outcomes = df['Reason For Close'].unique().tolist()

    # Get valid defaults (intersection of session state and current options)
    stored_outcomes = st.session_state.get('selected_outcomes', all_outcomes)
    valid_outcome_defaults = [o for o in stored_outcomes if o in all_outcomes]
    if not valid_outcome_defaults:
        valid_outcome_defaults = all_outcomes

    selected_outcomes = st.sidebar.multiselect(
        "Select outcomes",
        options=all_outcomes,
        default=valid_outcome_defaults,
        key='outcome_multiselect'
    )

    # Reset filters button
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Filters", key="reset_filters"):
        reset_filters(df)
        st.rerun()

    # Show active filter count
    total_filters = 0
    if start_date != min_date or end_date != max_date:
        total_filters += 1
    if len(selected_strategies) != len(all_strategies):
        total_filters += 1
    if len(selected_outcomes) != len(all_outcomes):
        total_filters += 1

    if total_filters > 0:
        st.sidebar.info(f"Active filters: {total_filters}")

    return {
        'date_range': (start_date, end_date),
        'selected_strategies': selected_strategies,
        'selected_outcomes': selected_outcomes,
    }


def apply_filters(df, filters):
    """
    Apply filters to dataframe and return filtered data.
    """
    filtered_df = df.copy()

    # Date filter
    start_date, end_date = filters['date_range']
    filtered_df = filtered_df[
        (filtered_df['Date Opened'].dt.date >= start_date) &
        (filtered_df['Date Opened'].dt.date <= end_date)
    ]

    # Strategy filter
    if filters['selected_strategies']:
        filtered_df = filtered_df[filtered_df['Strategy'].isin(filters['selected_strategies'])]

    # Outcome filter
    if filters['selected_outcomes']:
        filtered_df = filtered_df[filtered_df['Reason For Close'].isin(filters['selected_outcomes'])]

    return filtered_df


def get_year_filter_options(df):
    """
    Get available years for filtering.
    """
    years = sorted(df['Year'].unique().tolist())
    return ['All Years'] + [str(y) for y in years]


def apply_year_filter(df, selected_years):
    """
    Apply year filter to dataframe.
    """
    if 'All Years' in selected_years or not selected_years:
        return df

    years = [int(y) for y in selected_years if y != 'All Years']
    return df[df['Year'].isin(years)]


def filter_by_quarter(df, quarter):
    """
    Filter dataframe by a specific quarter.
    """
    if quarter == 'All Quarters':
        return df
    return df[df['Quarter'] == quarter]


def filter_by_vix_regime(df, regime):
    """
    Filter dataframe by VIX regime.
    """
    if regime == 'All Regimes':
        return df
    return df[df['VIX_Regime'] == regime]


def get_filter_summary(filters, df_original, df_filtered):
    """
    Generate a summary of active filters and their impact.
    """
    original_count = len(df_original)
    filtered_count = len(df_filtered)
    pct_remaining = (filtered_count / original_count * 100) if original_count > 0 else 0

    return {
        'original_count': original_count,
        'filtered_count': filtered_count,
        'pct_remaining': pct_remaining,
        'trades_excluded': original_count - filtered_count,
    }
