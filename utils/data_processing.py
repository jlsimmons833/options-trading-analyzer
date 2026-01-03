"""
Data loading and transformation utilities.
"""

import pandas as pd
import streamlit as st
from config import REQUIRED_COLUMNS


def get_quarter(date):
    """
    Assign calendar quarter based on month.
    Q1: January - March
    Q2: April - June
    Q3: July - September
    Q4: October - December
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


def detect_file_format(uploaded_file):
    """
    Detect whether file is Options Omega or TradeSteward format.
    """
    df = pd.read_csv(uploaded_file, nrows=5)
    uploaded_file.seek(0)  # Reset file pointer

    if 'Strategy' in df.columns and 'Opening VIX' in df.columns:
        return 'options_omega'
    # Placeholder for TradeSteward format
    # elif 'trade_steward_column' in df.columns:
    #     return 'trade_steward'
    else:
        return 'unknown'


def validate_columns(df, required_cols=None):
    """
    Validate that required columns exist in the dataframe.
    Returns tuple of (is_valid, missing_columns).
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS

    missing = [col for col in required_cols if col not in df.columns]
    return len(missing) == 0, missing


@st.cache_data
def load_and_process_data(uploaded_file):
    """
    Load CSV and add derived columns.
    """
    df = pd.read_csv(uploaded_file)

    # Parse dates
    df['Date Opened'] = pd.to_datetime(df['Date Opened'])
    df['Date Closed'] = pd.to_datetime(df['Date Closed'])

    # Add derived columns
    df['Quarter'] = df['Date Opened'].apply(get_quarter)
    df['Year'] = df['Date Opened'].dt.year
    df['Month'] = df['Date Opened'].dt.month
    df['Year-Quarter'] = df['Year'].astype(str) + '-' + df['Quarter']
    df['Is_Win'] = df['P/L'] > 0

    # VIX regime
    df['VIX_Regime'] = df['Opening VIX'].apply(categorize_vix)

    # Trade duration
    df['Duration_Days'] = (df['Date Closed'] - df['Date Opened']).dt.days

    # Sort by date
    df = df.sort_values('Date Opened').reset_index(drop=True)

    return df


def normalize_to_standard_format(df, source_format):
    """
    Convert any format to internal standard format.
    """
    if source_format == 'options_omega':
        return load_and_process_data(df)
    # Placeholder for TradeSteward
    # elif source_format == 'trade_steward':
    #     return process_trade_steward(df)
    else:
        raise ValueError(f"Unknown format: {source_format}")


# Placeholder for TradeSteward column mapping
TRADE_STEWARD_COLUMN_MAP = {
    # 'trade_steward_col': 'standard_col'
    # To be defined when format is provided
}
