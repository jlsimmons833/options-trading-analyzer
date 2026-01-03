"""
Configuration constants for the Trading Strategy Analyzer.
"""

# Color scheme
COLORS = {
    'positive': '#28a745',
    'negative': '#dc3545',
    'neutral': '#6c757d',
    'highlight': '#007bff',
    'background': '#f8f9fa',
    'white': '#ffffff',
}

# Plotly color scales
COLOR_SCALE_DIVERGING = 'RdYlGn'

# VIX regime thresholds
VIX_THRESHOLDS = {
    'low': 15,
    'normal': 20,
    'elevated': 25,
    'high': 30,
}

VIX_REGIME_LABELS = {
    'low': 'Low (<15)',
    'normal': 'Normal (15-20)',
    'elevated': 'Elevated (20-25)',
    'high': 'High (25-30)',
    'extreme': 'Extreme (>30)',
}

# Rolling window options
ROLLING_WINDOWS = [30, 60, 90]

# Minimum trades for statistical significance
MIN_TRADES_FOR_STATS = 5

# Chart dimensions
CHART_HEIGHT = 500
HEATMAP_HEIGHT = 600

# Quarters
QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']

# Required columns for Options Omega format
REQUIRED_COLUMNS = [
    'Date Opened',
    'Time Opened',
    'Opening Price',
    'Legs',
    'Premium',
    'Closing Price',
    'Date Closed',
    'Time Closed',
    'Avg. Closing Cost',
    'Reason For Close',
    'P/L',
    'P/L %',
    'No. of Contracts',
    'Funds at Close',
    'Margin Req.',
    'Strategy',
    'Opening Commissions + Fees',
    'Closing Commissions + Fees',
    'Opening Short/Long Ratio',
    'Closing Short/Long Ratio',
    'Opening VIX',
    'Closing VIX',
    'Gap',
    'Movement',
    'Max Profit',
    'Max Loss',
]

# Page configuration
PAGE_CONFIG = {
    'page_title': 'Trading Strategy Analyzer',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}
