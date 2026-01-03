"""
Trading Strategy Analyzer - Main Application
Analyzes options trading performance from Options Omega export files.
"""

import streamlit as st
from config import PAGE_CONFIG

# Page configuration
st.set_page_config(
    page_title=PAGE_CONFIG['page_title'],
    page_icon=PAGE_CONFIG['page_icon'],
    layout=PAGE_CONFIG['layout'],
    initial_sidebar_state=PAGE_CONFIG['initial_sidebar_state'],
)

# Main page content
st.title("Trading Strategy Analyzer")

st.markdown("""
Welcome to the **Trading Strategy Analyzer**. This application helps you analyze
options trading performance from Options Omega export files.

### Features

- **Dashboard**: Upload data and view summary metrics
- **Quarterly Analysis**: Analyze strategy performance by calendar quarter
- **Performance Trends**: Track rolling performance and identify trends
- **Regime Analysis**: Correlate performance with VIX and market conditions
- **Portfolio Builder**: Construct optimized strategy portfolios

### Getting Started

1. Navigate to the **Dashboard** page using the sidebar
2. Upload your Options Omega CSV export file
3. Explore the various analysis pages

---

*Select a page from the sidebar to begin.*
""")

# Show navigation hint
st.sidebar.success("Select a page above to get started.")

# Display info about data persistence
if 'data_loaded' not in st.session_state:
    st.info("No data loaded. Please go to the Dashboard to upload your trading data.")
else:
    st.success("Data loaded! Navigate to any page to analyze your trades.")
