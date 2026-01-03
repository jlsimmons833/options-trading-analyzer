"""
Utility modules for the Trading Strategy Analyzer.
"""

from .data_processing import load_and_process_data, detect_file_format
from .calculations import (
    calculate_expected_value,
    calculate_strategy_metrics,
    calculate_rolling_ev,
    get_quarter,
    categorize_vix,
)
from .visualizations import (
    create_ev_heatmap,
    create_ev_bar_chart,
    create_pie_chart,
    create_rolling_ev_chart,
    create_equity_curve,
    create_scatter_with_trendline,
)
from .filters import (
    initialize_session_state,
    apply_filters,
    render_sidebar_filters,
)
