"""
Regime Analysis Page - VIX correlation and market condition analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PAGE_CONFIG, VIX_REGIME_LABELS, MIN_TRADES_FOR_STATS
from utils.calculations import (
    calculate_expected_value,
    calculate_vix_regime_matrix,
    calculate_correlation_metrics,
)
from utils.visualizations import (
    create_ev_heatmap,
    create_scatter_with_trendline,
    create_prediction_chart,
)
from utils.filters import render_sidebar_filters, apply_filters
from utils.auth import check_authentication, render_user_info_sidebar

st.set_page_config(
    page_title=f"Regime Analysis - {PAGE_CONFIG['page_title']}",
    page_icon=PAGE_CONFIG['page_icon'],
    layout=PAGE_CONFIG['layout'],
)

# Auth check
if not check_authentication():
    st.stop()

render_user_info_sidebar()

st.title("Regime Analysis")

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

# VIX Regime Analysis
st.header("VIX Regime Analysis")

st.markdown("""
Analyze how strategies perform under different VIX (volatility) conditions.
VIX regimes are defined as:
- **Low (<15)**: Calm markets
- **Normal (15-20)**: Average volatility
- **Elevated (20-25)**: Increased uncertainty
- **High (25-30)**: Significant volatility
- **Extreme (>30)**: Market stress/crisis
""")

# VIX Regime Heat Map
vix_matrix = calculate_vix_regime_matrix(filtered_df)

heatmap_fig = create_ev_heatmap(
    vix_matrix,
    title="Expected Value by Strategy and VIX Regime",
    x_title="VIX Regime",
    y_title="Strategy",
)
st.plotly_chart(heatmap_fig, use_container_width=True)

# VIX Regime Summary
st.subheader("VIX Regime Summary")

vix_regimes = ['Low (<15)', 'Normal (15-20)', 'Elevated (20-25)', 'High (25-30)', 'Extreme (>30)']

regime_summary = []
for regime in vix_regimes:
    regime_df = filtered_df[filtered_df['VIX_Regime'] == regime]
    if len(regime_df) >= MIN_TRADES_FOR_STATS:
        ev = calculate_expected_value(regime_df)
        win_rate = (regime_df['P/L'] > 0).mean() * 100
        regime_summary.append({
            'VIX Regime': regime,
            'Trades': len(regime_df),
            'Win Rate': f"{win_rate:.1f}%",
            'Avg P/L': f"${regime_df['P/L'].mean():.2f}",
            'Expected Value': f"${ev:.2f}",
        })

if regime_summary:
    st.dataframe(
        pd.DataFrame(regime_summary),
        use_container_width=True,
        hide_index=True,
    )

# Correlation Dashboard
st.header("Correlation Dashboard")

st.markdown("""
Analyze correlations between market factors (VIX, Gap, Movement) and trade P/L.
""")

# Strategy selector for correlation analysis
corr_strategy = st.selectbox(
    "Select strategy for correlation analysis",
    options=['All Strategies'] + sorted(filtered_df['Strategy'].unique().tolist()),
)

if corr_strategy == 'All Strategies':
    corr_df = filtered_df
else:
    corr_df = filtered_df[filtered_df['Strategy'] == corr_strategy]

if len(corr_df) < MIN_TRADES_FOR_STATS:
    st.info("Not enough data for correlation analysis.")
else:
    # Scatter plots with trend lines
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("VIX vs P/L")
        vix_scatter = create_scatter_with_trendline(
            corr_df, 'Opening VIX', 'P/L',
            title="VIX vs P/L"
        )
        st.plotly_chart(vix_scatter, use_container_width=True)

    with col2:
        st.subheader("Gap vs P/L")
        gap_scatter = create_scatter_with_trendline(
            corr_df, 'Gap', 'P/L',
            title="Gap vs P/L"
        )
        st.plotly_chart(gap_scatter, use_container_width=True)

    with col3:
        st.subheader("Movement vs P/L")
        movement_scatter = create_scatter_with_trendline(
            corr_df, 'Movement', 'P/L',
            title="Movement vs P/L"
        )
        st.plotly_chart(movement_scatter, use_container_width=True)

    # Correlation coefficients
    st.subheader("Correlation Coefficients")

    corr_data = []
    strategies_to_analyze = filtered_df['Strategy'].unique() if corr_strategy == 'All Strategies' else [corr_strategy]

    for strategy in strategies_to_analyze:
        metrics = calculate_correlation_metrics(filtered_df, strategy)
        if not np.isnan(metrics['VIX Correlation']):
            corr_data.append({
                'Strategy': strategy,
                'VIX Correlation': f"{metrics['VIX Correlation']:.3f}",
                'Gap Correlation': f"{metrics['Gap Correlation']:.3f}",
                'Movement Correlation': f"{metrics['Movement Correlation']:.3f}",
            })

    if corr_data:
        st.dataframe(
            pd.DataFrame(corr_data),
            use_container_width=True,
            hide_index=True,
        )

# Predictive Model Panel
st.header("Predictive Model")

st.markdown("""
Simple linear regression models to predict strategy performance based on market conditions.
""")

# Model strategy selector
model_strategy = st.selectbox(
    "Select strategy for predictive model",
    options=sorted(filtered_df['Strategy'].unique().tolist()),
    key='model_strategy',
)

model_df = filtered_df[filtered_df['Strategy'] == model_strategy].copy()

if len(model_df) < MIN_TRADES_FOR_STATS * 2:
    st.info(f"Not enough data for {model_strategy} to build a predictive model.")
else:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # Prepare features
    feature_cols = ['Opening VIX', 'Gap', 'Movement']
    model_df_clean = model_df.dropna(subset=feature_cols + ['P/L'])

    if len(model_df_clean) < MIN_TRADES_FOR_STATS:
        st.info("Not enough clean data for model training.")
    else:
        X = model_df_clean[feature_cols]
        y = model_df_clean['P/L']

        model = LinearRegression()
        model.fit(X, y)

        # Display model coefficients
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Coefficients")
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_,
            })
            coef_df['Interpretation'] = coef_df.apply(
                lambda row: f"${row['Coefficient']:.2f} per unit increase",
                axis=1
            )
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Model Quality")
            r2 = r2_score(y, model.predict(X))
            st.metric("R² Score", f"{r2:.3f}")
            st.metric("Intercept", f"${model.intercept_:.2f}")

            if r2 < 0.1:
                st.warning("Low R² indicates market factors explain little of the P/L variance.")
            elif r2 < 0.3:
                st.info("Moderate R² - some predictive power from market factors.")
            else:
                st.success("Good R² - market factors have significant predictive power.")

        # "What If" Scenario Tool
        st.subheader("What-If Scenario Tool")

        st.markdown("Enter expected market conditions to predict strategy performance:")

        col1, col2, col3 = st.columns(3)

        with col1:
            expected_vix = st.slider(
                "Expected VIX",
                min_value=10.0,
                max_value=50.0,
                value=20.0,
                step=1.0,
            )

        with col2:
            expected_gap = st.slider(
                "Expected Gap (%)",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
            )

        with col3:
            expected_movement = st.slider(
                "Expected Movement (%)",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
            )

        # Predict for selected strategy
        prediction_input = np.array([[expected_vix, expected_gap, expected_movement]])
        predicted_pnl = model.predict(prediction_input)[0]

        st.metric(
            f"Predicted P/L for {model_strategy}",
            f"${predicted_pnl:.2f}",
        )

        # Predict for all strategies
        st.subheader("Predictions for All Strategies")

        all_predictions = {}
        for strategy in filtered_df['Strategy'].unique():
            strategy_df = filtered_df[filtered_df['Strategy'] == strategy].dropna(subset=feature_cols + ['P/L'])

            if len(strategy_df) >= MIN_TRADES_FOR_STATS * 2:
                X_strat = strategy_df[feature_cols]
                y_strat = strategy_df['P/L']

                strat_model = LinearRegression()
                strat_model.fit(X_strat, y_strat)

                pred = strat_model.predict(prediction_input)[0]
                all_predictions[strategy] = pred

        if all_predictions:
            # Sort by predicted EV
            sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))

            prediction_chart = create_prediction_chart(
                sorted_predictions,
                title=f"Predicted P/L (VIX={expected_vix}, Gap={expected_gap}%, Movement={expected_movement}%)",
            )
            st.plotly_chart(prediction_chart, use_container_width=True)

            # Top recommendations
            st.subheader("Strategy Recommendations")
            top_3 = list(sorted_predictions.items())[:3]
            st.success(f"**Top 3 strategies for current conditions:** {', '.join([f'{s} (${p:.2f})' for s, p in top_3])}")

# VIX Regime Persistence Analysis
st.header("VIX Regime Persistence Analysis")

st.markdown("""
Analyze how strategies perform when VIX has been stable for extended periods.
""")

# Calculate VIX stability periods
def identify_stable_vix_periods(df, stability_days=30, vix_range=3):
    """Identify periods where VIX was stable within a range for N days."""
    df = df.sort_values('Date Opened').copy()
    df['VIX_Rolling_Std'] = df['Opening VIX'].rolling(window=stability_days, min_periods=stability_days//2).std()
    df['VIX_Stable'] = df['VIX_Rolling_Std'] < vix_range
    return df

stability_days = st.slider(
    "Stability period (days)",
    min_value=10,
    max_value=60,
    value=30,
    step=5,
)

stable_df = identify_stable_vix_periods(filtered_df, stability_days=stability_days)
stable_periods = stable_df[stable_df['VIX_Stable'] == True]

if len(stable_periods) < MIN_TRADES_FOR_STATS:
    st.info(f"Not enough trades during stable VIX periods ({stability_days} days).")
else:
    st.subheader(f"Performance During Stable VIX Periods ({stability_days}+ days)")

    # Group by VIX level during stable periods
    stable_periods['VIX_Level'] = pd.cut(
        stable_periods['Opening VIX'],
        bins=[0, 15, 20, 25, 30, 100],
        labels=['~15 (Low)', '~17.5 (Normal)', '~22.5 (Elevated)', '~27.5 (High)', '>30 (Extreme)']
    )

    stability_summary = []
    for level in stable_periods['VIX_Level'].unique():
        if pd.isna(level):
            continue
        level_df = stable_periods[stable_periods['VIX_Level'] == level]
        if len(level_df) >= MIN_TRADES_FOR_STATS:
            ev = calculate_expected_value(level_df)
            stability_summary.append({
                'VIX Level': level,
                'Trades': len(level_df),
                'Win Rate': f"{(level_df['P/L'] > 0).mean() * 100:.1f}%",
                'Expected Value': f"${ev:.2f}",
            })

    if stability_summary:
        st.dataframe(
            pd.DataFrame(stability_summary),
            use_container_width=True,
            hide_index=True,
        )

    # Strategy-specific performance during stable periods
    st.subheader("Best Strategies During Stable VIX")

    stable_strategy_metrics = []
    for strategy in filtered_df['Strategy'].unique():
        strategy_stable = stable_periods[stable_periods['Strategy'] == strategy]
        if len(strategy_stable) >= MIN_TRADES_FOR_STATS:
            ev = calculate_expected_value(strategy_stable)
            stable_strategy_metrics.append({
                'Strategy': strategy,
                'Stable Period Trades': len(strategy_stable),
                'Win Rate': (strategy_stable['P/L'] > 0).mean() * 100,
                'Expected Value': ev,
            })

    if stable_strategy_metrics:
        stable_metrics_df = pd.DataFrame(stable_strategy_metrics)
        stable_metrics_df = stable_metrics_df.sort_values('Expected Value', ascending=False)
        stable_metrics_df['Win Rate'] = stable_metrics_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
        stable_metrics_df['Expected Value'] = stable_metrics_df['Expected Value'].apply(lambda x: f"${x:.2f}")

        st.dataframe(
            stable_metrics_df,
            use_container_width=True,
            hide_index=True,
        )
