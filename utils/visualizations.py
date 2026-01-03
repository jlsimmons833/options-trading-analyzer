"""
Visualization utilities using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import COLORS, COLOR_SCALE_DIVERGING, CHART_HEIGHT, HEATMAP_HEIGHT


def create_ev_heatmap(matrix_df, title="Expected Value by Strategy", x_title="", y_title="Strategy"):
    """
    Create a heatmap showing EV across strategies and a dimension (quarters, VIX regimes, etc.).
    """
    fig = px.imshow(
        matrix_df,
        color_continuous_scale=COLOR_SCALE_DIVERGING,
        color_continuous_midpoint=0,
        labels=dict(x=x_title, y=y_title, color="Expected Value ($)"),
        aspect='auto',
    )

    fig.update_layout(
        title=title,
        height=HEATMAP_HEIGHT,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )

    # Add text annotations
    for i, strategy in enumerate(matrix_df.index):
        for j, col in enumerate(matrix_df.columns):
            val = matrix_df.loc[strategy, col]
            if not pd.isna(val):
                text_color = 'white' if abs(val) > matrix_df.abs().max().max() * 0.5 else 'black'
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"${val:.0f}",
                    showarrow=False,
                    font=dict(color=text_color, size=10),
                )

    return fig


def create_ev_bar_chart(metrics_df, title="Expected Value by Strategy"):
    """
    Create a horizontal bar chart showing EV by strategy.
    """
    df = metrics_df.sort_values('Expected Value', ascending=True)

    colors = [COLORS['positive'] if ev > 0 else COLORS['negative'] for ev in df['Expected Value']]

    fig = go.Figure(go.Bar(
        x=df['Expected Value'],
        y=df['Strategy'],
        orientation='h',
        marker_color=colors,
        text=[f"${ev:.2f}" for ev in df['Expected Value']],
        textposition='outside',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Expected Value ($)",
        yaxis_title="Strategy",
        height=max(400, len(df) * 30),
        showlegend=False,
    )

    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS['neutral'])

    return fig


def create_pie_chart(df, title="Trade Distribution by Strategy"):
    """
    Create a pie chart showing trade distribution.
    """
    strategy_counts = df['Strategy'].value_counts()

    fig = px.pie(
        values=strategy_counts.values,
        names=strategy_counts.index,
        title=title,
        hole=0.3,
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=CHART_HEIGHT)

    return fig


def create_rolling_ev_chart(df, strategies, window_size=30, title="Rolling Expected Value"):
    """
    Create a line chart showing rolling EV over time for selected strategies.
    """
    from .calculations import calculate_rolling_ev

    fig = go.Figure()

    for strategy in strategies:
        strategy_data = df[df['Strategy'] == strategy].copy()
        strategy_data = strategy_data.sort_values('Date Opened')

        if len(strategy_data) < window_size // 2:
            continue

        # Calculate rolling EV
        rolling_ev = strategy_data['P/L'].rolling(
            window=window_size,
            min_periods=window_size // 2
        ).apply(calculate_rolling_ev)

        fig.add_trace(go.Scatter(
            x=strategy_data['Date Opened'],
            y=rolling_ev,
            name=strategy,
            mode='lines',
            hovertemplate='%{y:$.2f}<extra>%{fullData.name}</extra>',
        ))

    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'])

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rolling Expected Value ($)",
        height=CHART_HEIGHT,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig


def create_equity_curve(df, title="Portfolio Equity Curve"):
    """
    Create an equity curve chart from trade data.
    """
    portfolio = df.sort_values('Date Closed').copy()
    portfolio['Cumulative P/L'] = portfolio['P/L'].cumsum()

    fig = go.Figure()

    # Main equity curve
    fig.add_trace(go.Scatter(
        x=portfolio['Date Closed'],
        y=portfolio['Cumulative P/L'],
        mode='lines',
        name='Cumulative P/L',
        fill='tozeroy',
        fillcolor='rgba(40, 167, 69, 0.3)' if portfolio['Cumulative P/L'].iloc[-1] > 0 else 'rgba(220, 53, 69, 0.3)',
        line=dict(color=COLORS['positive'] if portfolio['Cumulative P/L'].iloc[-1] > 0 else COLORS['negative']),
        hovertemplate='$%{y:,.2f}<extra></extra>',
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'])

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative P/L ($)",
        height=CHART_HEIGHT,
        showlegend=False,
    )

    return fig


def create_scatter_with_trendline(df, x_col, y_col, strategy=None, title=""):
    """
    Create a scatter plot with trendline.
    """
    if strategy:
        plot_df = df[df['Strategy'] == strategy].copy()
        title = f"{title} - {strategy}"
    else:
        plot_df = df.copy()

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        trendline="ols",
        title=title,
        color_discrete_sequence=[COLORS['highlight']],
    )

    fig.update_layout(
        height=400,
        xaxis_title=x_col,
        yaxis_title=y_col,
    )

    # Color points by P/L
    fig.update_traces(
        marker=dict(
            color=[COLORS['positive'] if pl > 0 else COLORS['negative'] for pl in plot_df[y_col]],
            size=8,
            opacity=0.6,
        ),
        selector=dict(mode='markers')
    )

    return fig


def create_correlation_heatmap(corr_df, title="Strategy Correlation Matrix"):
    """
    Create a correlation heatmap between strategies.
    """
    fig = px.imshow(
        corr_df,
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
        labels=dict(color="Correlation"),
        aspect='auto',
        zmin=-1,
        zmax=1,
    )

    fig.update_layout(
        title=title,
        height=500,
    )

    # Add text annotations
    for i, row in enumerate(corr_df.index):
        for j, col in enumerate(corr_df.columns):
            val = corr_df.loc[row, col]
            if not pd.isna(val):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color='white' if abs(val) > 0.5 else 'black', size=10),
                )

    return fig


def create_quarterly_comparison_chart(df, strategies, title="Quarterly Performance Comparison"):
    """
    Create a grouped bar chart comparing strategies across quarters.
    """
    from .calculations import calculate_expected_value

    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    data = []

    for strategy in strategies:
        for quarter in quarters:
            quarter_df = df[(df['Strategy'] == strategy) & (df['Quarter'] == quarter)]
            if len(quarter_df) >= 5:
                ev = calculate_expected_value(quarter_df)
                data.append({
                    'Strategy': strategy,
                    'Quarter': quarter,
                    'Expected Value': ev,
                })

    if not data:
        return go.Figure()

    plot_df = pd.DataFrame(data)

    fig = px.bar(
        plot_df,
        x='Quarter',
        y='Expected Value',
        color='Strategy',
        barmode='group',
        title=title,
    )

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'])

    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis_title="Quarter",
        yaxis_title="Expected Value ($)",
    )

    return fig


def create_trend_indicator(trend_direction, trend_strength):
    """
    Create a visual trend indicator.
    Returns HTML for display.
    """
    if trend_direction == 'up':
        arrow = "↑"
        color = COLORS['positive']
    elif trend_direction == 'down':
        arrow = "↓"
        color = COLORS['negative']
    else:
        arrow = "→"
        color = COLORS['neutral']

    return f'<span style="color: {color}; font-size: 20px;">{arrow}</span> {trend_strength:.1f}%'


def create_prediction_chart(predictions, title="Predicted Strategy Performance"):
    """
    Create a bar chart showing predicted EV for each strategy.
    """
    strategies = list(predictions.keys())
    evs = list(predictions.values())

    colors = [COLORS['positive'] if ev > 0 else COLORS['negative'] for ev in evs]

    fig = go.Figure(go.Bar(
        x=strategies,
        y=evs,
        marker_color=colors,
        text=[f"${ev:.2f}" for ev in evs],
        textposition='outside',
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['neutral'])

    fig.update_layout(
        title=title,
        xaxis_title="Strategy",
        yaxis_title="Predicted Expected Value ($)",
        height=400,
    )

    return fig


def create_drawdown_chart(df, title="Drawdown Analysis"):
    """
    Create a chart showing drawdown over time.
    """
    portfolio = df.sort_values('Date Closed').copy()
    portfolio['Cumulative P/L'] = portfolio['P/L'].cumsum()

    # Calculate drawdown
    peak = portfolio['Cumulative P/L'].expanding(min_periods=1).max()
    drawdown = portfolio['Cumulative P/L'] - peak

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio['Date Closed'],
        y=drawdown,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(220, 53, 69, 0.3)',
        line=dict(color=COLORS['negative']),
        name='Drawdown',
        hovertemplate='$%{y:,.2f}<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown ($)",
        height=300,
        showlegend=False,
    )

    return fig


def create_win_loss_chart(df, title="Win/Loss Distribution"):
    """
    Create a histogram showing P/L distribution.
    """
    fig = go.Figure()

    wins = df[df['P/L'] > 0]['P/L']
    losses = df[df['P/L'] <= 0]['P/L']

    fig.add_trace(go.Histogram(
        x=wins,
        name='Wins',
        marker_color=COLORS['positive'],
        opacity=0.7,
    ))

    fig.add_trace(go.Histogram(
        x=losses,
        name='Losses',
        marker_color=COLORS['negative'],
        opacity=0.7,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="P/L ($)",
        yaxis_title="Count",
        barmode='overlay',
        height=400,
    )

    return fig
