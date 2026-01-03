# Options Trading Strategy Analyzer - Development Instructions

## Project Overview

Build a Streamlit-based web application to analyze options trading performance from Options Omega export files. The app helps identify which trading strategies perform best during specific calendar quarters and market regimes, enabling the construction of optimized strategy portfolios.

---

## Data Format Specification

### Source: Options Omega CSV Export

**Required Columns (26 total):**

| Column | Type | Description |
|--------|------|-------------|
| `Date Opened` | date | Trade entry date (YYYY-MM-DD) |
| `Time Opened` | time | Trade entry time |
| `Opening Price` | float | Underlying price at open |
| `Legs` | string | Option contract details |
| `Premium` | float | Net premium received/paid |
| `Closing Price` | float | Underlying price at close |
| `Date Closed` | date | Trade exit date |
| `Time Closed` | time | Trade exit time |
| `Avg. Closing Cost` | float | Average closing cost |
| `Reason For Close` | string | Exit reason (Expired, Profit Target, Backtest Completed, etc.) |
| `P/L` | float | Profit/Loss in dollars |
| `P/L %` | float | Profit/Loss percentage |
| `No. of Contracts` | int | Number of contracts |
| `Funds at Close` | float | Account value at close |
| `Margin Req.` | float | Margin requirement |
| `Strategy` | string | Strategy name identifier |
| `Opening Commissions + Fees` | float | Entry costs |
| `Closing Commissions + Fees` | float | Exit costs |
| `Opening Short/Long Ratio` | float | Position ratio at open |
| `Closing Short/Long Ratio` | float | Position ratio at close |
| `Opening VIX` | float | VIX level at trade open |
| `Closing VIX` | float | VIX level at trade close |
| `Gap` | float | Market gap percentage |
| `Movement` | float | Market movement during trade |
| `Max Profit` | float | Maximum potential profit % |
| `Max Loss` | float | Maximum potential loss % |

**Sample Data Characteristics:**
- ~4,700+ trades spanning May 2022 to December 2025
- 25 unique strategy names
- Mix of 0DTE, multi-day, and 22DTE butterfly strategies

---

## Core Calculations

### Expected Value (EV) Formula

This is the PRIMARY metric used throughout the application:

```python
def calculate_expected_value(trades_df):
    """
    Calculate Expected Value for a set of trades.
    EV = (Win Rate × Average Win) - (Loss Rate × Average Loss)
    
    Returns EV in dollars.
    """
    wins = trades_df[trades_df['P/L'] > 0]
    losses = trades_df[trades_df['P/L'] <= 0]
    
    total_trades = len(trades_df)
    if total_trades == 0:
        return 0
    
    win_rate = len(wins) / total_trades
    loss_rate = len(losses) / total_trades
    
    avg_win = wins['P/L'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['P/L'].mean()) if len(losses) > 0 else 0
    
    expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)
    return expected_value
```

### Calendar Quarter Assignment

```python
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
```

---

## Application Architecture

### Page Structure

```
📊 Trading Strategy Analyzer
├── 📈 Dashboard (Home)
├── 🗓️ Quarterly Analysis
├── 📉 Performance Trends
├── 🔬 Regime Analysis
└── 💼 Portfolio Builder
```

---

## Page 1: Dashboard (Home)

### Purpose
High-level summary view with key metrics and quick insights.

### Components

#### 1.1 File Upload Section
```python
uploaded_file = st.file_uploader(
    "Upload Options Omega CSV",
    type=['csv'],
    help="Export your trades from Options Omega in CSV format"
)
```

#### 1.2 Global Filters Sidebar
- **Date Range Selector**: Start and end date pickers
- **Strategy Multi-Select**: Checkbox list of all strategies (select/deselect all option)
- **Trade Outcome Filter**: Multi-select for "Reason For Close" values
  - Expired
  - Profit Target
  - Backtest Completed
  - Exited at Specified Time
  - (others as detected in data)

#### 1.3 Summary Metrics Row
Display as `st.metric()` cards:
- Total Trades (filtered)
- Overall Win Rate %
- Total P/L $
- Overall Expected Value $
- Best Performing Strategy (by EV)
- Worst Performing Strategy (by EV)

#### 1.4 Strategy Performance Table
Sortable dataframe showing for each strategy:
| Strategy | Trades | Win Rate | Avg Win | Avg Loss | Total P/L | Expected Value |
|----------|--------|----------|---------|----------|-----------|----------------|

#### 1.5 Quick Visualizations
- **Pie Chart**: Trade distribution by strategy
- **Bar Chart**: EV by strategy (horizontal, sorted)

---

## Page 2: Quarterly Analysis

### Purpose
Analyze how strategies perform across calendar quarters. Test the hypothesis that certain strategies perform better during specific quarters due to business/tax cycles.

### Components

#### 2.1 Heat Map - Strategy × Quarter Performance

**X-axis**: Quarters (Q1, Q2, Q3, Q4)
**Y-axis**: Strategies (all 25)
**Cell Value**: Expected Value
**Color Scale**: Diverging (red for negative EV, white for zero, green for positive EV)

```python
import plotly.express as px

# Create pivot table: rows=Strategy, columns=Quarter, values=EV
pivot_data = df.pivot_table(
    index='Strategy',
    columns='Quarter',
    values='P/L',
    aggfunc=calculate_ev_for_group  # Custom aggregation
)

fig = px.imshow(
    pivot_data,
    color_continuous_scale='RdYlGn',
    color_continuous_midpoint=0,
    labels=dict(x="Quarter", y="Strategy", color="Expected Value ($)")
)
```

#### 2.2 Year Filter for Heat Map
- Dropdown: "All Years", "2022", "2023", "2024", "2025"
- Dynamically regenerates heat map for selected year(s)

#### 2.3 Quarter Deep-Dive Panel
When user clicks a quarter column header or selects from dropdown:
- Show ranked list of strategies for that quarter
- Display detailed metrics for top 5 and bottom 5 strategies
- Mini time-series showing that quarter's performance across years

#### 2.4 Quarter Comparison Table
Side-by-side comparison:
| Strategy | Q1 EV | Q2 EV | Q3 EV | Q4 EV | Best Quarter | Worst Quarter |
|----------|-------|-------|-------|-------|--------------|---------------|

---

## Page 3: Performance Trends

### Purpose
Visualize whether strategies are improving or degrading over time using rolling windows.

### Components

#### 3.1 Rolling Window Selector
```python
window_size = st.radio(
    "Rolling Window",
    options=[30, 60, 90],
    format_func=lambda x: f"{x} Days",
    horizontal=True
)
```

#### 3.2 Strategy Selector
Multi-select to choose which strategies to display (limit to 5-7 for readability)

#### 3.3 Rolling EV Line Chart
**Primary Visualization**

```python
import plotly.graph_objects as go

fig = go.Figure()

for strategy in selected_strategies:
    strategy_data = df[df['Strategy'] == strategy].copy()
    strategy_data = strategy_data.sort_values('Date Opened')
    
    # Calculate rolling EV
    rolling_ev = strategy_data['P/L'].rolling(
        window=window_size,
        min_periods=window_size // 2
    ).apply(calculate_rolling_ev)
    
    fig.add_trace(go.Scatter(
        x=strategy_data['Date Opened'],
        y=rolling_ev,
        name=strategy,
        mode='lines'
    ))

# Add zero reference line
fig.add_hline(y=0, line_dash="dash", line_color="gray")
```

#### 3.4 Trend Indicators
For each strategy, calculate and display:
- **Trend Direction**: Arrow up/down/neutral based on slope of last 90 days
- **Trend Strength**: Percentage change from 6 months ago to now
- **Volatility**: Standard deviation of rolling EV

#### 3.5 Performance Regime Bands
Add shaded regions to chart indicating:
- High performance periods (EV consistently above threshold)
- Drawdown periods (EV consistently negative)
- Recovery periods (transitioning from negative to positive)

---

## Page 4: Regime Analysis

### Purpose
Explore correlations between strategy performance and market conditions (VIX, gaps, movement). Enable predictive insights.

### Components

#### 4.1 VIX Regime Analysis

**VIX Bucketing:**
```python
def categorize_vix(vix_value):
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
```

**Heat Map: Strategy × VIX Regime**
- Same format as quarterly heat map
- Cells show EV for each strategy in each VIX bucket

#### 4.2 Correlation Dashboard

**Scatter Plots with Trend Lines:**
1. Opening VIX vs P/L (by strategy)
2. Gap vs P/L (by strategy)
3. Movement vs P/L (by strategy)

**Correlation Coefficients Table:**
| Strategy | VIX Correlation | Gap Correlation | Movement Correlation |
|----------|-----------------|-----------------|---------------------|

#### 4.3 Predictive Model Panel

**Simple Regression Models:**
For each strategy, fit and display:
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Features: Opening VIX, Gap, Movement
X = df[['Opening VIX', 'Gap', 'Movement']]
y = df['P/L']

model = LinearRegression()
model.fit(X, y)

# Display coefficients and R²
st.write(f"VIX coefficient: {model.coef_[0]:.2f}")
st.write(f"Gap coefficient: {model.coef_[1]:.2f}")
st.write(f"Movement coefficient: {model.coef_[2]:.2f}")
st.write(f"R² Score: {r2_score(y, model.predict(X)):.3f}")
```

#### 4.4 "What If" Scenario Tool
User inputs:
- Expected VIX level
- Expected gap direction/magnitude
- Expected movement range

App outputs:
- Predicted EV for each strategy under those conditions
- Ranked recommendation of strategies to deploy

#### 4.5 VIX Regime Persistence Analysis
Show how strategies perform when VIX has been stable for extended periods:
- "VIX stable at ~15 for 30+ days"
- "VIX stable at ~20 for 30+ days"
- Calculate performance specifically during these stable periods

---

## Page 5: Portfolio Builder

### Purpose
Select and combine strategies to build an optimized portfolio for specific quarters or regimes.

### Components

#### 5.1 Scenario Selector
```python
col1, col2 = st.columns(2)

with col1:
    target_quarter = st.selectbox(
        "Target Quarter",
        options=['Q1', 'Q2', 'Q3', 'Q4', 'All Quarters']
    )

with col2:
    year_filter = st.selectbox(
        "Year Range",
        options=['All Time', '2022', '2023', '2024', '2025', 'Last 12 Months']
    )
```

#### 5.2 Strategy Performance for Selected Scenario
Ranked table of all strategies for the selected quarter/time period:
| Rank | Strategy | EV | Win Rate | Trades | Sharpe-like Ratio |

#### 5.3 Strategy Selection Interface
```python
st.subheader("Build Your Portfolio")

selected_strategies = st.multiselect(
    "Select strategies to include in portfolio",
    options=all_strategies,
    default=top_5_by_ev
)
```

#### 5.4 Portfolio Analysis Panel

**Combined Portfolio Metrics:**
- Total trades if all selected strategies run
- Combined Expected Value (sum or weighted)
- Portfolio Win Rate
- Max Drawdown estimate
- Diversification score (correlation between selected strategies)

**Portfolio Equity Curve:**
Simulated equity curve if running all selected strategies:
```python
# Combine all trades from selected strategies
portfolio_trades = df[df['Strategy'].isin(selected_strategies)]
portfolio_trades = portfolio_trades.sort_values('Date Closed')
portfolio_trades['Cumulative P/L'] = portfolio_trades['P/L'].cumsum()

fig = px.line(
    portfolio_trades,
    x='Date Closed',
    y='Cumulative P/L',
    title='Portfolio Equity Curve'
)
```

#### 5.5 Quarterly Rotation Strategy
Suggest automatic strategy rotation:
```
Recommended Quarterly Rotation:
- Q1: [Strategy A, Strategy B, Strategy C] → Combined EV: $X
- Q2: [Strategy D, Strategy E] → Combined EV: $Y
- Q3: [Strategy A, Strategy F] → Combined EV: $Z
- Q4: [Strategy B, Strategy G, Strategy H] → Combined EV: $W

Full Year Projected EV: $Total
```

---

## Technical Implementation Details

### Required Python Packages

```python
# requirements.txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

### Data Processing Pipeline

```python
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
    
    return df
```

### Session State Management

```python
# Initialize session state for filters
if 'date_range' not in st.session_state:
    st.session_state.date_range = (df['Date Opened'].min(), df['Date Opened'].max())
    
if 'selected_strategies' not in st.session_state:
    st.session_state.selected_strategies = df['Strategy'].unique().tolist()
    
if 'selected_outcomes' not in st.session_state:
    st.session_state.selected_outcomes = df['Reason For Close'].unique().tolist()
```

### Caching for Performance

```python
@st.cache_data
def calculate_strategy_metrics(df):
    """
    Cache expensive calculations.
    """
    metrics = []
    for strategy in df['Strategy'].unique():
        strategy_df = df[df['Strategy'] == strategy]
        metrics.append({
            'Strategy': strategy,
            'Trades': len(strategy_df),
            'Win_Rate': (strategy_df['P/L'] > 0).mean() * 100,
            'Avg_Win': strategy_df[strategy_df['P/L'] > 0]['P/L'].mean(),
            'Avg_Loss': strategy_df[strategy_df['P/L'] <= 0]['P/L'].mean(),
            'Total_PL': strategy_df['P/L'].sum(),
            'EV': calculate_expected_value(strategy_df)
        })
    return pd.DataFrame(metrics)
```

---

## UI/UX Guidelines

### Color Scheme
- **Positive values**: Green (#28a745)
- **Negative values**: Red (#dc3545)
- **Neutral**: Gray (#6c757d)
- **Highlights**: Blue (#007bff)

### Chart Standards
- Use Plotly for all interactive charts
- Consistent hover templates showing key metrics
- Include export buttons for charts
- Responsive sizing

### Filter Persistence
- Filters should persist across page navigation
- "Reset Filters" button on each page
- Show active filter count in sidebar

### Loading States
```python
with st.spinner('Calculating metrics...'):
    # Expensive operation
    results = heavy_computation()
```

---

## Future Extensibility

### TradeSteward Import (Planned)
Reserve architecture for additional file format:
```python
def detect_file_format(uploaded_file):
    """
    Detect whether file is Options Omega or TradeSteward format.
    """
    df = pd.read_csv(uploaded_file, nrows=1)
    
    if 'Strategy' in df.columns and 'Opening VIX' in df.columns:
        return 'options_omega'
    elif 'trade_steward_column' in df.columns:  # Placeholder
        return 'trade_steward'
    else:
        return 'unknown'

def normalize_to_standard_format(df, source_format):
    """
    Convert any format to internal standard format.
    """
    if source_format == 'options_omega':
        return process_options_omega(df)
    elif source_format == 'trade_steward':
        return process_trade_steward(df)  # To be implemented
```

### Column Mapping for TradeSteward
Leave placeholder for mapping configuration:
```python
TRADE_STEWARD_COLUMN_MAP = {
    # 'trade_steward_col': 'standard_col'
    # To be defined when format is provided
}
```

---

## Testing Checklist

### Data Validation
- [ ] Handles missing values gracefully
- [ ] Validates date formats
- [ ] Handles edge cases (single trade, single strategy)
- [ ] Correct EV calculation verified against manual calculation

### UI Testing
- [ ] All filters work correctly
- [ ] Charts render properly
- [ ] Page navigation preserves state
- [ ] Mobile-responsive layout

### Performance Testing
- [ ] Handles 5,000+ trades without lag
- [ ] Caching works correctly
- [ ] Charts render within 2 seconds

---

## Sample Code Structure

```
trading_analyzer/
├── app.py                  # Main Streamlit app with page routing
├── pages/
│   ├── 1_📈_Dashboard.py
│   ├── 2_🗓️_Quarterly_Analysis.py
│   ├── 3_📉_Performance_Trends.py
│   ├── 4_🔬_Regime_Analysis.py
│   └── 5_💼_Portfolio_Builder.py
├── utils/
│   ├── data_processing.py  # Data loading and transformation
│   ├── calculations.py     # EV, metrics, rolling calculations
│   ├── visualizations.py   # Chart generation functions
│   └── filters.py          # Filter logic and session state
├── config.py               # Constants, color schemes, settings
└── requirements.txt
```

---

## Key Success Metrics

The application successfully achieves its goals when users can:

1. **Identify Quarter Patterns**: Clearly see which strategies historically perform best in each quarter
2. **Track Improvement**: Determine if a strategy is improving or degrading over time
3. **Understand Regimes**: Know which strategies to deploy based on current VIX levels
4. **Build Portfolios**: Construct and evaluate multi-strategy portfolios with confidence
5. **Make Data-Driven Decisions**: Replace intuition with quantified Expected Value analysis

---

## Questions for User During Development

If unclear during implementation, ask the user:

1. Should EV calculations use raw P/L or P/L % (percentage)?
2. For portfolio combination, should EV be simple sum or consider trade overlap/conflicts?
3. What minimum number of trades should be required before showing statistics for a strategy/period combination?
4. Should the app support dark mode toggle?
5. Is there a preference for chart library (Plotly vs Altair)?
