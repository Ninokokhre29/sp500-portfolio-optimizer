import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import warnings
import requests
from io import BytesIO
import os
warnings.filterwarnings('ignore')

st.set_page_config(page_title="S&P500 Portfolio Optimizer", layout="wide")
st.markdown(""" 
<h1 style="text-align: center; font-size: 3rem; font-weight: bold; color: #1f2937;"> S&P500 Portfolio Optimizer </h1>
""", unsafe_allow_html=True)
st.markdown(""" 
<style> 
.metric-card { background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; }
.prediction-up { color: #10b981; font-weight: bold; }
.prediction-down { color: #ef4444; font-weight: bold; }
div[data-testid="metric-container"] div[data-testid="metric-label"] { font-size: 1.5rem !important; font-weight: 600 !important; color: #34495e !important; }
.optimization-card { background-color: #f0f9ff; padding: 1.5rem; border-radius: 0.75rem; border: 2px solid #0ea5e9; margin: 1rem 0; }
.stock-allocation { background-color: #fefce8; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all required data files"""
    try:
        # Existing data
        portfolio_df = pd.read_csv("https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/portfolio_returns_cleaned.csv")
        monthly_df = pd.read_csv('https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/monthly_comparison.csv')
        annual_df = pd.read_csv("https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/annual_comparison.csv")
        hist_df = pd.read_excel("https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/portfolio_returns%20top14%20(regular).xlsx")
        arima_df = pd.read_excel("https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/ARIMA%20(1%2C0%2C1).xlsx")
        
        # Stock diversification data
        ticker_port_df = pd.read_excel("https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/Weights%20(ARIMA%2C%2044%20tickers).xlsx")
        hist_ticker_port_df = pd.read_excel("https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/Weights%20(regular%2C%2044%20tickers).xlsx")
        ret_arima_df = pd.read_excel("https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/Monthly%20returns%20(ARIMA).xlsx")
        ret_regular_df = pd.read_excel("https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/Monthly%20returns%20(standard%2C%2044%20tickers).xlsx")
        
        # Market data
        sp500_data = pd.read_csv('https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/top14_results.csv')
        bond_data = pd.read_csv('https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/DGS10.csv')
        
        # Metrics
        response = requests.get('https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/metrics.xlsx')
        response.raise_for_status()
        metrics_data = pd.read_excel(BytesIO(response.content))
        
        return {
            'portfolio_df': portfolio_df,
            'monthly_df': monthly_df,
            'annual_df': annual_df,
            'hist_df': hist_df,
            'arima_df': arima_df,
            'ticker_port_df': ticker_port_df,
            'hist_ticker_port_df': hist_ticker_port_df,
            'ret_arima_df': ret_arima_df,
            'ret_regular_df': ret_regular_df,
            'sp500_data': sp500_data,
            'bond_data': bond_data,
            'metrics_data': metrics_data
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data_dict):
    """Preprocess all loaded data"""
    if data_dict is None:
        return None
    
    # Clean column names
    for df_name, df in data_dict.items():
        if hasattr(df, 'columns'):
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    # Process dates and directions
    data_dict['sp500_data']['date'] = pd.to_datetime(data_dict['sp500_data']['date'])
    data_dict['sp500_data']['Direction'] = np.where(
        data_dict['sp500_data']['y_pred'] > data_dict['sp500_data']['y_true'].shift(1), 'up', 'down'
    )
    
    data_dict['bond_data']['observation_date'] = pd.to_datetime(data_dict['bond_data']['observation_date'])
    data_dict['bond_data']['Direction'] = np.where(
        data_dict['bond_data']['DGS10'] > data_dict['bond_data']['DGS10'].shift(1), 'up', 'down'
    )
    
    if 'pred' in data_dict['arima_df'].columns:
        data_dict['arima_df']['Direction'] = np.where(
            data_dict['arima_df']['pred'] > data_dict['arima_df']['pred'].shift(1), 'up', 'down'
        )
    
    # Create merged dataframe for index/bond optimization
    predicted_df = data_dict['portfolio_df'].copy()
    merged_df = pd.merge(
        predicted_df[["month", "SP500 weight", "Tbill weight", "portfolio_return"]], 
        data_dict['hist_df'][["month", "SP500 weight", "Tbill weight", "portfolio_return"]], 
        on="month", suffixes=("_pred", "_hist")
    )
    merged_df["year"] = np.where(merged_df["month"].between(5, 12), 2024, 2025)
    merged_df["month_label"] = pd.to_datetime(
        merged_df["year"].astype(str) + "-" + merged_df["month"].astype(str).str.zfill(2) + "-01"
    ).dt.strftime("%B %Y")
    merged_df = merged_df.sort_values(["year", "month"])
    
    data_dict['merged_df'] = merged_df
    return data_dict

def create_line_chart(data, x_col, y_col, title, color='#3b82f6', selected_date=None):
    """Create line chart with optional date selector"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_col], y=data[y_col], mode='lines', name=title,
        line=dict(color=color, width=2)
    ))
    if selected_date:
        fig.add_vline(x=selected_date, line_width=2, line_dash="dash", line_color="green")
    fig.update_layout(
        title=title, xaxis_title="Date", 
        yaxis_title="S&P 500 Index" if y_col == 'y_true' else y_col,
        hovermode='x unified', title_x=0.5
    )
    return fig

def display_stock_diversification_optimization(data_dict, amount):
    """Display stock diversification optimization results"""
    st.subheader("🎯 Individual Stock Diversification Strategy")
    
    # Strategy selection
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        st.markdown("""
        <div class="optimization-card">
            <h4>📊 ARIMA-Based Strategy</h4>
            <p>Uses time series forecasting to predict individual stock returns and optimize weights accordingly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with strategy_col2:
        st.markdown("""
        <div class="optimization-card">
            <h4>📈 Historical Mean Strategy</h4>
            <p>Based on historical performance patterns of individual S&P 500 stocks for stable allocation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display top stock allocations for both strategies
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ARIMA-Based Allocation")
        if not data_dict['ticker_port_df'].empty:
            # Get latest weights (assuming last row is most recent)
            latest_arima_weights = data_dict['ticker_port_df'].iloc[-1]
            
            # Convert to series and sort by weight
            weight_series = pd.Series(latest_arima_weights).drop(['month'], errors='ignore')
            weight_series = weight_series.sort_values(ascending=False)
            
            # Display top 10 allocations
            top_10_arima = weight_series.head(10)
            
            for ticker, weight in top_10_arima.items():
                if weight > 0:
                    allocation_amount = amount * weight
                    st.markdown(f"""
                    <div class="stock-allocation">
                        <strong>{ticker}</strong>: {weight:.2%} → ${allocation_amount:,.2f}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Calculate expected return for ARIMA strategy
            if not data_dict['ret_arima_df'].empty:
                latest_return = data_dict['ret_arima_df'].iloc[-1]['portfolio_return']
                expected_gain_arima = amount * (latest_return / 100)
                st.success(f"**Expected Return:** ${expected_gain_arima:.2f} ({latest_return:.2f}%)")
    
    with col2:
        st.subheader("Historical Mean Allocation")
        if not data_dict['hist_ticker_port_df'].empty:
            # Get latest weights
            latest_hist_weights = data_dict['hist_ticker_port_df'].iloc[-1]
            
            # Convert to series and sort by weight
            weight_series_hist = pd.Series(latest_hist_weights).drop(['month'], errors='ignore')
            weight_series_hist = weight_series_hist.sort_values(ascending=False)
            
            # Display top 10 allocations
            top_10_hist = weight_series_hist.head(10)
            
            for ticker, weight in top_10_hist.items():
                if weight > 0:
                    allocation_amount = amount * weight
                    st.markdown(f"""
                    <div class="stock-allocation">
                        <strong>{ticker}</strong>: {weight:.2%} → ${allocation_amount:,.2f}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Calculate expected return for historical strategy
            if not data_dict['ret_regular_df'].empty:
                latest_return_hist = data_dict['ret_regular_df'].iloc[-1]['portfolio_return']
                expected_gain_hist = amount * (latest_return_hist / 100)
                st.info(f"**Expected Return:** ${expected_gain_hist:.2f} ({latest_return_hist:.2f}%)")

def display_index_bond_optimization(data_dict, amount):
    """Display S&P 500 Index vs Treasury Bond optimization"""
    st.subheader("📈 S&P 500 Index vs 10-Year Treasury Strategy")
    
    # Month selection
    month_options = data_dict['merged_df']["month_label"].tolist()
    selected_label = st.selectbox("Select Month for Analysis", month_options)
    selected_row = data_dict['merged_df'][data_dict['merged_df']["month_label"] == selected_label].iloc[0]
    
    # Extract values
    sp500_weight = selected_row["SP500 weight_pred"] 
    tbill_weight = selected_row["Tbill weight_pred"] 
    port_return = selected_row["portfolio_return_pred"]
    hist_sp500_weight = selected_row["SP500 weight_hist"]
    hist_tbill_weight = selected_row["Tbill weight_hist"]
    hist_return = selected_row["portfolio_return_hist"] 
    
    # Display pie charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model-Based Allocation")
        pie1 = go.Figure(data=[go.Pie(
            labels=["S&P 500 Index", "10-Year Treasury"], 
            values=[sp500_weight, tbill_weight], 
            hole=0.4,
            marker_colors=["#4CAF50", "#FF9800"]
        )])
        pie1.update_layout(width=400, height=350, showlegend=True)
        st.plotly_chart(pie1, use_container_width=True)
    
    with col2:
        st.subheader("Historical Mean Allocation")
        pie2 = go.Figure(data=[go.Pie(
            labels=["S&P 500 Index", "10-Year Treasury"], 
            values=[hist_sp500_weight, hist_tbill_weight],
            hole=0.4, 
            marker_colors=["#2196F3", "#FFB300"]
        )])
        pie2.update_layout(width=400, height=350, showlegend=True)
        st.plotly_chart(pie2, use_container_width=True)
    
    # Calculate allocations
    sp500_amt = amount * sp500_weight 
    tbill_amt = amount * tbill_weight
    expected_gain_pred = amount * (port_return / 100)
    hist_sp500_amt = amount * hist_sp500_weight
    hist_tbill_amt = amount * hist_tbill_weight
    expected_gain_hist = amount * (hist_return / 100)
    
    # Display recommendations
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""**Model-Based Allocation:**
- S&P 500 Index: ${sp500_amt:,.2f} ({sp500_weight:.2%})
- 10-Year Treasury: ${tbill_amt:,.2f} ({tbill_weight:.2%})

**Expected Return:** ${expected_gain_pred:.2f} ({port_return:.2f}%)""")
    
    with col2:
        st.info(f"""**Historical Mean Allocation:**
- S&P 500 Index: ${hist_sp500_amt:,.0f} ({hist_sp500_weight:.1%})
- 10-Year Treasury: ${hist_tbill_amt:,.0f} ({hist_tbill_weight:.1%})

**Expected Return:** ${expected_gain_hist:.2f} ({hist_return:.2f}%)""")

def main():
    # Load and preprocess data
    data_dict = load_data()
    if data_dict is None:
        st.error("Failed to load data. Please check your internet connection and try again.")
        return
    
    data_dict = preprocess_data(data_dict)
    if data_dict is None:
        st.error("Failed to preprocess data.")
        return
    
    min_date = data_dict['sp500_data']['date'].min().date()
    max_date = data_dict['sp500_data']['date'].max().date()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Market Analysis", "Portfolio Optimization", "Performance"])
    
    with tab1:
        st.header("Financial Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("S&P 500 Index")
            st.markdown("""
            The S&P 500 is a stock market index that tracks the stock performance of 500 large companies listed on stock exchanges in the United States. It is one of the 
            most commonly followed equity indices and is considered a benchmark for the overall U.S. stock market.
            
            **Key Features:**
            - Market-cap weighted index
            - Covers approximately 80% of U.S. equity market capitalization
            - Rebalanced quarterly
            - Widely used for benchmarking and passive investing
            """)
        
        with col2:
            st.subheader("10-Year Treasury Bond")
            st.markdown("""
            The 10-year Treasury note is a debt obligation issued by the United States government with a maturity of 10 years. It's considered one of the safest investments and serves 
            as a benchmark for other interest rates.
            
            **Key Features:**
            - Backed by the full faith and credit of the U.S. government
            - Fixed interest payments every six months
            - Inverse relationship with interest rates
            - Used as a risk-free rate in financial models
            """)
        
        st.divider()
        
        # Markowitz Theory section (keeping existing content)
        st.markdown("""
        <div style="background-color: white; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 1.5rem; margin: 1rem 0;">
            <h2 style="font-size: 1.5rem; font-weight: bold; color: #111827; margin-bottom: 1rem;">Modern Portfolio Theory</h2>
            <div style="color: #374151; margin-bottom: 1rem;">
                <p>Modern Portfolio Theory, introduced by Harry Markowitz, provides a mathematical framework for assembling a portfolio of assets such that the expected return is maximized for a given level of risk, or equivalently, the risk is minimized for a given level of expected return.</p>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="background-color: #faf5ff; padding: 1rem; border-radius: 0.5rem;">
                    <h3 style="font-weight: 600; color: #581c87; margin-bottom: 0.5rem;">Diversification</h3>
                    <p style="color: #6b21a8; font-size: 0.875rem;">Reduces portfolio risk by combining assets with different risk-return profiles</p>
                </div>
                <div style="background-color: #faf5ff; padding: 1rem; border-radius: 0.5rem;">
                    <h3 style="font-weight: 600; color: #581c87; margin-bottom: 0.5rem;">Efficient Frontier</h3>
                    <p style="color: #6b21a8; font-size: 0.875rem;">The set of optimal portfolios offering the highest expected return for each level of risk</p>
                </div>
                <div style="background-color: #faf5ff; padding: 1rem; border-radius: 0.5rem;">
                    <h3 style="font-weight: 600; color: #581c87; margin-bottom: 0.5rem;">Risk-Return Tradeoff</h3>
                    <p style="color: #6b21a8; font-size: 0.875rem;">Balances expected returns against the volatility of those returns</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Model description (keeping existing content but making it more concise)
        st.subheader("About the Optimization Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Individual Stock Strategy**
            
            Optimizes allocation across individual S&P 500 stocks using:
            - ARIMA time series forecasting for individual stock returns
            - Historical mean reversion patterns
            - Modern Portfolio Theory optimization
            - Risk-return optimization across 44+ stocks
            """)
        
        with col2:
            st.markdown("""
            **📈 Index/Bond Strategy**
            
            Balances between S&P 500 index and Treasury bonds using:
            - LightGBM machine learning predictions
            - Walk-forward validation approach
            - 60+ technical and fundamental features
            - SHAP feature importance analysis
            """)

        # Data summary
        if st.button("Show Data Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Dataset Overview")
                st.metric("Date Range", "2021-01-04 - 2025-04-30")
                st.metric("Total Predictions", f"{len(data_dict['sp500_data'])}")
                st.metric("Prediction Period", f"{min_date} - {max_date}")
                
            with col2:
                st.subheader("Model Performance")
                if len(data_dict['metrics_data']) > 2:
                    metrics_row = data_dict['metrics_data'].iloc[2]
                    st.metric("Features Used", f"{int(metrics_row['n_feat_used'])}")
                    st.metric("RMSE", f"{metrics_row['rmse']:.4f}")
                    st.metric("R²", f"{metrics_row['r2'] * 100:.2f}%")
                    st.metric("Hit Rate", f"{metrics_row['hit_rate']:.2%}")

    with tab2:
        st.header("Market Analysis")
        
        # Date selection
        selected_date = st.date_input(
            "Select Date for Analysis", 
            value=min_date, 
            min_value=min_date, 
            max_value=max_date
        )
        
        selected_data = data_dict['sp500_data'][data_dict['sp500_data']['date'].dt.date <= selected_date]
        bond_data_filtered = data_dict['bond_data'][data_dict['bond_data']['observation_date'].dt.date <= selected_date]
        
        if not selected_data.empty:
            # Current market conditions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("S&P 500 Performance")
                row = selected_data.iloc[-1]
                direction_class = "prediction-up" if row['Direction'] == 'up' else "prediction-down"
                st.markdown(f"""
                <div class="metric-card">
                    <p style="font-size: 1.5rem; font-weight: bold;">{row['y_pred']*100:.2f}%</p>
                    <p class="{direction_class}">Direction: {row['Direction'].upper()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("10-Year Treasury Rate")
                if not bond_data_filtered.empty:
                    latest_bond = bond_data_filtered.iloc[-1]
                    bond_color_class = "prediction-up" if latest_bond['Direction'] == "up" else "prediction-down"
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="font-size: 1.5rem; font-weight: bold;">{latest_bond['DGS10']:.2f}%</p>
                        <p class="{bond_color_class}">Direction: {latest_bond['Direction'].upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Historical performance charts
            st.subheader("Historical Performance")
            
            if len(selected_data) > 1:
                fig_sp500 = create_line_chart(
                    selected_data, 'date', 'y_true', 
                    'S&P 500 Actual Performance', '#3b82f6'
                )
                st.plotly_chart(fig_sp500, use_container_width=True)
            
            if len(bond_data_filtered) > 1:
                fig_bond = create_line_chart(
                    bond_data_filtered, 'observation_date', 'DGS10', 
                    '10-Year Treasury Rate', '#ef4444'
                )
                st.plotly_chart(fig_bond, use_container_width=True)

    with tab3:
        st.header("Portfolio Optimization Strategies")
        
        # Investment amount input
        amount = st.number_input(
            "💰 Enter Investment Amount ($)", 
            min_value=1000, 
            value=10000, 
            step=100,
            help="Enter the total amount you want to invest"
        )
        
        # Strategy selection
        optimization_strategy = st.radio(
            "Choose Optimization Strategy:",
            ["Individual Stock Diversification", "S&P 500 Index vs Treasury Bonds", "Compare Both Strategies"],
            help="Select which optimization approach you want to analyze"
        )
        
        if optimization_strategy == "Individual Stock Diversification":
            display_stock_diversification_optimization(data_dict, amount)
            
        elif optimization_strategy == "S&P 500 Index vs Treasury Bonds":
            display_index_bond_optimization(data_dict, amount)
            
        else:  # Compare Both Strategies
            st.subheader("📊 Strategy Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Individual Stock Strategy")
                display_stock_diversification_optimization(data_dict, amount)
            
            with col2:
                st.markdown("### Index/Bond Strategy")
                display_index_bond_optimization(data_dict, amount)

    with tab4:
        st.header("Performance Analysis")
        
        # Annual performance comparison
        if 'methodology' in data_dict['annual_df'].columns and 'portfolio return' in data_dict['annual_df'].columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Annual Return Comparison")
                if data_dict['annual_df']["portfolio return"].dtype == 'object':
                    data_dict['annual_df']["portfolio return"] = data_dict['annual_df']["portfolio return"].str.rstrip('%').astype(float)
                
                bar_fig = px.bar(
                    data_dict['annual_df'], 
                    x="methodology", 
                    y="portfolio return", 
                    color="methodology",
                    color_discrete_sequence=["#4CAF50", "#FF9800"],
                    labels={"portfolio return": "Return (%)"},
                    title="Annual Return: Traditional vs ML-Enhanced"
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            with col2:
                st.subheader("Risk-Adjusted Returns")
                if len(data_dict['annual_df']) >= 2:
                    st.metric("Traditional Sharpe Ratio", f"{data_dict['annual_df'].iloc[0]['Sharpe Ratio']:.3f}")
                    st.metric("ML-Enhanced Sharpe Ratio", f"{data_dict['annual_df'].iloc[1]['Sharpe Ratio']:.3f}")
                    
                    # Calculate improvement
                    improvement = ((data_dict['annual_df'].iloc[1]['Sharpe Ratio'] / data_dict['annual_df'].iloc[0]['Sharpe Ratio']) - 1) * 100
                    st.metric("Improvement", f"{improvement:.1f}%")

        # Monthly performance tracking
        if all(col in data_dict['monthly_df'].columns for col in ['Date', 'Historical Mean', 'Predicted']):
            st.subheader("Monthly Performance Tracking")
            
            data_dict['monthly_df']["Date"] = pd.to_datetime(data_dict['monthly_df']["Date"])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data_dict['monthly_df']["Date"],
                y=data_dict['monthly_df']["Historical Mean"],
                name="Traditional Strategy",
                line=dict(color='#2196F3', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=data_dict['monthly_df']["Date"],
                y=data_dict['monthly_df']["Predicted"],
                name="ML-Enhanced Strategy",
                line=dict(color='#FF9800', width=3)
            ))
            
            fig.update_layout(
                title="Monthly Returns: Traditional vs ML-Enhanced Strategy",
                yaxis_title="Monthly Return (%)",
                xaxis_title="Date",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            if len(data_dict['monthly_df']) > 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    traditional_mean = data_dict['monthly_df']["Historical Mean"].mean()
                    st.metric("Traditional Avg Return", f"{traditional_mean:.2f}%")
                
                with col2:
                    ml_mean = data_dict['monthly_df']["Predicted"].mean()
                    st.metric("ML-Enhanced Avg Return", f"{ml_mean:.2f}%")
                
                with col3:
                    outperformance = ml_mean - traditional_mean
                    st.metric("Monthly Outperformance", f"{outperformance:.2f}%")

if __name__ == "__main__":
    main()
