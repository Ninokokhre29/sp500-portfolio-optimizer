import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
import warnings
import os
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SP500 Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .prediction-up {
        color: #10b981;
        font-weight: bold;
    }
    .prediction-down {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sp500_data' not in st.session_state:
    st.session_state.sp500_data = None
if 'bond_data' not in st.session_state:
    st.session_state.bond_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Helper functions
def load_static_data():
    """Load actual CSV data directly (e.g., DGS10 returns, SP500, actual returns, predicted returns)"""
    
    # File paths (use the paths you provided)
    sp500_file_path = r"C:\Users\Nino Kokhreidze\OneDrive\Desktop\project\top14_results.csv"
    bond_file_path = r"C:\Users\Nino Kokhreidze\OneDrive\Desktop\project\DGS10.csv"
    
    try:
        # Check if files exist
        if not os.path.exists(sp500_file_path):
            st.error(f"SP500 file not found at: {sp500_file_path}")
            return None, None, None
        
        if not os.path.exists(bond_file_path):
            st.error(f"Bond file not found at: {bond_file_path}")
            return None, None, None
            
        # Read data from local CSV files
        sp500_data = pd.read_csv(sp500_file_path)
        bond_data = pd.read_csv(bond_file_path)
        
        # Print column names to debug
        print("SP500 Columns:", sp500_data.columns.tolist())
        print("DGS10 Columns:", bond_data.columns.tolist())
        
        # Remove any leading/trailing spaces from column names
        sp500_data.columns = sp500_data.columns.str.strip()
        bond_data.columns = bond_data.columns.str.strip()

        # Convert the 'date' column to datetime for SP500 data
        if 'date' in sp500_data.columns:
            sp500_data['date'] = pd.to_datetime(sp500_data['date'])
        else:
            st.error("'date' column not found in SP500 data")
            return None, None, None
        
        # Check if required columns exist
        required_sp500_cols = ['y_true', 'y_pred']
        missing_cols = [col for col in required_sp500_cols if col not in sp500_data.columns]
        if missing_cols:
            st.error(f"Missing columns in SP500 data: {missing_cols}")
            return None, None, None
        
        # Add 'Direction' column based on y_true and y_pred in SP500 data
        sp500_data['Direction'] = np.where(sp500_data['y_pred'] > sp500_data['y_true'], 'up', 'down')

        # For predictions, adjust based on your actual data columns and structure
        predictions = sp500_data[['date', 'y_true', 'y_pred', 'Direction']].copy()

        # Convert 'observation_date' to datetime for DGS10 data
        if 'observation_date' in bond_data.columns:
            # Try different date formats
            try:
                bond_data['observation_date'] = pd.to_datetime(bond_data['observation_date'], format='%m/%d/%Y')
            except ValueError:
                try:
                    bond_data['observation_date'] = pd.to_datetime(bond_data['observation_date'], format='ISO8601')
                except ValueError:
                    # If both fail, let pandas infer the format
                    bond_data['observation_date'] = pd.to_datetime(bond_data['observation_date'], format='mixed')
        else:
            st.error("'observation_date' column not found in bond data")
            return None, None, None
        
        # Check if DGS10 column exists
        if 'DGS10' not in bond_data.columns:
            st.error("'DGS10' column not found in bond data")
            return None, None, None
        
        return sp500_data, bond_data, predictions
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def calculate_optimal_weights(expected_returns, cov_matrix, risk_free_rate=0.02):
    n = len(expected_returns)
    
    # For demonstration - replace with your actual optimization
    weights = np.random.dirichlet(np.ones(n), size=1)[0]
    
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    
    return weights, portfolio_return, portfolio_risk, sharpe_ratio

def create_line_chart(data, x_col, y_col, title, color='#3b82f6'):
    """Create a line chart using Plotly"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines',
        name=title,
        line=dict(color=color, width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='x unified',
        showlegend=False
    )
    
    return fig

def create_pie_chart(weights, labels):
    """Create a pie chart for portfolio allocation"""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=weights,
        hole=0.3,
        marker_colors=['#3b82f6', '#ef4444']
    )])
    
    fig.update_layout(
        title="Optimal Portfolio Allocation",
        showlegend=True
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">📈 SP500 Portfolio Optimizer</h1>', unsafe_allow_html=True)
    
    # Load static data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            sp500_data, bond_data, predictions = load_static_data()
            
            if sp500_data is not None and bond_data is not None and predictions is not None:
                st.session_state.sp500_data = sp500_data
                st.session_state.bond_data = bond_data
                st.session_state.predictions = predictions
                st.session_state.data_loaded = True
            else:
                st.error("Failed to load data. Please check your file paths and data format.")
                return
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.error("Data not loaded. Please check your file paths.")
        return
    
    sp500_data = st.session_state.sp500_data
    bond_data = st.session_state.bond_data
    predictions = st.session_state.predictions
    
    # Date selection
    st.subheader("📅 Select Analysis Date")
    
    # Get date range from data
    min_date = sp500_data['date'].min().date()
    max_date = sp500_data['date'].max().date()
    
    selected_date = st.date_input(
        "Choose Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Education", "📊 Market Analysis", "🎯 Optimization", "📈 Performance"])
    
    with tab1:
        st.header("📚 Financial Education")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏢 S&P 500 Index")
            st.markdown("""
            The S&P 500 is a stock market index tracking the performance of 500 large companies 
            listed on stock exchanges in the United States. It is one of the most commonly followed 
            equity indices and is considered a benchmark for the overall U.S. stock market performance.
            """)
        
        with col2:
            st.subheader("🏦 10-Year Treasury Bond")
            st.markdown("""
            The 10-Year Treasury Bond is a debt security issued by the U.S. government that matures 
            in 10 years. It's considered one of the safest investments and is often used as a benchmark 
            for other interest rates. The yield represents the return an investor would receive.
            """)
        
        st.divider()
        
        st.subheader("🎯 Markowitz Portfolio Theory")
        st.markdown("""
        Modern Portfolio Theory suggests that investors can construct an "efficient frontier" 
        of optimal portfolios offering the maximum expected return for each level of risk. 
        The theory emphasizes diversification to reduce risk while maintaining expected returns.
        """)
    
    with tab2:
        st.header("📊 Market Analysis")
        
        # Filter data by selected date
        filtered_sp500 = sp500_data[sp500_data['date'] <= pd.to_datetime(selected_date)]
        filtered_bond = bond_data[bond_data['observation_date'] <= pd.to_datetime(selected_date)]
        
        # Get current prediction
        current_prediction = predictions[predictions['date'] <= pd.to_datetime(selected_date)]
        if not current_prediction.empty:
            current_pred = current_prediction.iloc[-1]
        else:
            current_pred = predictions.iloc[0]  # fallback
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 S&P 500 Historical Price")
            if not filtered_sp500.empty:
                fig = create_line_chart(filtered_sp500, 'date', 'y_true', 'S&P 500 Price')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for selected date")
        
        with col2:
            st.subheader("🔮 Price Prediction")
            direction_color = "prediction-up" if current_pred['Direction'] == 'up' else "prediction-down"
            direction_icon = "📈" if current_pred['Direction'] == 'up' else "📉"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{direction_icon} Direction: <span class="{direction_color}">{current_pred['Direction'].upper()}</span></h3>
                <p><strong>Actual:</strong> ${current_pred['y_true']:.2f}</p>
                <p><strong>Predicted:</strong> ${current_pred['y_pred']:.2f}</p>
                <p><strong>Difference:</strong> ${abs(current_pred['y_pred'] - current_pred['y_true']):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        
        st.divider()
        
        st.subheader("🏦 10-Year Treasury Yield")
        if not filtered_bond.empty:
            # Remove NaN values for plotting
            filtered_bond_clean = filtered_bond.dropna(subset=['DGS10'])
            if not filtered_bond_clean.empty:
                fig_bond = create_line_chart(filtered_bond_clean, 'observation_date', 'DGS10', '10-Year Treasury Yield', '#ef4444')
                st.plotly_chart(fig_bond, use_container_width=True)
            else:
                st.warning("No valid bond data available for selected date range")
        else:
            st.warning("No bond data available for selected date")
    
    with tab3:
        st.header("🎯 Portfolio Optimization")
        
        # Calculate optimal weights (simplified)
        expected_returns = np.array([0.08, 0.03])  # SP500, Bonds
        cov_matrix = np.array([[0.04, 0.01], [0.01, 0.01]])
        
        weights, portfolio_return, portfolio_risk, sharpe_ratio = calculate_optimal_weights(expected_returns, cov_matrix)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Optimal Allocation")
            
            # Pie chart
            fig_pie = create_pie_chart(weights, ['S&P 500', '10-Year Treasury'])
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📈 Portfolio Metrics")
            st.metric("Expected Return", f"{portfolio_return:.2%}")
            st.metric("Portfolio Risk", f"{portfolio_risk:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            st.metric("Risk-Free Rate", "2.0%")
    
    with tab4:
        st.header("📈 Portfolio Performance")
        
        # Generate sample portfolio performance data
        dates = pd.date_range(start=sp500_data['date'].min(), end=sp500_data['date'].max(), freq='D')
        
        # Simulate portfolio performance
        np.random.seed(42)  # For reproducible results
        portfolio_returns = np.random.normal(0.0003, 0.01, len(dates))
        sp500_returns = np.random.normal(0.0004, 0.015, len(dates))
        bond_returns = np.random.normal(0.0001, 0.005, len(dates))
        
        portfolio_value = 10000 * np.cumprod(1 + portfolio_returns)
        sp500_value = 10000 * np.cumprod(1 + sp500_returns)
        bond_value = 10000 * np.cumprod(1 + bond_returns)
        
        performance_data = pd.DataFrame({
            'Date': dates,
            'Optimized Portfolio': portfolio_value,
            'S&P 500 Only': sp500_value,
            'Bonds Only': bond_value
        })
        
        # Performance chart
        st.subheader("📊 Cumulative Returns Comparison")
        fig_performance = go.Figure()
        
        for column, color in zip(['Optimized Portfolio', 'S&P 500 Only', 'Bonds Only'], 
                                ['#8b5cf6', '#3b82f6', '#ef4444']):
            fig_performance.add_trace(go.Scatter(
                x=performance_data['Date'],
                y=performance_data[column],
                mode='lines',
                name=column,
                line=dict(color=color, width=2)
            ))
        
        fig_performance.update_layout(
            title="Portfolio Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Performance metrics
        st.subheader("📊 Performance Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Optimized Portfolio", f"${portfolio_value[-1]:,.0f}", 
                     f"{(portfolio_value[-1]/10000-1)*100:.1f}%")
        
        with col2:
            st.metric("S&P 500 Only", f"${sp500_value[-1]:,.0f}", 
                     f"{(sp500_value[-1]/10000-1)*100:.1f}%")
        
        with col3:
            st.metric("Bonds Only", f"${bond_value[-1]:,.0f}", 
                     f"{(bond_value[-1]/10000-1)*100:.1f}%")

if __name__ == "__main__":
    main()