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
    
    sp500_file_url = 'https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/top14_results.csv'
    bond_file_url = 'https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/DGS10.csv'

    try:
        # Load data
        sp500_data = pd.read_csv(sp500_file_url)
        bond_data = pd.read_csv(bond_file_url)
        
        # Debug: Print column names
        print("SP500 Columns:", sp500_data.columns.tolist())
        print("DGS10 Columns:", bond_data.columns.tolist())
        
        # Clean column names (remove any invisible characters)
        sp500_data.columns = sp500_data.columns.str.strip().str.replace('\ufeff', '')
        bond_data.columns = bond_data.columns.str.strip().str.replace('\ufeff', '')

        
        # Convert the 'date' column to datetime for SP500 data
        if 'date' in sp500_data.columns:
            sp500_data['date'] = pd.to_datetime(sp500_data['date'])
        else:
            st.error(f"'date' column not found in SP500 data. Available columns: {sp500_data.columns.tolist()}")
            return None, None, None
        
        # Check if required columns exist
        required_sp500_cols = ['y_true', 'y_pred']
        missing_cols = [col for col in required_sp500_cols if col not in sp500_data.columns]
        if missing_cols:
            st.error(f"Missing columns in SP500 data: {missing_cols}")
            st.write("Available columns:", sp500_data.columns.tolist())
            return None, None, None
        
        # Add 'Direction' column based on y_true and y_pred in SP500 data
        sp500_data['Direction'] = np.where(sp500_data['y_pred'] > sp500_data['y_true'], 'up', 'down')

        # For predictions, adjust based on your actual data columns and structure
        predictions = sp500_data[['date', 'y_true', 'y_pred', 'Direction']].copy()

        # Convert 'observation_date' to datetime for DGS10 data
        if 'observation_date' in bond_data.columns:
            # Try different date formats
            try:
                bond_data['observation_date'] = pd.to_datetime(bond_data['observation_date'])
            except Exception as e:
                st.error(f"Error converting bond observation_date: {e}")
                return None, None, None
        else:
            st.error(f"'observation_date' column not found in bond data. Available columns: {bond_data.columns.tolist()}")
            return None, None, None
        
        # Check if DGS10 column exists
        if 'DGS10' not in bond_data.columns:
            st.error(f"'DGS10' column not found in bond data. Available columns: {bond_data.columns.tolist()}")
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
            The S&P 500 is a stock market index that tracks the stock performance of 500 large companies 
            listed on stock exchanges in the United States. It is one of the most commonly followed equity 
            indices and is considered a benchmark for the overall U.S. stock market.
            
            **Key Features:**
            - Market-cap weighted index
            - Covers approximately 80% of U.S. equity market capitalization
            - Rebalanced quarterly
            - Widely used for benchmarking and passive investing
            """)
        
        with col2:
            st.subheader("🏦 10-Year Treasury Bond")
            st.markdown("""
            The 10-year Treasury note is a debt obligation issued by the United States government 
            with a maturity of 10 years. It's considered one of the safest investments and serves 
            as a benchmark for other interest rates.
            
            **Key Features:**
            - Backed by the full faith and credit of the U.S. government
            - Fixed interest payments every six months
            - Inverse relationship with interest rates
            - Used as a risk-free rate in financial models
            """)
        
        st.divider()
        
        st.subheader("🎯 Markowitz Portfolio Theory")
        st.markdown("""
        Modern Portfolio Theory, introduced by Harry Markowitz, provides a mathematical framework 
        for assembling a portfolio of assets such that the expected return is maximized for a given 
        level of risk, or equivalently, the risk is minimized for a given level of expected return.
        """)
        
        if st.button("Show Data Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("SP500 Records", len(sp500_data))
                st.metric("Date Range", f"{min_date} to {max_date}")
            with col2:
                st.metric("Bond Records", len(bond_data))
                accuracy = (predictions['y_true'] == predictions['y_pred']).mean()
                st.metric("Prediction Accuracy", f"{accuracy:.2%}")
    
    with tab2:
        st.header("📊 Market Analysis")
        
        # Filter data for selected date
        selected_data = sp500_data[sp500_data['date'].dt.date == selected_date]
        
        if not selected_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SP500 Performance")
                row = selected_data.iloc[0]
                
                direction_class = "prediction-up" if row['Direction'] == 'up' else "prediction-down"
                direction_symbol = "📈" if row['Direction'] == 'up' else "📉"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Market Direction {direction_symbol}</h4>
                    <p style="font-size: 1.5rem; font-weight: bold;">{row['y_pred']*100:.2f}%</p>
                    <p class="{direction_class}">Predicted: {row['Direction'].upper()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Bond Rate")
                bond_data_filtered = bond_data[bond_data['observation_date'].dt.date <= selected_date]
                if not bond_data_filtered.empty:
                    latest_bond = bond_data_filtered.iloc[-1]
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>10-Year Treasury Rate</h4>
                        <p style="font-size: 1.5rem; font-weight: bold;">{latest_bond['DGS10']:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Charts
        st.subheader("Historical Performance")
        
        # SP500 chart
        if len(sp500_data) > 1:
            fig_sp500 = create_line_chart(sp500_data, 'date', 'y_true', 'SP500 Actual Performance')
            st.plotly_chart(fig_sp500, use_container_width=True)
        
        # Bond chart
        if len(bond_data) > 1:
            fig_bond = create_line_chart(bond_data, 'observation_date', 'DGS10', '10-Year Treasury Rate', '#ef4444')
            st.plotly_chart(fig_bond, use_container_width=True)
    
    with tab3:
        st.header("🎯 Portfolio Optimization")
        
        st.subheader("Asset Allocation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_tolerance = st.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.1)
            expected_return_sp500 = st.number_input("Expected SP500 Return", value=0.10, step=0.01)
        
        with col2:
            risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, step=0.01)
            expected_return_bonds = st.number_input("Expected Bond Return", value=0.03, step=0.01)
        
        if st.button("Optimize Portfolio"):
            # Simple optimization example
            returns = np.array([expected_return_sp500, expected_return_bonds])
            cov_matrix = np.array([[0.04, 0.01], [0.01, 0.01]])  # Example covariance matrix
            
            weights, portfolio_return, portfolio_risk, sharpe_ratio = calculate_optimal_weights(returns, cov_matrix, risk_free_rate)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimal Allocation")
                fig_pie = create_pie_chart(weights, ['SP500', 'Bonds'])
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Portfolio Metrics")
                st.metric("Expected Return", f"{portfolio_return:.2%}")
                st.metric("Risk (Std Dev)", f"{portfolio_risk:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with tab4:
        st.header("📈 Performance Tracking")
        
        st.subheader("Prediction Accuracy")
        
        if len(predictions) > 0:
            # Calculate accuracy metrics
            total_predictions = len(predictions)
            correct_predictions = sum(predictions['y_true'] == predictions['y_pred'])
            accuracy = correct_predictions / total_predictions
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", total_predictions)
            with col2:
                st.metric("Correct Predictions", correct_predictions)
            with col3:
                st.metric("Accuracy", f"{accuracy:.2%}")
            
            # Performance chart
            fig_performance = go.Figure()
            fig_performance.add_trace(go.Scatter(
                x=predictions['date'],
                y=predictions['y_true'],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            fig_performance.add_trace(go.Scatter(
                x=predictions['date'],
                y=predictions['y_pred'],
                mode='lines',
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            
            fig_performance.update_layout(
                title="Actual vs Predicted Performance",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)

if __name__ == "__main__":
    main()
