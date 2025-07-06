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
    
    sp500_file_url = 'https://github.com/Ninokokhre29/sp500-portfolio-optimizer/blob/master/top14_results.csv'
    bond_file_url = 'https://github.com/Ninokokhre29/sp500-portfolio-optimizer/blob/master/DGS10.csv'

    try:
        sp500_data = pd.read_csv(sp500_file_url)
        bond_data = pd.read_csv(bond_file_url)
        
        print("SP500 Columns:", sp500_data.columns.tolist())
        print("DGS10 Columns:", bond_data.columns.tolist())
        
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


if __name__ == "__main__":
    main()
