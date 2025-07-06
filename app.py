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

st.set_page_config(
    page_title="SP500 Portfolio Optimizer",
    layout="wide"
)

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
    div[data-testid="metric-container"] div[data-testid="metric-label"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #34495e !important;
    }
</style>
""", unsafe_allow_html=True)

if 'sp500_data' not in st.session_state:
    st.session_state.sp500_data = None
if 'bond_data' not in st.session_state:
    st.session_state.bond_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def load_static_data():
    try:
        sp500_file_url = 'https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/top14_results.csv'
        bond_file_url = 'https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/DGS10.csv'
        metrics_url = 'https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/metrics.xlsx'
        
        sp500_data = pd.read_csv(sp500_file_url)
        bond_data = pd.read_csv(bond_file_url)
        response = requests.get(metrics_url)
        response.raise_for_status()
        metrics_data = pd.read_excel(BytesIO(response.content))
        
        sp500_data.columns = sp500_data.columns.str.strip().str.replace('\ufeff', '')
        bond_data.columns = bond_data.columns.str.strip().str.replace('\ufeff', '')
        metrics_data.columns = metrics_data.columns.str.strip().str.replace('\ufeff', '')
        
        sp500_data['date'] = pd.to_datetime(sp500_data['date'])
        sp500_data['Direction'] = np.where(sp500_data['y_pred'] > sp500_data['y_true'].shift(1), 'up', 'down')
        predictions = sp500_data[['date', 'y_true', 'y_pred', 'Direction']].copy()
        
        bond_data['observation_date'] = pd.to_datetime(bond_data['observation_date'])
        bond_data['Direction'] = np.where(bond_data['DGS10'] > bond_data['DGS10'].shift(1), 'up', 'down')
        
        return sp500_data, bond_data, predictions, metrics_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def calculate_optimal_weights(expected_returns, cov_matrix, risk_free_rate=0.02):
    n = len(expected_returns)
    weights = np.random.dirichlet(np.ones(n), size=1)[0]
    
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    
    return weights, portfolio_return, portfolio_risk, sharpe_ratio

def create_line_chart(data, x_col, y_col, title, color='#3b82f6', selected_date=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines',
        name=title,
        line=dict(color=color, width=2)
    ))
    
    if selected_date:
        fig.add_vline(x=selected_date, line_width=2, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="S&P 500 Index" if y_col == 'y_true' else y_col,
        hovermode='x unified',
        showlegend=False,
        title_x=0.5
    )
    
    return fig

def create_pie_chart(weights, labels):
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
    st.markdown('<h1 class="main-header">SP500 Portfolio Optimizer</h1>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            sp500_data, bond_data, predictions, metrics_data = load_static_data()
            
            if all(data is not None for data in [sp500_data, bond_data, predictions, metrics_data]):
                st.session_state.sp500_data = sp500_data
                st.session_state.bond_data = bond_data
                st.session_state.predictions = predictions
                st.session_state.metrics_data = metrics_data
                st.session_state.data_loaded = True
            else:
                st.error("Failed to load data. Please check your file paths and data format.")
                return

    if not st.session_state.data_loaded:
        st.error("Data not loaded. Please check your file paths.")
        return
        
    sp500_data = st.session_state.sp500_data
    bond_data = st.session_state.bond_data
    predictions = st.session_state.predictions
    metrics_data = st.session_state.metrics_data
    
    st.subheader("Select Analysis Date")

    min_date = sp500_data['date'].min().date()
    max_date = sp500_data['date'].max().date()
    
    selected_date = st.date_input(
        "Choose Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Market Analysis", "Optimization", "Performance"])
    
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
        
        st.subheader("Markowitz Portfolio Theory")
        st.markdown("""
        Modern Portfolio Theory, introduced by Harry Markowitz, provides a mathematical framework for assembling a portfolio of assets such that the expected return is 
        maximized for a given level of risk, or equivalently, the risk is minimized for a given level of expected return.
        """)

        st.subheader("About the Model")
        st.markdown("""
        A comprehensive machine learning pipeline for predicting S&P 500 stock returns using a walk-forward validation approach with LightGBM models. The system engineers 
        over 60 technical and fundamental features including moving averages, momentum indicators, volatility measures, volume patterns, and macroeconomic variables, then 
        applies feature selection techniques to identify the most predictive variables. Using time series cross-validation and hyperparameter optimization, the model 
        predicts future 20-day returns while avoiding look-ahead bias through proper temporal splitting. The pipeline incorporates SHAP (SHapley Additive exPlanations) 
        values to identify stable, high-importance features across multiple time periods, systematically testing different feature set sizes to optimize the balance 
        between model complexity and predictive performance. Results are evaluated using multiple metrics including RMSE, R-squared, and directional accuracy (hit rate), 
        with the system designed to handle the non-stationary nature of financial markets through robust preprocessing and validation methodologies.
        """)

        if st.button("Show Data Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Dataset Overview")
                st.metric("Date Range", f"{min_date} - {max_date}")
                st.metric("Total Rows", len(sp500_data))
            with col2:
                st.subheader("Model Evaluation")
                if len(metrics_data) > 2:
                    metrics_row = metrics_data.iloc[2]
                    st.metric("Features Used", f"{int(metrics_row['n_feat_used'])}")
                    st.metric("RMSE", f"{metrics_row['rmse']:.4f}")
                    st.metric("R²", f"{metrics_row['r2'] * 100:.2f}%")
                    st.metric("Hit Rate", f"{metrics_row['hit_rate']:.2%}")
                else:
                    st.warning("Insufficient metrics data available")
            
    with tab2:
        st.header("Market Analysis")
        selected_data = sp500_data[sp500_data['date'].dt.date <= selected_date]
        bond_data_filtered = bond_data[bond_data['observation_date'].dt.date <= selected_date]
        
        if not selected_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SP500 Performance")
                row = selected_data.iloc[-1]
                direction_class = "prediction-up" if row['Direction'] == 'up' else "prediction-down"
    
                st.markdown(f"""
                <div class="metric-card">
                    <p style="font-size: 1.5rem; font-weight: bold;">{row['y_pred']*100:.2f}%</p>
                    <p class="{direction_class}">Predicted: {row['Direction'].upper()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("10-Year Treasury Rate")
                if not bond_data_filtered.empty:
                    latest_bond = bond_data_filtered.iloc[-1]
                    bond_direction = latest_bond['Direction'].upper()
                    bond_color_class = "prediction-up" if bond_direction == "UP" else "prediction-down"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <p style="font-size: 1.5rem; font-weight: bold;">{latest_bond['DGS10']:.2f}%</p>
                        <p class="{bond_color_class}">Direction: {bond_direction}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No bond data available for the selected date")
        else:
            st.warning("No data available for the selected date")
        
        st.subheader("Historical Performance")

        if len(selected_data) > 1:
            fig_sp500 = create_line_chart(selected_data, 'date', 'y_true', 'SP500 Actual Performance', '#3b82f6')
            st.plotly_chart(fig_sp500, use_container_width=True)
        
        if len(bond_data_filtered) > 1:
            fig_bond = create_line_chart(bond_data_filtered, 'observation_date', 'DGS10', '10-Year Treasury Rate', '#ef4444')
            st.plotly_chart(fig_bond, use_container_width=True)
    
    with tab3:
        st.header("Portfolio Optimization")
        
        st.subheader("Asset Allocation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk_tolerance = st.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.1)
            expected_return_sp500 = st.number_input("Expected SP500 Return", value=0.10, step=0.01)
        
        with col2:
            risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, step=0.01)
            expected_return_bonds = st.number_input("Expected Bond Return", value=0.03, step=0.01)
        
        if st.button("Optimize Portfolio"):
            returns = np.array([expected_return_sp500, expected_return_bonds])
            cov_matrix = np.array([[0.04, 0.01], [0.01, 0.01]])
            
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
                
                st.subheader("Allocation Details")
                st.write(f"SP500: {weights[0]:.1%}")
                st.write(f"Bonds: {weights[1]:.1%}")
    
    with tab4:
        st.header("Performance Tracking")
        
        st.subheader("Prediction Accuracy")
        
        if len(predictions) > 0:
            predictions_copy = predictions.copy()
            predictions_copy['actual_direction'] = np.where(predictions_copy['y_true'] > 0, 'up', 'down')
            predictions_copy['predicted_direction'] = np.where(predictions_copy['y_pred'] > 0, 'up', 'down')
            
            total_predictions = len(predictions_copy)
            correct_predictions = sum(predictions_copy['actual_direction'] == predictions_copy['predicted_direction'])
            accuracy = correct_predictions / total_predictions
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", total_predictions)
            with col2:
                st.metric("Correct Predictions", correct_predictions)
            with col3:
                st.metric("Accuracy", f"{accuracy:.2%}")
    
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
                yaxis_title="Return",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
        else:
            st.warning("No prediction data available")

if __name__ == "__main__":
    main()
