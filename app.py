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
    layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem; }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6; }
    .prediction-up {
        color: #10b981;
        font-weight: bold; }
    .prediction-down {
        color: #ef4444;
        font-weight: bold; }
    div[data-testid="metric-container"] div[data-testid="metric-label"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #34495e !important; }
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
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = None
if 'monthly_df' not in st.session_state:
    st.session_state.monthly_df = None
if 'annual_df' not in st.session_state:
    st.session_state.annual_df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def load_static_data():
    try:
        portfolio_url = "https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/portfolio_returns_cleaned.csv"
        monthly_url = 'https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/monthly_comparison.csv'
        annual_url = "https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/annual_comparison.csv"
        hist_url = "https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/portfolio_returns%20top14%20(regular).xlsx"
        
        portfolio_df = pd.read_csv(portfolio_url)
        monthly_df = pd.read_csv(monthly_url)
        annual_df = pd.read_csv(annual_url)
        hist_df = pd.read_excel(hist_url)
        predicted_df = portfolio_df.copy() 

        merged_df = pd.merge(
            predicted_df[["month", "SP500 weight", "Tbill weight", "portfolio_return"]],
            hist_df[["month", "SP500 weight", "Tbill weight", "portfolio_return"]], on="month", suffixes=("_pred", "_hist"))
        
        merged_df["year"] = np.where(merged_df["month"].between(5, 12), 2024, 2025)
        merged_df["month_label"] = pd.to_datetime(merged_df["year"].astype(str) + "-" + merged_df["month"].astype(str).str.zfill(2) + "-01").dt.strftime("%B %Y")
        merged_df = merged_df.sort_values("month_label")
        
        sp500_data = pd.read_csv('https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/top14_results.csv')
        bond_data = pd.read_csv('https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/DGS10.csv')
        
        response = requests.get('https://raw.githubusercontent.com/Ninokokhre29/sp500-portfolio-optimizer/master/metrics.xlsx')
        response.raise_for_status()
        metrics_data = pd.read_excel(BytesIO(response.content))
        
        for df in [sp500_data, bond_data, metrics_data, portfolio_df, monthly_df, annual_df]:
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        sp500_data['date'] = pd.to_datetime(sp500_data['date'])
        sp500_data['Direction'] = np.where(sp500_data['y_pred'] > sp500_data['y_true'].shift(1), 'up', 'down')
        predictions = sp500_data[['date', 'y_true', 'y_pred', 'Direction']].copy()
    
        bond_data['observation_date'] = pd.to_datetime(bond_data['observation_date'])
        bond_data['Direction'] = np.where(bond_data['DGS10'] > bond_data['DGS10'].shift(1), 'up', 'down')
        
        return sp500_data, bond_data, predictions, metrics_data, portfolio_df, monthly_df, annual_df, merged_df

    except Exception as e:
        st.error(f"Error {str(e)}")
        return None, None, None, None, None, None, None, None


def create_line_chart(data, x_col, y_col, title, color='#3b82f6', selected_date=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines',
        name=title,
        line=dict(color=color, width=2)  ))
    
    if selected_date:
        fig.add_vline(x=selected_date, line_width=2, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="S&P 500 Index" if y_col == 'y_true' else y_col,
        hovermode='x unified',
        title_x=0.5 )
    
    return fig

def create_pie_chart(weights, labels):
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=weights,
        hole=0.3,
        marker_colors=['#3b82f6', '#ef4444']  )])
    
    fig.update_layout(
        title="Optimal Portfolio Allocation",
        showlegend=True )
    return fig

def main():
    st.markdown('<h1 class="main-header">SP500 Portfolio Optimizer</h1>', unsafe_allow_html=True)
    if not st.session_state.data_loaded:
        results = load_static_data()
            
        if all(data is not None for data in results):
            (st.session_state.sp500_data, st.session_state.bond_data, 
            st.session_state.predictions, st.session_state.metrics_data,
            st.session_state.portfolio_df, st.session_state.monthly_df,
            st.session_state.annual_df, st.session_state.merged_df) = results
            st.session_state.data_loaded = True
        
    sp500_data = st.session_state.sp500_data
    bond_data = st.session_state.bond_data
    predictions = st.session_state.predictions
    metrics_data = st.session_state.metrics_data
    portfolio_df = st.session_state.portfolio_df
    monthly_df = st.session_state.monthly_df
    annual_df = st.session_state.annual_df
    merged_df = st.session_state.merged_df

    min_date = sp500_data['date'].min().date() 
    max_date = sp500_data['date'].max().date()
    
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
            - Widely used for benchmarking and passive investing   """)
        
        with col2:
            st.subheader("10-Year Treasury Bond")
            st.markdown("""
            The 10-year Treasury note is a debt obligation issued by the United States government with a maturity of 10 years. It's considered one of the safest investments and serves 
            as a benchmark for other interest rates.
            
            **Key Features:**
            - Backed by the full faith and credit of the U.S. government
            - Fixed interest payments every six months
            - Inverse relationship with interest rates
            - Used as a risk-free rate in financial models  """)
        
        st.divider()
        
        st.markdown("""
        <div style="background-color: white; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 1.5rem; margin: 1rem 0;">
            <h2 style="font-size: 1.5rem; font-weight: bold; color: #111827; margin-bottom: 1rem;">Markowitz Portfolio Theory</h2>
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

        st.subheader("About the Model")
        st.markdown("""
        A comprehensive machine learning pipeline for predicting S&P 500 stock returns using a walk-forward validation approach with LightGBM models. The system engineers 
        over 60 technical and fundamental features including moving averages, momentum indicators, volatility measures, volume patterns, and macroeconomic variables, then 
        applies feature selection techniques to identify the most predictive variables. Using time series cross-validation and hyperparameter optimization, the model 
        predicts future 20-day returns while avoiding look-ahead bias through proper temporal splitting. The pipeline incorporates SHAP (SHapley Additive exPlanations) 
        values to identify stable, high-importance features across multiple time periods, systematically testing different feature set sizes to optimize the balance 
        between model complexity and predictive performance. Results are evaluated using multiple metrics including RMSE, R-squared, and directional accuracy (hit rate), 
        with the system designed to handle the non-stationary nature of financial markets through robust preprocessing and validation methodologies. """)

        if st.button("Show Data Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Dataset Overview")
                st.metric("Date Range", "2021-01-04 - 2025-04-30")
                st.metric("Total Rows", "1086")
                st.metric("Predictions Made On", f"{min_date} - {max_date}")
            with col2:
                st.subheader("Model Evaluation")
                if len(metrics_data) > 2:
                    metrics_row = metrics_data.iloc[2]
                    st.metric("Features Used", f"{int(metrics_row['n_feat_used'])}")
                    st.metric("RMSE", f"{metrics_row['rmse']:.4f}")
                    st.metric("R²", f"{metrics_row['r2'] * 100:.2f}%")
                    st.metric("Hit Rate", f"{metrics_row['hit_rate']:.2%}")
            
    with tab2:
        st.header("Market Analysis") 
        
        selected_date = st.date_input( "Select Date",   
                                      value=min_date,
                                      min_value=min_date,
                                      max_value=max_date)
        
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
                    <p class="{direction_class}">Direction: {row['Direction'].upper()}</p>
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
        
        st.subheader("Historical Performance")

        if len(selected_data) > 1:
            fig_sp500 = create_line_chart(selected_data, 'date', 'y_true', 'SP500 Actual Performance', '#3b82f6')
            st.plotly_chart(fig_sp500, use_container_width=True)
        
        if len(bond_data_filtered) > 1:
            fig_bond = create_line_chart(bond_data_filtered, 'observation_date', 'DGS10', '10-Year Treasury Rate', '#ef4444')
            st.plotly_chart(fig_bond, use_container_width=True)
            
    with tab3:
        st.header("Investment Optimizer")
        merged_df = st.session_state.merged_df
        month_options = merged_df["month_label"].tolist()
        selected_label = st.selectbox("Select Month", month_options)
        selected_row = merged_df[merged_df["month_label"] == selected_label].iloc[0]
        sp500_weight = selected_row["SP500 weight_pred"] 
        tbill_weight = selected_row["Tbill weight_pred"] 
        port_return = selected_row["portfolio_return_pred"]
        hist_sp500_weight = selected_row["SP500 weight_hist"]
        hist_tbill_weight = selected_row["Tbill weight_hist"]
        hist_return = selected_row["portfolio_return_hist"] 
        col1, col2 = st.columns(2) 
        with col1:
            st.subheader("Predicted Allocation") 
            pie1 = go.Figure(data=[go.Pie(
            labels=["SP500", "T-Bills"],
            values=[sp500_weight, tbill_weight],
            hole=0.4,
            marker_colors=["#4CAF50", "#FF9800"])])
            pie1.update_layout(width=400, height=350)
            st.plotly_chart(pie1)
            
        with col2:
            st.subheader("Historical Mean Allocation")
            pie2 = go.Figure(data=[go.Pie(
            labels=["SP500", "T-Bills"],
            values=[hist_sp500_weight, hist_tbill_weight],
            hole=0.4,
            marker_colors=["#2196F3", "#FFB300"])])
            pie2.update_layout(width=400, height=350)
            st.plotly_chart(pie2)
            
        st.subheader("Investment Recommendations")
        amount = st.number_input("Enter investment amount ($)", min_value=1000, value=10000, step=100)
        
        sp500_amt = amount * sp500_weight
        tbill_amt = amount * tbill_weight
        expected_gain_pred = amount * (port_return / 100)
        hist_sp500_amt = amount * hist_sp500_weight
        hist_tbill_amt = amount * hist_tbill_weight
        expected_gain_hist = amount * (hist_return / 100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Predicted Allocation:**\n"
                   f"- SP500: ${sp500_amt:,.0f} ({sp500_weight:.1%})\n"
                   f"- T-Bills: ${tbill_amt:,.0f} ({tbill_weight:.1%})\n\n"
                   f"**Expected Return:** ${expected_gain_pred:.2f} ({port_return:.2f}%)")       
            st.write("*This return reflects the predicted performance of the model-based allocation.*")
        with col2:
            st.info(f"**Historical Mean Allocation:**\n"
                f"- SP500: ${hist_sp500_amt:,.0f} ({hist_sp500_weight:.1%})\n"
                f"- T-Bills: ${hist_tbill_amt:,.0f} ({hist_tbill_weight:.1%})\n\n"
                f"**Expected Return:** ${expected_gain_hist:.2f} ({hist_return:.2f}%)")
            st.write("*This return reflects the actual performance using historical mean allocation.*")
            
        st.subheader("Monthly Return Table")
        table_df = merged_df[["month_label", "portfolio_return_pred", "portfolio_return_hist"]].copy()
        table_df["Difference"] = table_df["portfolio_return_pred"] - table_df["portfolio_return_hist"]
        table_df.columns = ["Month", "Predicted Return", "Historical Return", "Difference"]
        st.dataframe(table_df.style.format({
        "Predicted Return": "{:.2%}",
        "Historical Return": "{:.2%}",
        "Difference": "{:+.2%}"}))


    with tab4:
        st.header("Performance Comparison")
        
        if 'methodology' in annual_df.columns and 'portfolio return' in annual_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Annual Return Comparison")
                if annual_df["portfolio return"].dtype == 'object':
                    annual_df["portfolio return"] = annual_df["portfolio return"].str.rstrip('%').astype(float)
                
                bar_fig = px.bar(
                    annual_df,
                    x="methodology",
                    y="portfolio return",
                    color="methodology",
                    color_discrete_sequence=["#4CAF50", "#FF9800"],
                    labels={"portfolio return": "Return (%)"},
                    title="Annual Return: MV vs MV + LightGBM" )
                st.plotly_chart(bar_fig, use_container_width=True)

            with col2:
                st.subheader("Sharpe Ratio")
                if len(annual_df) >= 2:
                    st.metric("MV", f"{annual_df.iloc[0]['Sharpe Ratio']:.2f}")
                    st.metric("MV + LightGBM", f"{annual_df.iloc[1]['Sharpe Ratio']:.2f}")

        if 'Date' in monthly_df.columns and 'Historical Mean' in monthly_df.columns and 'Predicted' in monthly_df.columns:
            st.subheader("Monthly Return Comparison")
            monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_df["Date"], 
                y=monthly_df["Historical Mean"], 
                name="Historical Mean", 
                line=dict(color='blue')))
            fig.add_trace(go.Scatter(
                x=monthly_df["Date"], 
                y=monthly_df["Predicted"], 
                name="Predicted", 
                line=dict(color='orange') ))
            fig.update_layout(
                title="Monthly Returns: Actual vs Predicted", 
                yaxis_title="Monthly Return", 
                xaxis_title="Date" )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
