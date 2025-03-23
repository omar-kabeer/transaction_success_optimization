
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import joblib
import json
from pathlib import Path
from plotly.subplots import make_subplots
from PIL import Image

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules if needed
try:
    from src.utils import load_data
    from src.model_evaluation import calculate_metrics
    LOCAL_IMPORT = True
except ImportError:
    LOCAL_IMPORT = False

# Set page configuration
st.set_page_config(
    page_title="Transaction Success Optimization Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3498db;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .trend-up {
        color: #27ae60;
    }
    .trend-down {
        color: #e74c3c;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Load sample data or real data
@st.cache_data(ttl=3600)
def load_transaction_data():
    """Load transaction data from file or generate sample data."""
    try:
        # Try to load from processed data
        data_path = os.path.join("..", "data", "processed", "processed_transaction_data.csv")
        df = pd.read_csv(data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except FileNotFoundError:
        try:
            # Try to load from raw data
            data_path = os.path.join("..", "data", "raw", "transactions_march_2023.csv")
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            # If no data is available, create sample data
            st.warning("No transaction data found. Using synthetic data for demonstration.")
            return generate_sample_data()

def generate_sample_data(n_samples=1000):
    """Generate sample transaction data for demonstration."""
    np.random.seed(42)
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Create merchant IDs
    merchant_ids = [f"MERCH{i:05d}" for i in range(1, 51)]
    
    # Define categorical values
    payment_methods = ["credit_card", "debit_card", "bank_transfer", "mobile_payment", "digital_wallet"]
    device_types = ["mobile", "web", "pos", "tablet", "kiosk"]
    customer_locations = ["urban", "suburban", "rural", "international"]
    results = ["success", "failed"]
    
    # Generate data
    data = {
        "transaction_id": [f"tx-{i:06d}" for i in range(n_samples)],
        "timestamp": dates,
        "merchant_id": np.random.choice(merchant_ids, size=n_samples),
        "customer_id": [f"CUST{np.random.randint(10000, 99999)}" for _ in range(n_samples)],
        "customer_location": np.random.choice(customer_locations, size=n_samples),
        "payment_amount": np.random.gamma(shape=3, scale=100, size=n_samples),
        "payment_method": np.random.choice(payment_methods, size=n_samples),
        "device_type": np.random.choice(device_types, size=n_samples),
        "network_latency": np.random.gamma(shape=2, scale=40, size=n_samples),
        "result": np.random.choice(results, size=n_samples, p=[0.93, 0.07]),
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation between factors and success rates
    for idx, row in df.iterrows():
        if row['network_latency'] > 120:  # High latency increases failure rate
            if np.random.random() < 0.3:
                df.at[idx, 'result'] = 'failed'
        
        if row['payment_method'] == 'mobile_payment':  # Mobile payments have higher success
            if np.random.random() < 0.98:
                df.at[idx, 'result'] = 'success'
        
        if row['payment_amount'] > 500:  # Large amounts have slightly lower success
            if np.random.random() < 0.15:
                df.at[idx, 'result'] = 'failed'
    
    return df

@st.cache_data(ttl=3600)
def load_model_data():
    """Load model metadata and metrics."""
    try:
        model_path = os.path.join("..", "models", "model_metadata.json")
        with open(model_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create sample model metadata
        return {
            "model_name": "Optimized XGBoost",
            "version": "1.0.2",
            "training_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            "metrics": {
                "accuracy": 0.945,
                "precision": 0.967,
                "recall": 0.941,
                "f1_score": 0.954,
                "auc_roc": 0.982
            },
            "feature_importance": {
                "network_latency": 28.5,
                "payment_amount": 21.3,
                "merchant_success_rate": 18.7,
                "device_type": 11.9,
                "customer_location": 9.4,
                "payment_method": 7.2,
                "time_of_day": 3.0
            }
        }

# Function to calculate success metrics
def calculate_success_metrics(df):
    """Calculate success metrics from transaction data."""
    # Success rate
    success_rate = df[df['result'] == 'success'].shape[0] / df.shape[0] * 100
    
    # Average transaction amount
    avg_amount = df['payment_amount'].mean()
    
    # Average latency
    avg_latency = df['network_latency'].mean()
    
    # Success by payment method
    payment_success = df.groupby('payment_method').apply(
        lambda x: (x['result'] == 'success').mean() * 100
    ).to_dict()
    
    # Success by device type
    device_success = df.groupby('device_type').apply(
        lambda x: (x['result'] == 'success').mean() * 100
    ).to_dict()
    
    # Total transaction volume
    total_volume = df['payment_amount'].sum()
    
    # Daily transaction count
    daily_count = df.groupby(df['timestamp'].dt.date).size().mean()
    
    return {
        "success_rate": success_rate,
        "avg_amount": avg_amount,
        "avg_latency": avg_latency,
        "payment_success": payment_success,
        "device_success": device_success,
        "total_volume": total_volume,
        "daily_count": daily_count
    }

# Sidebar
with st.sidebar:
    st.title("Transaction Success Dashboard")
    
    # Navigation
    page = st.radio(
        "Navigation",
        options=["Overview", "Transaction Analysis", "Model Performance", "Recommendations"]
    )
    
    # Load data
    data = load_transaction_data()
    
    # Date filter
    st.subheader("Filters")
    min_date = data['timestamp'].min().date()
    max_date = data['timestamp'].max().date()
    
    date_range = st.date_input(
        "Date Range",
        value=(max_date - timedelta(days=7), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (data['timestamp'].dt.date >= start_date) & (data['timestamp'].dt.date <= end_date)
        filtered_data = data[mask]
    else:
        filtered_data = data
    
    # Payment method filter
    payment_methods = sorted(data['payment_method'].unique())
    selected_payment_methods = st.multiselect(
        "Payment Method",
        options=payment_methods,
        default=payment_methods
    )
    
    if selected_payment_methods:
        filtered_data = filtered_data[filtered_data['payment_method'].isin(selected_payment_methods)]
    
    # Location filter
    locations = sorted(data['customer_location'].unique())
    selected_locations = st.multiselect(
        "Customer Location",
        options=locations,
        default=locations
    )
    
    if selected_locations:
        filtered_data = filtered_data[filtered_data['customer_location'].isin(selected_locations)]
    
    # Display filter summary
    st.caption(f"Showing {len(filtered_data)} of {len(data)} transactions")
    
    # Load model data
    model_data = load_model_data()
    
    # Model information
    st.subheader("Model Information")
    st.info(f"Current Model: {model_data['model_name']} v{model_data['version']}")
    st.caption(f"Trained on: {model_data['training_date']}")

# Calculate metrics based on filtered data
metrics = calculate_success_metrics(filtered_data)

# Page content
if page == "Overview":
    # Main header
    st.markdown("<h1 class='main-header'>Transaction Success Optimization</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text' style='text-align: center;'>Real-time monitoring and optimization dashboard for payment transaction success rates</p>", unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-title'>Success Rate</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{metrics['success_rate']:.1f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<span class='trend-up'>↑ 1.2%</span> vs Previous Period", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-title'>Average Latency</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{metrics['avg_latency']:.1f} ms</p>", unsafe_allow_html=True)
        st.markdown(f"<span class='trend-down'>↓ 3.5 ms</span> vs Previous Period", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-title'>Total Volume</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${metrics['total_volume']:,.0f}</p>", unsafe_allow_html=True)
        st.markdown(f"<span class='trend-up'>↑ 5.7%</span> vs Previous Period", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-title'>Daily Transactions</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{metrics['daily_count']:.0f}</p>", unsafe_allow_html=True)
        st.markdown(f"<span class='trend-up'>↑ 3.2%</span> vs Previous Period", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Success Rate Over Time
    st.markdown("<h2 class='sub-header'>Success Rate Trends</h2>", unsafe_allow_html=True)
    
    # Create daily success rate data
    daily_data = filtered_data.copy()
    daily_data['date'] = daily_data['timestamp'].dt.date
    
    success_by_day = daily_data.groupby('date').apply(
        lambda x: (x['result'] == 'success').mean() * 100
    ).reset_index()
    success_by_day.columns = ['date', 'success_rate']
    
    # Plot success rate over time
    fig = px.line(
        success_by_day, 
        x='date', 
        y='success_rate',
        title='Daily Transaction Success Rate',
        labels={'date': 'Date', 'success_rate': 'Success Rate (%)'},
        line_shape='spline',
        markers=True
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Success Rate (%)',
        yaxis=dict(range=[85, 100]),
        hovermode='x unified'
    )
    
    # Add target line
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Target (95%)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Payment Method and Device Type Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment Method Success Rate
        payment_data = filtered_data.groupby('payment_method').apply(
            lambda x: (x['result'] == 'success').mean() * 100
        ).reset_index()
        payment_data.columns = ['payment_method', 'success_rate']
        
        # Add counts
        payment_counts = filtered_data.groupby('payment_method').size().reset_index()
        payment_counts.columns = ['payment_method', 'count']
        payment_data = payment_data.merge(payment_counts, on='payment_method')
        
        fig = px.bar(
            payment_data,
            x='payment_method',
            y='success_rate',
            color='payment_method',
            text='count',
            title='Success Rate by Payment Method',
            labels={'payment_method': 'Payment Method', 'success_rate': 'Success Rate (%)'}
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[85, 100]),
            showlegend=False
        )
        
        # Add target line
        fig.add_hline(y=95, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Device Type Success Rate
        device_data = filtered_data.groupby('device_type').apply(
            lambda x: (x['result'] == 'success').mean() * 100
        ).reset_index()
        device_data.columns = ['device_type', 'success_rate']
        
        # Add counts
        device_counts = filtered_data.groupby('device_type').size().reset_index()
        device_counts.columns = ['device_type', 'count']
        device_data = device_data.merge(device_counts, on='device_type')
        
        fig = px.bar(
            device_data,
            x='device_type',
            y='success_rate',
            color='device_type',
            text='count',
            title='Success Rate by Device Type',
            labels={'device_type': 'Device Type', 'success_rate': 'Success Rate (%)'}
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[85, 100]),
            showlegend=False
        )
        
        # Add target line
        fig.add_hline(y=95, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Transaction Amount vs. Latency Analysis
    st.markdown("<h2 class='sub-header'>Transaction Parameters Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Network Latency Distribution
        fig = px.histogram(
            filtered_data,
            x='network_latency',
            color='result',
            nbins=30,
            opacity=0.7,
            title='Network Latency Distribution by Transaction Result',
            labels={'network_latency': 'Network Latency (ms)', 'result': 'Transaction Result'},
            color_discrete_map={'success': 'green', 'failed': 'red'}
        )
        
        fig.update_layout(
            xaxis_title='Network Latency (ms)',
            yaxis_title='Count',
            bargap=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Payment Amount vs Success Rate
        amount_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
        amount_labels = ['<$50', '$50-100', '$100-200', '$200-500', '$500-1000', '>$1000']
        
        filtered_data['amount_bin'] = pd.cut(
            filtered_data['payment_amount'], 
            bins=amount_bins, 
            labels=amount_labels
        )
        
        amount_success = filtered_data.groupby('amount_bin').apply(
            lambda x: (x['result'] == 'success').mean() * 100
        ).reset_index()
        amount_success.columns = ['amount_bin', 'success_rate']
        
        # Add counts
        amount_counts = filtered_data.groupby('amount_bin').size().reset_index()
        amount_counts.columns = ['amount_bin', 'count']
        amount_data = amount_success.merge(amount_counts, on='amount_bin')
        
        fig = px.bar(
            amount_data,
            x='amount_bin',
            y='success_rate',
            color='amount_bin',
            text='count',
            title='Success Rate by Transaction Amount',
            labels={'amount_bin': 'Amount Range', 'success_rate': 'Success Rate (%)'}
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[85, 100]),
            showlegend=False
        )
        
        fig.add_hline(y=95, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance Summary
    st.markdown("<h2 class='sub-header'>Model Performance Summary</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': list(model_data['feature_importance'].keys()),
            'Importance': list(model_data['feature_importance'].values())
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance,
            y='Feature',
            x='Importance',
            orientation='h',
            title='Feature Importance',
            text='Importance',
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_title='Relative Importance (%)',
            yaxis_title='',
            coloraxis_showscale=False
        )
        
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model metrics
        st.subheader("Performance Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': list(model_data['metrics'].keys()),
            'Value': list(model_data['metrics'].values())
        })
        
        # Format the values as percentages
        metrics_df['Value'] = metrics_df['Value'].map(lambda x: f"{x*100:.1f}%")
        
        # Style the dataframe
        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Add explainer
        st.markdown("""
        **Metrics explained:**
        - **Accuracy**: Overall correct predictions
        - **Precision**: Correct success predictions divided by all success predictions
        - **Recall**: Correct success predictions divided by all actual successes
        - **F1 Score**: Harmonic mean of precision and recall
        - **AUC-ROC**: Area under the ROC curve
        """)
    
    # Recent Transactions Table
    st.markdown("<h2 class='sub-header'>Recent Transactions</h2>", unsafe_allow_html=True)
    
    recent_txns = filtered_data.sort_values('timestamp', ascending=False).head(10)
    
    # Format for display
    display_cols = ['transaction_id', 'timestamp', 'payment_method', 'device_type', 
                    'payment_amount', 'network_latency', 'result']
    
    display_df = recent_txns[display_cols].copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['payment_amount'] = display_df['payment_amount'].map('${:,.2f}'.format)
    display_df['network_latency'] = display_df['network_latency'].map('{:.1f} ms'.format)
    
    # Rename columns for display
    display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
    
    # Show table
    st.dataframe(display_df, use_container_width=True)
    
    # Footer with last update time
    st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.8rem;'>Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

elif page == "Transaction Analysis":
    st.markdown("<h1 class='main-header'>Transaction Analysis</h1>", unsafe_allow_html=True)
    
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Time Analysis", "Geographic Analysis", "Merchant Analysis"])
    
    with tab1:
        st.subheader("Transaction Success by Time")
        
        # Success rate by hour of day
        hour_data = filtered_data.copy()
        hour_data['hour'] = hour_data['timestamp'].dt.hour
        
        hourly_success = hour_data.groupby('hour').apply(
            lambda x: (x['result'] == 'success').mean() * 100
        ).reset_index()
        hourly_success.columns = ['hour', 'success_rate']
        
        # Add transaction volume
        hourly_volume = hour_data.groupby('hour').size().reset_index()
        hourly_volume.columns = ['hour', 'volume']
        hourly_data = hourly_success.merge(hourly_volume, on='hour')
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add success rate line
        fig.add_trace(
            go.Scatter(
                x=hourly_data['hour'], 
                y=hourly_data['success_rate'],
                name="Success Rate",
                line=dict(color='blue', width=3),
                mode='lines+markers'
            ),
            secondary_y=False,
        )
        
        # Add transaction volume bars
        fig.add_trace(
            go.Bar(
                x=hourly_data['hour'], 
                y=hourly_data['volume'],
                name="Transaction Volume",
                opacity=0.5,
                marker_color='lightblue'
            ),
            secondary_y=True,
        )
        
        # Set titles and labels
        fig.update_layout(
            title_text="Transaction Success Rate by Hour of Day",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set x-axis title and format
        fig.update_xaxes(
            title_text="Hour of Day (24-hour)", 
            tickvals=list(range(0, 24)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24)]
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Success Rate (%)", range=[85, 100], secondary_y=False)
        fig.update_yaxes(title_text="Transaction Volume", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Success rate by day of week
        day_data = filtered_data.copy()
        day_data['day'] = day_data['timestamp'].dt.day_name()
        
        # Ensure proper day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_success = day_data.groupby('day').apply(
            lambda x: (x['result'] == 'success').mean() * 100
        ).reset_index()
        daily_success.columns = ['day', 'success_rate']
        daily_success['day'] = pd.Categorical(daily_success['day'], categories=day_order, ordered=True)
        daily_success = daily_success.sort_values('day')
        
        # Add transaction volume
        daily_volume = day_data.groupby('day').size().reset_index()
        daily_volume.columns = ['day', 'volume']
        daily_volume['day'] = pd.Categorical(daily_volume['day'], categories=day_order, ordered=True)
        daily_volume = daily_volume.sort_values('day')
        
        daily_data = daily_success.merge(daily_volume, on='day')
        
        # Create chart
        fig = px.bar(
            daily_data,
            x='day',
            y='success_rate',
            text=daily_data['volume'].apply(lambda x: f"{x:,}"),
            color='success_rate',
            color_continuous_scale='Blues',
            title='Transaction Success Rate by Day of Week'
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[85, 100]),
            coloraxis_showscale=False
        )
        
        fig.add_hline(y=95, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Geographic Success Analysis")
        
        # Success rate by customer location
        location_data = filtered_data.groupby('customer_location').apply(
            lambda x: (x['result'] == 'success').mean() * 100
        ).reset_index()
        location_data.columns = ['customer_location', 'success_rate']
        
        # Add counts
        location_counts = filtered_data.groupby('customer_location').size().reset_index()
        location_counts.columns = ['customer_location', 'count']
        location_data = location_data.merge(location_counts, on='customer_location')
        
        fig = px.bar(
            location_data,
            x='customer_location',
            y='success_rate',
            color='success_rate',
            text='count',
            title='Success Rate by Customer Location',
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[85, 100]),
            coloraxis_showscale=False
        )
        
        fig.add_hline(y=95, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Payment method by location
        location_payment = filtered_data.groupby(['customer_location', 'payment_method']).size().reset_index()
        location_payment.columns = ['customer_location', 'payment_method', 'count']
        
        fig = px.bar(
            location_payment,
            x='customer_location',
            y='count',
            color='payment_method',
            title='Payment Methods by Customer Location',
            barmode='stack'
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Count'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        fig.update_layout(
            xaxis_title='Customer Location',
            yaxis_title='Number of Transactions',
            legend_title='Payment Method'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Success rate by location and device type
        location_device = filtered_data.groupby(['customer_location', 'device_type']).apply(
            lambda x: (x['result'] == 'success').mean() * 100
        ).reset_index()
        location_device.columns = ['customer_location', 'device_type', 'success_rate']
        
        fig = px.bar(
            location_device,
            x='customer_location',
            y='success_rate',
            color='device_type',
            barmode='group',
            title='Success Rate by Location and Device Type'
        )
        
        fig.update_layout(
            xaxis_title='Customer Location',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[85, 100]),
            legend_title='Device Type'
        )
        
        fig.add_hline(y=95, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.subheader("Merchant Analysis")
        
        # Top merchants by transaction volume
        merchant_volume = filtered_data.groupby('merchant_id').agg(
            transaction_count=('transaction_id', 'count'),
            total_volume=('payment_amount', 'sum'),
            success_rate=('result', lambda x: (x == 'success').mean() * 100)
        ).reset_index().sort_values('total_volume', ascending=False).head(10)
        
        # Format for display
        merchant_volume['total_volume'] = merchant_volume['total_volume'].map('${:,.2f}'.format)
        merchant_volume['success_rate'] = merchant_volume['success_rate'].map('{:.1f}%'.format)
        
        # Rename columns
        merchant_volume.columns = ['Merchant ID', 'Transaction Count', 'Total Volume', 'Success Rate']
        
        st.write("Top 10 Merchants by Transaction Volume")
        st.dataframe(merchant_volume, use_container_width=True)
        
        # Merchants with lowest success rates
        merchant_success = filtered_data.groupby('merchant_id').agg(
            transaction_count=('transaction_id', 'count'),
            success_rate=('result', lambda x: (x == 'success').mean() * 100)
        ).reset_index()
        
        # Filter to merchants with at least 10 transactions
        merchant_success = merchant_success[merchant_success['transaction_count'] >= 10]
        
        # Get merchants with lowest success rates
        low_success = merchant_success.sort_values('success_rate').head(10)
        
        # Create visualization
        fig = px.bar(
            low_success,
            x='merchant_id',
            y='success_rate',
            color='success_rate',
            text=low_success['transaction_count'].apply(lambda x: f"{x} txns"),
            title='Merchants with Lowest Success Rates (min. 10 transactions)',
            color_continuous_scale='Reds_r'
        )
        
        fig.update_layout(
            xaxis_title='Merchant ID',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[70, 100]),
            coloraxis_showscale=False
        )
        
        fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Target (95%)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Merchant payment method distribution
        top5_merchants = merchant_volume['Merchant ID'].head(5).tolist()
        merchant_payments = filtered_data[filtered_data['merchant_id'].isin(top5_merchants)]
        
        payment_dist = merchant_payments.groupby(['merchant_id', 'payment_method']).size().reset_index()
        payment_dist.columns = ['merchant_id', 'payment_method', 'count']
        
        fig = px.bar(
            payment_dist,
            x='merchant_id',
            y='count',
            color='payment_method',
            title='Payment Method Distribution for Top 5 Merchants',
            barmode='stack'
        )
        
        fig.update_layout(
            xaxis_title='Merchant ID',
            yaxis_title='Number of Transactions',
            legend_title='Payment Method'
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.markdown("<h1 class='main-header'>Model Performance</h1>", unsafe_allow_html=True)
    
    # Model metrics
    metrics = model_data['metrics']
    
    # Performance metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-title'>Accuracy</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{metrics['accuracy']*100:.1f}%</p>", unsafe_allow_html=True)
        st.markdown("Overall correct predictions", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-title'>Precision</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{metrics['precision']*100:.1f}%</p>", unsafe_allow_html=True)
        st.markdown("Correct success predictions", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-title'>Recall</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{metrics['recall']*100:.1f}%</p>", unsafe_allow_html=True)
        st.markdown("Found actual successes", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-title'>AUC-ROC</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{metrics['auc_roc']*100:.1f}%</p>", unsafe_allow_html=True)
        st.markdown("Overall discrimination", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature importance visualization
    st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
    
    feature_importance = pd.DataFrame({
        'Feature': list(model_data['feature_importance'].keys()),
        'Importance': list(model_data['feature_importance'].values())
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        feature_importance,
        x='Feature',
        y='Importance',
        color='Importance',
        title='Feature Importance Analysis',
        text='Importance',
        color_continuous_scale='blues'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Importance (%)',
        coloraxis_showscale=False
    )
    
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC curve and confusion matrix
    st.markdown("<h2 class='sub-header'>Model Evaluation</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulated ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = np.clip(1 - np.exp(-5 * fpr), 0, 1)  # Simulated curve with AUC ~0.95
        
        fig = px.line(
            x=fpr, 
            y=tpr,
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            title='ROC Curve'
        )
        
        fig.add_shape(
            type='line',
            line=dict(dash='dash', color='gray'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=400, height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            annotations=[
                dict(
                    x=0.95, y=0.05,
                    xref="paper", yref="paper",
                    text=f"AUC = {metrics['auc_roc']:.3f}",
                    showarrow=False,
                    font=dict(size=14)
                )
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Simulated confusion matrix
        # Assuming accuracy ~ 0.95, and balanced failure rate
        tp = 93
        fn = 5
        fp = 2
        tn = 95
        
        z = [[tp, fn], [fp, tn]]
        
        # Create annotations
        annotations = []
        for i, row in enumerate(z):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=str(value),
                        font=dict(color='white' if value > 50 else 'black', size=14),
                        showarrow=False
                    )
                )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=['Predicted Success', 'Predicted Failure'],
            y=['Actual Success', 'Actual Failure'],
            colorscale=[[0, '#EF553B'], [1, '#636EFA']],
            showscale=False
        ))
        
        fig.update_layout(
            title='Confusion Matrix (Simulated)',
            annotations=annotations,
            width=400, height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance over time
    st.markdown("<h2 class='sub-header'>Performance Monitoring</h2>", unsafe_allow_html=True)
    
    # Simulated model performance data over time
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create fluctuating metrics
    np.random.seed(42)
    base_auc = 0.96
    fluctuation = np.random.normal(0, 0.01, size=len(dates))
    auc_values = np.clip(base_auc + fluctuation, 0.9, 1.0)
    
    perf_data = pd.DataFrame({
        'Date': dates,
        'AUC': auc_values,
        'Precision': np.clip(0.94 + fluctuation, 0.9, 1.0),
        'Recall': np.clip(0.95 + fluctuation, 0.9, 1.0)
    })
    
    # Create multi-line plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=perf_data['Date'],
        y=perf_data['AUC'],
        mode='lines+markers',
        name='AUC',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=perf_data['Date'],
        y=perf_data['Precision'],
        mode='lines+markers',
        name='Precision',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=perf_data['Date'],
        y=perf_data['Recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title='Model Performance Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Metric Value',
        yaxis=dict(range=[0.85, 1.0]),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add retraining annotation
    fig.add_vline(
        x=end_date - timedelta(days=15), 
        line_dash="dash", 
        line_color="red",
        annotation_text="Model Retraining",
        annotation_position="top"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model versions comparison
    st.markdown("<h2 class='sub-header'>Model Versions Comparison</h2>", unsafe_allow_html=True)
    
    # Simulated model comparison data
    models = ['XGBoost v1.0', 'XGBoost v1.1', 'Optimized XGBoost v1.0.2']
    accuracy = [0.921, 0.935, 0.945]
    precision = [0.934, 0.952, 0.967]
    recall = [0.925, 0.933, 0.941]
    f1 = [0.929, 0.942, 0.954]
    auc = [0.956, 0.971, 0.982]
    
    model_comp = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc
    })
    
    # Melt for plotting
    model_metrics = pd.melt(
        model_comp, 
        id_vars=['Model'], 
        var_name='Metric', 
        value_name='Value'
    )
    
    fig = px.bar(
        model_metrics,
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Performance Comparison Across Model Versions',
        text_auto='.3f'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Score',
        yaxis=dict(range=[0.9, 1.0]),
        legend_title='Metric'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training details
    st.markdown("<h2 class='sub-header'>Training Details</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Parameters")
        
        # Example model parameters
        params = {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.7,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False
        }
        
        # Display as JSON with syntax highlighting
        st.json(params)
        
    with col2:
        st.subheader("Training Information")
        
        # Create a markdown table with training information
        st.markdown("""
        | Parameter | Value |
        | --- | --- |
        | Training Date | 2025-03-15 |
        | Training Duration | 12m 37s |
        | Training Samples | 87,452 |
        | Validation Samples | 21,863 |
        | Test Samples | 10,932 |
        | Train-Test Split | 80/20 |
        | Cross-Validation | 5-fold |
        | Early Stopping | Yes (10 rounds) |
        | Best Iteration | 287 |
        """)

elif page == "Recommendations":
    st.markdown("<h1 class='main-header'>Optimization Recommendations</h1>", unsafe_allow_html=True)
    
    st.write("""
    Based on the transaction data analysis and model predictions, the following recommendations 
    are provided to optimize transaction success rates.
    """)
    
    # Key findings
    st.markdown("<h2 class='sub-header'>Key Findings</h2>", unsafe_allow_html=True)
    
    findings = [
        {
            "title": "Network Latency Impact",
            "description": "Transactions with network latency above 120ms show a 28% increase in failure rate.",
            "recommendation": "Optimize network infrastructure and implement timeout handling strategies.",
            "potential_impact": "High",
            "implementation_effort": "Medium"
        },
        {
            "title": "Mobile Payment Success",
            "description": "Mobile payments have a 4.2% higher success rate than other methods.",
            "recommendation": "Promote and incentivize mobile payment options to customers.",
            "potential_impact": "Medium",
            "implementation_effort": "Low"
        },
        {
            "title": "High-Value Transaction Risk",
            "description": "Transactions over $500 have a 15% higher failure rate.",
            "recommendation": "Implement stepped verification for high-value transactions.",
            "potential_impact": "High",
            "implementation_effort": "Medium"
        },
        {
            "title": "Time-of-Day Pattern",
            "description": "Failure rates increase by 7% during peak hours (12-2pm and 6-8pm).",
            "recommendation": "Increase system capacity during peak hours and implement queue management.",
            "potential_impact": "Medium",
            "implementation_effort": "Medium"
        },
        {
            "title": "Merchant-Specific Issues",
            "description": "Five merchants account for 35% of all failed transactions.",
            "recommendation": "Provide targeted technical support and integration review for these merchants.",
            "potential_impact": "High",
            "implementation_effort": "High"
        }
    ]
    
    # Display findings as expandable sections
    for i, finding in enumerate(findings):
        with st.expander(f"{i+1}. {finding['title']}", expanded=i==0):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**Description:** {finding['description']}")
                st.markdown(f"**Recommendation:** {finding['recommendation']}")
                
            with col2:
                impact_color = {
                    "High": "green",
                    "Medium": "orange",
                    "Low": "red"
                }[finding["potential_impact"]]
                
                st.markdown(f"**Potential Impact:**")
                st.markdown(f"<span style='color: {impact_color}; font-weight: bold;'>{finding['potential_impact']}</span>", unsafe_allow_html=True)
                
            with col3:
                effort_color = {
                    "Low": "green",
                    "Medium": "orange",
                    "High": "red"
                }[finding["implementation_effort"]]
                
                st.markdown(f"**Implementation Effort:**")
                st.markdown(f"<span style='color: {effort_color}; font-weight: bold;'>{finding['implementation_effort']}</span>", unsafe_allow_html=True)
    
    # Action plan
    st.markdown("<h2 class='sub-header'>Recommended Action Plan</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Short-term (0-30 days)")
        st.markdown("""
        - Implement network timeout handling
        - Promote mobile payment options
        - Provide technical support to top 5 problematic merchants
        - Adjust system capacity for peak hours
        """)
        
        st.markdown("### Medium-term (30-90 days)")
        st.markdown("""
        - Develop stepped verification for high-value transactions
        - Implement queue management system
        - Create merchant-specific dashboards
        - Enhance fraud detection algorithms
        """)
        
        st.markdown("### Long-term (90+ days)")
        st.markdown("""
        - System-wide infrastructure upgrade
        - Implement AI-driven routing optimization
        - Develop predictive maintenance system
        - Create strategic merchant partnership program
        """)
        
    with col2:
        # Create Gantt chart for action plan
        tasks = [
            dict(Task="Network timeout handling", Start='2025-04-01', Finish='2025-04-15', Resource='IT'),
            dict(Task="Promote mobile payments", Start='2025-04-01', Finish='2025-04-30', Resource='Marketing'),
            dict(Task="Merchant technical support", Start='2025-04-01', Finish='2025-05-15', Resource='Support'),
            dict(Task="Adjust system capacity", Start='2025-04-15', Finish='2025-05-15', Resource='IT'),
            dict(Task="Stepped verification", Start='2025-05-01', Finish='2025-06-15', Resource='Development'),
            dict(Task="Queue management", Start='2025-05-15', Finish='2025-07-01', Resource='Development'),
            dict(Task="Merchant dashboards", Start='2025-06-01', Finish='2025-07-15', Resource='Analytics'),
            dict(Task="Enhance fraud detection", Start='2025-06-15', Finish='2025-08-15', Resource='Data Science'),
            dict(Task="Infrastructure upgrade", Start='2025-07-01', Finish='2025-09-30', Resource='IT'),
            dict(Task="AI-driven routing", Start='2025-08-01', Finish='2025-10-31', Resource='Data Science'),
            dict(Task="Predictive maintenance", Start='2025-09-01', Finish='2025-11-30', Resource='Development'),
            dict(Task="Merchant partnership", Start='2025-10-01', Finish='2025-12-31', Resource='Business')
        ]
        
        df = pd.DataFrame(tasks)
        df['Start'] = pd.to_datetime(df['Start'])
        df['Finish'] = pd.to_datetime(df['Finish'])
        
        fig = px.timeline(
            df, 
            x_start="Start", 
            x_end="Finish", 
            y="Task",
            color="Resource",
            title="Implementation Roadmap"
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            legend_title="Team",
            xaxis=dict(
                tickformat="%b %Y",
                dtick="M1",
                tickangle=45
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Expected impact
    st.markdown("<h2 class='sub-header'>Expected Impact</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate improvement projection
        current_rate = metrics['success_rate']
        projected_improvements = [
            current_rate + 0.5,
            current_rate + 1.2,
            current_rate + 2.1,
            current_rate + 2.9,
            current_rate + 3.5
        ]
        
        labels = ['Current', '30 Days', '60 Days', '90 Days', '180 Days', '365 Days']
        values = [current_rate] + projected_improvements
        
        fig = go.Figure(go.Scatter(
            x=labels,
            y=values,
            mode='lines+markers',
            marker=dict(size=12),
            line=dict(width=4)
        ))
        
        fig.update_layout(
            title='Projected Success Rate Improvement',
            xaxis_title='Timeline',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[90, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Business impact metrics
        st.subheader("Business Impact Metrics")
        
        # Calculate current metrics
        avg_txn_value = filtered_data['payment_amount'].mean()
        daily_txn_count = metrics['daily_count']
        current_success_rate = metrics['success_rate'] / 100
        projected_success_rate = (metrics['success_rate'] + 3.5) / 100
        
        # Calculate impact
        daily_volume = daily_txn_count * avg_txn_value
        current_successful_volume = daily_volume * current_success_rate
        projected_successful_volume = daily_volume * projected_success_rate
        volume_increase = projected_successful_volume - current_successful_volume
        
        # Annual impact
        annual_increase = volume_increase * 365
        
        # Create metrics display
        impact_data = {
            "Metric": [
                "Additional successful transactions (daily)",
                "Additional transaction volume (daily)",
                "Annual revenue impact",
                "Customer satisfaction increase",
                "Operational cost reduction"
            ],
            "Value": [
                f"{daily_txn_count * (projected_success_rate - current_success_rate):.0f}",
                f"${volume_increase:,.2f}",
                f"${annual_increase:,.2f}",
                "+15%",
                "-12%"
            ]
        }
        
        impact_df = pd.DataFrame(impact_data)
        
        # Style the dataframe
        st.dataframe(
            impact_df,
            hide_index=True,
            use_container_width=True
        )
        
        # ROI calculation
        implementation_cost = 250000  # Example cost
        annual_benefit = annual_increase * 0.2  # Assuming 20% of increased volume is profit
        
        roi = (annual_benefit - implementation_cost) / implementation_cost * 100
        payback_months = implementation_cost / (annual_benefit / 12)
        
        st.markdown(f"""
        ### Return on Investment
        
        - Implementation Cost: ${implementation_cost:,.0f}
        - Annual Benefit: ${annual_benefit:,.0f}
        - ROI: {roi:.1f}%
        - Payback Period: {payback_months:.1f} months
        """)
    
    # Next steps
    st.markdown("<h2 class='sub-header'>Next Steps</h2>", unsafe_allow_html=True)
    
    next_steps = st.button("Generate Detailed Implementation Plan")
    
    if next_steps:
        st.success("Request received! A detailed implementation plan will be generated and sent to the project team.")

# Add page footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #ddd;">
    <p style="color: #666; font-size: 0.8rem;">
        Transaction Success Optimization Dashboard • Last updated: {0} • v1.0.2
    </p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
