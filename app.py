"""
Interactive Sales Prediction Web Application

This web app allows users to adjust prediction parameters and visualize
sales forecasts with confidence intervals in real-time.
"""

import streamlit as st
import numpy as np

import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Sales Prediction with Confidence Intervals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stSlider {
        padding-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(filename='data/sales_data.txt'):
    """Load sales data from file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(float(line.strip()))
    return np.array(data)

def create_sequences(data, lookback=6):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

@st.cache_data
def train_models(train_data, lookback, lower_percentile, upper_percentile):
    """Train three quantile regression models"""
    X_train, y_train = create_sequences(train_data, lookback)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    # Calculate quantile values (e.g., 90% CI means 5th and 95th percentiles)
    alpha_lower = (100 - lower_percentile) / 200  # Convert to 0-1 scale
    alpha_upper = 1 - alpha_lower
    
    # Model for lower bound
    model_lower = GradientBoostingRegressor(
        loss='quantile', alpha=alpha_lower,
        n_estimators=800, max_depth=7, learning_rate=0.015,
        min_samples_split=2, min_samples_leaf=1,
        subsample=0.85, max_features=None,
        random_state=42
    )
    model_lower.fit(X_train_scaled, y_train_scaled)
    
    # Model for median
    model_median = GradientBoostingRegressor(
        loss='quantile', alpha=0.50,
        n_estimators=800, max_depth=7, learning_rate=0.015,
        min_samples_split=2, min_samples_leaf=1,
        subsample=0.85, max_features=None,
        random_state=42
    )
    model_median.fit(X_train_scaled, y_train_scaled)
    
    # Model for upper bound
    model_upper = QuantileRegressor(quantile=alpha_upper, alpha=0.01, solver='highs')
    model_upper.fit(X_train_scaled, y_train_scaled)
    
    return model_lower, model_median, model_upper, scaler_X, scaler_y, X_train, y_train

def generate_predictions(train_data, test_data, lookback, lower_percentile, upper_percentile):
    """Generate predictions with confidence intervals"""
    model_lower, model_median, model_upper, scaler_X, scaler_y, X_train, y_train = train_models(
        train_data, lookback, lower_percentile, upper_percentile
    )
    
    # Use a SINGLE sequence based on median predictions to avoid divergence
    # This prevents the bounds from crossing and reduces oscillations
    current_sequence = train_data[-lookback:].copy()
    
    predictions_median = []
    predictions_lower = []
    predictions_upper = []
    
    # Generate predictions
    for i in range(len(test_data)):
        # All three models use the SAME input sequence
        seq_scaled = scaler_X.transform(current_sequence.reshape(1, -1))
        
        pred_median_scaled = model_median.predict(seq_scaled)[0]
        pred_lower_scaled = model_lower.predict(seq_scaled)[0]
        pred_upper_scaled = model_upper.predict(seq_scaled)[0]
        
        pred_median = scaler_y.inverse_transform([[pred_median_scaled]])[0][0]
        pred_lower = scaler_y.inverse_transform([[pred_lower_scaled]])[0][0]
        pred_upper = scaler_y.inverse_transform([[pred_upper_scaled]])[0][0]
        
        # Enforce bound constraints: lower < median < upper
        # This prevents crossing and maintains proper confidence intervals
        pred_lower = min(pred_lower, pred_median)
        pred_upper = max(pred_upper, pred_median)
        
        # Add stability constraint: limit extreme predictions
        # Prevent exponential growth by capping predictions relative to historical range
        data_range = train_data.max() - train_data.min()
        data_mean = train_data.mean()
        max_deviation = 3 * data_range  # Allow up to 3x historical range
        
        pred_median = np.clip(pred_median, data_mean - max_deviation, data_mean + max_deviation)
        pred_lower = np.clip(pred_lower, data_mean - max_deviation, data_mean + max_deviation)
        pred_upper = np.clip(pred_upper, data_mean - max_deviation, data_mean + max_deviation)
        
        predictions_median.append(pred_median)
        predictions_lower.append(pred_lower)
        predictions_upper.append(pred_upper)
        
        # Update sequence with median prediction (not lower/upper to avoid divergence)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred_median
    
    # Calculate metrics
    predictions_median = np.array(predictions_median)
    predictions_lower = np.array(predictions_lower)
    predictions_upper = np.array(predictions_upper)
    
    # Training metrics
    train_pred_lower = []
    train_pred_median = []
    train_pred_upper = []
    
    X_train_scaled = scaler_X.transform(X_train)
    for i in range(len(X_train_scaled)):
        pred_l = scaler_y.inverse_transform([[model_lower.predict(X_train_scaled[i:i+1])[0]]])[0][0]
        pred_m = scaler_y.inverse_transform([[model_median.predict(X_train_scaled[i:i+1])[0]]])[0][0]
        pred_u = scaler_y.inverse_transform([[model_upper.predict(X_train_scaled[i:i+1])[0]]])[0][0]
        train_pred_lower.append(pred_l)
        train_pred_median.append(pred_m)
        train_pred_upper.append(pred_u)
    
    train_pred_lower = np.array(train_pred_lower)
    train_pred_median = np.array(train_pred_median)
    train_pred_upper = np.array(train_pred_upper)
    
    train_in_bounds = np.sum((y_train >= train_pred_lower) & (y_train <= train_pred_upper))
    train_coverage = train_in_bounds / len(y_train) * 100
    
    # Test metrics
    actual_in_bounds = np.sum((test_data >= predictions_lower) & (test_data <= predictions_upper))
    test_coverage = actual_in_bounds / len(test_data) * 100
    
    mae = np.mean(np.abs(test_data - predictions_median))
    mape = np.mean(np.abs((test_data - predictions_median) / test_data)) * 100
    
    return {
        'predictions_median': predictions_median,
        'predictions_lower': predictions_lower,
        'predictions_upper': predictions_upper,
        'train_coverage': train_coverage,
        'test_coverage': test_coverage,
        'mae': mae,
        'mape': mape,
        'train_in_bounds': train_in_bounds,
        'train_total': len(y_train),
        'test_in_bounds': actual_in_bounds,
        'test_total': len(test_data)
    }

def create_plot(sales_data, split_idx, results):
    """Create interactive Plotly chart"""
    train_data = sales_data[:split_idx]
    test_data = sales_data[split_idx:]
    
    train_indices = np.arange(len(train_data))
    test_indices = np.arange(len(train_data), len(train_data) + len(test_data))
    
    # Create prediction indices starting from the last training point
    # This ensures predictions connect to the last known value
    prediction_indices = np.concatenate([[len(train_data) - 1], test_indices])
    
    # Prepend the last training value to predictions for continuity
    last_train_value = train_data[-1]
    predictions_median_with_start = np.concatenate([[last_train_value], results['predictions_median']])
    predictions_lower_with_start = np.concatenate([[last_train_value], results['predictions_lower']])
    predictions_upper_with_start = np.concatenate([[last_train_value], results['predictions_upper']])
    
    fig = go.Figure()
    
    # Training data
    fig.add_trace(go.Scatter(
        x=train_indices,
        y=train_data,
        mode='lines',
        name='Training Data',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Month %{x}</b><br>Sales: %{y:.2f}<extra></extra>'
    ))
    
    # Test data - connect to last training point
    test_with_connection_x = np.concatenate([[len(train_data) - 1], test_indices])
    test_with_connection_y = np.concatenate([[last_train_value], test_data])
    fig.add_trace(go.Scatter(
        x=test_with_connection_x,
        y=test_with_connection_y,
        mode='lines',
        name='Actual Test Data',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='<b>Month %{x}</b><br>Sales: %{y:.2f}<extra></extra>'
    ))
    
    # Median prediction - starts from last training point
    fig.add_trace(go.Scatter(
        x=prediction_indices,
        y=predictions_median_with_start,
        mode='lines',
        name='Median Prediction',
        line=dict(color='#d62728', width=2, dash='dash'),
        hovertemplate='<b>Month %{x}</b><br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    # Confidence interval fill - starts from last training point
    fig.add_trace(go.Scatter(
        x=np.concatenate([prediction_indices, prediction_indices[::-1]]),
        y=np.concatenate([predictions_upper_with_start, predictions_lower_with_start[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        showlegend=True,
        name='Confidence Interval',
        hoverinfo='skip'
    ))
    
    # Upper bound - starts from last training point
    fig.add_trace(go.Scatter(
        x=prediction_indices,
        y=predictions_upper_with_start,
        mode='lines',
        name='Optimistic Bound',
        line=dict(color='#ff7f0e', width=1.5, dash='dot'),
        hovertemplate='<b>Month %{x}</b><br>Upper: %{y:.2f}<extra></extra>'
    ))
    
    # Lower bound - starts from last training point
    fig.add_trace(go.Scatter(
        x=prediction_indices,
        y=predictions_lower_with_start,
        mode='lines',
        name='Pessimistic Bound',
        line=dict(color='#ff7f0e', width=1.5, dash='dot'),
        hovertemplate='<b>Month %{x}</b><br>Lower: %{y:.2f}<extra></extra>'
    ))
    
    # Vertical line at split
    fig.add_vline(
        x=len(train_data) - 1,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Last Known Point",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Export Services Sales Prediction with Confidence Intervals',
        xaxis_title='Month',
        yaxis_title='Sales (MISK)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        template='plotly_white'
    )
    
    return fig

def main():
    # Title and description
    st.title("📊 Sales Prediction with Confidence Intervals")
    st.markdown("""
    This interactive tool predicts future export services sales with optimistic and pessimistic bounds.
    Adjust the settings in the sidebar to see how different parameters affect the predictions.
    """)
    
    # Load data
    sales_data = load_data()
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    st.sidebar.subheader("Data Split")
    split_percentage = st.sidebar.slider(
        f"Training Data Percentage",
        min_value=30,
        max_value=90,
        value=50,
        step=5,
        help="Percentage of data to use for training. The rest will be used for testing/prediction."
    )
    split_idx = int(len(sales_data) * split_percentage / 100)

    st.sidebar.markdown(f"""
    - **Training months**: {split_idx}
    - **Test months**: {len(sales_data) - split_idx}
    """)
    
    st.sidebar.subheader("Model Parameters")
    
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        options=[90, 95, 99],
        index=0,
        help="Confidence interval level. 90% means we expect 90% of actual values to fall within the bounds."
    )
    
    # Advanced settings (collapsible)
    with st.sidebar.expander("🔧 Advanced Settings"):
        st.markdown("""
        These settings control the model's behavior:
        - **Confidence Level**: Determines the percentile bounds (e.g., 90% = 5th to 95th percentile)
        """)
    
    # Split data
    train_data = sales_data[:split_idx]
    test_data = sales_data[split_idx:]
    
    # Calculate lookback to ensure we have enough training samples
    # We need at least 5 samples for the model to train properly with subsample=0.85
    # So lookback = len(train_data) - min_samples
    min_samples = 5
    lookback = max(6, len(train_data) - min_samples)  # At least 6 months, maximum all available
    
    # Generate predictions
    with st.spinner('Training models and generating predictions...'):
        results = generate_predictions(
            train_data, 
            test_data, 
            lookback, 
            confidence_level,
            100 - confidence_level  # upper percentile
        )

    # Display chart
    st.header("📊 Visualization")
    fig = create_plot(sales_data, split_idx, results)
    st.plotly_chart(fig, width="stretch")
    
    # Display sample predictions
    st.header("📋 Sample Predictions")
    
    sample_size = min(10, len(test_data))
    
    # Create a nice table
    st.markdown(f"**First {sample_size} test months:**")
    
    table_data = []
    for i in range(sample_size):
        in_bounds = "✓" if results['predictions_lower'][i] <= test_data[i] <= results['predictions_upper'][i] else "✗"
        table_data.append({
            "Month": i + 1,
            "Actual": f"{test_data[i]:.2f}",
            "Pessimistic": f"{results['predictions_lower'][i]:.2f}",
            "Median": f"{results['predictions_median'][i]:.2f}",
            "Optimistic": f"{results['predictions_upper'][i]:.2f}",
        })

    st.table(table_data)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 20px;'>"
        "Made with 🐍, 🧠, 🐳, and ❤️ by Team CircumAI during Hagstofan Datathon"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
