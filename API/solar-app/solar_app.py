import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Solar Power Prediction", layout="wide")

# Constants
API_URL = "http://solar-prediction-api-service:5000/predict"

def format_datetime(dt):
    """Format datetime to match API expectations"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_predictions(start_datetime, n_hours):
    params = {
        "datetime": format_datetime(start_datetime),
        "n_hours": n_hours
    }
    
    try:
        response = requests.get(API_URL, params=params)
        if response.status_code == 404:
            st.warning("No predictions available for the specified time range.")
            return None
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return None

def main():
    # Set the font family
    st.markdown("""
    <style>
    body {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    # Set the color palette
    colors = ['#0077B6', '#F4A460', '#008B8B']

    st.sidebar.header("Input Parameters")
    
    # Date and time selection
    start_date = st.sidebar.date_input("Select Date", datetime.now())
    start_time = st.sidebar.time_input("Select Time", datetime.now())
    
    # Combine date and time
    start_datetime = datetime.combine(start_date, start_time)
    
    # Number of hours to predict
    n_hours = st.sidebar.slider("Hours to Predict", min_value=1, max_value=72, value=24)
    
    if st.sidebar.button("Get Predictions"):
        with st.spinner("Fetching predictions..."):
            predictions = get_predictions(start_datetime, n_hours)
        
        if predictions:
            # Convert predictions to DataFrame
            df = pd.DataFrame(predictions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Predicted Solar Power Over Time")
                fig1 = px.line(df, x='timestamp', y='prediction', color_discrete_sequence=colors)
                fig1.update_layout(xaxis_title="Time", yaxis_title="Predicted Power (kW)", width=600, height=400, font_family='Roboto')
                st.plotly_chart(fig1)
            
            with col2:
                # Group data by time intervals
                df['time_interval'] = pd.cut(df['timestamp'].dt.hour, bins=[-1, 8, 16, 24], labels=['Night', 'Day', 'Evening'])
                
                st.subheader("Predicted Power Distribution")
                fig2 = px.pie(df, names='time_interval', values='prediction', color_discrete_sequence=colors)
                fig2.update_layout(width=600, height=400, font_family='Roboto', 
                                  font=dict(size=16))
                st.plotly_chart(fig2)
            
            # Prediction Data
            st.subheader("Prediction Data")
            st.dataframe(df)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            summary_stats = {
                "Average Prediction": f"{df['prediction'].mean():.4f} kW",
                "Maximum Prediction": f"{df['prediction'].max():.4f} kW",
                "Minimum Prediction": f"{df['prediction'].min():.4f} kW"
            }
            st.json(summary_stats)
            
            # Download button for CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"solar_predictions_{start_datetime.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()