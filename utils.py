import streamlit as st
import pandas as pd
from dataset import TicketDataset # Reusing existing logic if needed
from predict import predict # Reusing existing prediction logic

@st.cache_data
def load_data(file_path):
    """
    Loads and caches the ticket data.
    """
    try:
        df = pd.read_csv(file_path)
        # Basic cleaning for display
        df = df.dropna(subset=['Document', 'Topic_group'])
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

def get_metrics(df):
    """
    Calculates key metrics for the dashboard.
    """
    total_tickets = len(df)
    top_category = df['Topic_group'].mode()[0] if not df.empty else "N/A"
    return total_tickets, top_category
