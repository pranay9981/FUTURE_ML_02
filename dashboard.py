import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import load_data, get_metrics
from predict import predict

# Page Config
st.set_page_config(
    page_title="Support Ticket AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50; /* Green */
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
DATA_PATH = "all_tickets_processed_improved_v3.csv"
df = load_data(DATA_PATH)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80) 
    st.title("Support AI")
    st.markdown("---")
    menu = st.radio("Navigation", ["🏠 Home", "🔍 Classify Ticket", "📊 Analytics", "📂 Data Explorer", "🧠 Model Performance"])
    st.markdown("---")
    st.caption("AI-Powered Support System v1.1")

# --- HOME PAGE ---
if menu == "🏠 Home":
    st.title("🚀 Support Operations Dashboard")
    st.markdown("### Operational Overview")
    
    col1, col2, col3 = st.columns(3)
    
    total, top_cat = get_metrics(df)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Tickets</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Top Category</div>
            <div class="metric-value">{top_cat}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%); color: #333;">
            <div class="metric-title">Model Status</div>
            <div class="metric-value">Active 🟢</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Recent Ticket Stream")
    st.dataframe(df[['Document', 'Topic_group']].head(10), use_container_width=True)

# --- CLASSIFY PAGE ---
elif menu == "🔍 Classify Ticket":
    st.title("🤖 Live Validator")
    st.markdown("Input a support ticket to get an instant category and priority classification.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("ticket_form"):
            text_input = st.text_area("Ticket Description", height=200, placeholder="Describe the issue here...")
            submitted = st.form_submit_button("Analyze Ticket")
            
    with col2:
        st.markdown("### AI Predictions")
        if submitted and text_input:
            with st.spinner("Analyzing text patterns..."):
                try:
                    category, confidence, priority = predict(text_input)
                    
                    # Category Card
                    st.markdown(f"""
                    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #3498db; margin-bottom: 10px;">
                        <span style="color: #7f8c8d; font-size: 0.9em;">Category</span><br>
                        <span style="color: #2c3e50; font-size: 1.5em; font-weight: bold;">{category}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Priority Card (Dynamic Color)
                    priority_color = "#e74c3c" if priority == "High" else "#f39c12" if priority == "Medium" else "#27ae60"
                    st.markdown(f"""
                    <div style="background-color: #fff; padding: 15px; border-radius: 10px; border: 1px solid {priority_color}; margin-bottom: 10px;">
                        <span style="color: #7f8c8d; font-size: 0.9em;">Priority</span><br>
                        <span style="color: {priority_color}; font-size: 1.5em; font-weight: bold;">{priority}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric("Confidence", f"{confidence:.2%}")
                    
                except Exception as e:
                    st.error(f"Error predicting: {e}")
        else:
            st.info("Waiting for input...")

# --- ANALYTICS PAGE ---
elif menu == "📊 Analytics":
    st.title("📈 Performance Analytics")
    
    tab1, tab2 = st.tabs(["Category Insights", "Text Mining"])
    
    with tab1:
        st.subheader("Ticket Distribution")
        if not df.empty:
            fig = px.pie(df, names='Topic_group', title='Tickets by Category', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
            
            # Second chart: Bar chart
            fig2 = px.bar(df['Topic_group'].value_counts().reset_index(), x='Topic_group', y='count', color='Topic_group')
            st.plotly_chart(fig2, use_container_width=True)
            
    with tab2:
        st.subheader("Keyword Analysis")
        if not df.empty:
            text = " ".join(df['Document'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

# --- DATA PAGE ---
elif menu == "📂 Data Explorer":
    st.title("🗄️ Database View")
    
    # Advanced Filters
    col1, col2 = st.columns(2)
    with col1:
        categories = st.multiselect("Filter by Category", df['Topic_group'].unique())
    with col2:
        search_term = st.text_input("Search Text")
    
    filtered_df = df
    if categories:
        filtered_df = filtered_df[filtered_df['Topic_group'].isin(categories)]
    if search_term:
        filtered_df = filtered_df[filtered_df['Document'].str.contains(search_term, case=False, na=False)]
        
    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} records")

# --- MODEL PERFORMANCE PAGE ---
elif menu == "🧠 Model Performance":
    st.title("🧪 Model Evaluation")
    st.markdown("### Confusion Matrices")
    st.markdown("Visualizing how well the model distinguishes between different classes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("confusion_matrix_Category.png", caption="Category Classification Performance", use_container_width=True)
        
    with col2:
        st.image("confusion_matrix_Priority.png", caption="Priority Classification Performance", use_container_width=True)

