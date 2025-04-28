# main.py (streamlit_app.py)
import streamlit as st
from utils.data_loader import load_data

# === Page Configuration ===
st.set_page_config(
    page_title="Multi-Label Text Classification",
    layout="wide",
)

# === Initialize Session State ===
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'label_columns' not in st.session_state:
    st.session_state.label_columns = None

# === Load Dataset Once ===
if 'df' not in st.session_state:
    st.session_state.df = load_data()

# === App Title ===
st.title("📊 Bataknese Multi-Label Text Classification")

st.markdown("""
This app is designed to classify the sentiment of automotive-related texts across different aspects in **Batak Toba** language. These aspects include:

- Fuel (Positive, Neutral, Negative)
- Engine (Positive, Neutral, Negative)
- Service (Positive, Neutral, Negative)
- Parts (Positive, Neutral, Negative) 
- Price (Positive, Neutral, Negative) 
- Other (Positive, Neutral, Negative)

---
""")

# === Overview Section ===
st.header("Application Overview")
st.markdown("""
This project demonstrates a full pipeline for multi-label classification:

- **Dataset Explorer**: Visualize and understand the dataset.
- **Model Training**: Train and evaluate models (Random Forest, SVM) with various feature extraction techniques.
- **Text Prediction**: Input new sentences and predict associated labels across multiple aspects.

Use the sidebar to navigate between pages.

---
""")

# === Dataset Preview ===
st.header("Dataset Overview")

df = st.session_state.df
st.write(f"**Number of samples**: {df.shape[0]}")
st.write(f"**Number of features**: {df.shape[1]}")

st.dataframe(df.head(5))
