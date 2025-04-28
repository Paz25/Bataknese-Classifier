import streamlit as st
import matplotlib.pyplot as plt
from utils.visualization import plot_label_distribution

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("🗂️ Dataset Overview")

df = st.session_state.df

st.subheader("Dataset Overview")
st.write(f"Number of samples: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1]}")

st.subheader("Sample Data")
st.dataframe(df.head())

st.subheader("Label Distribution")
cols = st.columns(3)

for i, column in enumerate(['fuel', 'machine', 'service', 'part', 'price', 'others']):
    with cols[i % 3]:
        fig = plot_label_distribution(df, column)
        st.pyplot(fig)

st.subheader("Sample Sentences by Sentiment")

sentiment_to_explore = st.selectbox(
    "Choose sentiment to explore:",
    ['fuel', 'machine', 'service', 'part', 'price', 'others']
)

st.write(f"### {sentiment_to_explore.capitalize()} Sentiment Examples")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("#### Negative")
    for idx, row in df[df[sentiment_to_explore] == 'negative'].head(3).iterrows():
        with st.expander(f"*{row['translated']}*"):
            st.write(f"""
            **Original Sentence:**  
            {row['sentence']}
            """)

with col2:
    st.write("#### Neutral")
    for idx, row in df[df[sentiment_to_explore] == 'neutral'].head(3).iterrows():
        with st.expander(f"*{row['translated']}*"):
            st.write(f"""
            **Original Sentence:**  
            {row['sentence']}
            """)

with col3:
    st.write("#### Positive")
    for idx, row in df[df[sentiment_to_explore] == 'positive'].head(3).iterrows():
        with st.expander(f"*{row['translated']}*"):
            st.write(f"""
            **Original Sentence:**  
            {row['sentence']}
            """)
