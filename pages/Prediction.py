# pages/Prediction.py

import streamlit as st
from utils.preprocessing import preprocess_text
from utils.feature_engineering import get_average_word2vec
import numpy as np

st.set_page_config(page_title="Prediction", layout="wide")
st.title("🔮 Make Predictions")

df = st.session_state.df

user_input = st.text_area("Input your review (anything you want):", "Mesin Avanza na hebat, alai na manganhon gas")

if st.session_state.trained_model is None:
    st.warning("No trained model found. Please train a model in the 'Model Training' page first.")
else:
    st.subheader("🛠 Model Settings Used")
    st.markdown(f"- **Feature Representation:** {st.session_state.feature_engineering}")
    st.markdown(f"- **Model:** {st.session_state.model_name}")
    st.markdown(f"- **Problem Transformator:** {st.session_state.multi_label_strategy}")


if st.button("Predict"):
    st.info("Predicting...")

    preprocessed_input = preprocess_text(user_input)

    model = st.session_state.trained_model
    vectorizer = st.session_state.vectorizer
    label_columns = st.session_state.label_columns
    feature_engineering = st.session_state.feature_engineering

    if feature_engineering == "Word2Vec":
        tokens = preprocessed_input.split()
        input_vector = get_average_word2vec([tokens], vectorizer, 1000)
    else:
        input_vector = vectorizer.transform([preprocessed_input])

    prediction = model.predict(input_vector)

    st.success("Prediction complete!")

    st.subheader("Input Text")
    st.write(user_input)

    st.subheader("Preprocessed Text")
    st.write(preprocessed_input)

    st.subheader("Predicted Labels")
    results = []
    for i, label in enumerate(label_columns):
        if prediction.toarray()[0, i] == 1:
            results.append(label)

    if results:
        st.subheader("Prediction Summary")

        fuel_preds = [col for col in results if col.startswith('fuel_')]
        machine_preds = [col for col in results if col.startswith('machine_')]
        service_preds = [col for col in results if col.startswith('service_')]
        price_preds = [col for col in results if col.startswith('price_')]
        part_preds = [col for col in results if col.startswith('part_')]
        others_preds = [col for col in results if col.startswith('others_')]

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        with col1:
            st.write("**Fuel Sentiment:**")
            if fuel_preds:
                for pred in fuel_preds:
                    label = pred.replace('fuel_', '')
                    if label == 'positive':
                        color = 'green'
                    elif label == 'negative':
                        color = 'red'
                    elif label == 'neutral':
                        color = 'blue'
                    else:
                        color = 'white'
                    st.markdown(f"<span style='color: {color};'>- {label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: blue;'>- neutral</span>", unsafe_allow_html=True)


        with col2:
            st.write("**Machine Sentiment:**")
            if machine_preds:
                for pred in machine_preds:
                    label = pred.replace('machine_', '')
                    if label == 'positive':
                        color = 'green'
                    elif label == 'negative':
                        color = 'red'
                    elif label == 'neutral':
                        color = 'blue'
                    else:
                        color = 'white'
                    st.markdown(f"<span style='color: {color};'>- {label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: blue;'>- neutral</span>", unsafe_allow_html=True)
        

        with col3:
            st.write("**Service Sentiment:**")
            if service_preds:
                for pred in service_preds:
                    label = pred.replace('service_', '')
                    if label == 'positive':
                        color = 'green'
                    elif label == 'negative':
                        color = 'red'
                    elif label == 'neutral':
                        color = 'blue'
                    else:
                        color = 'white'
                    st.markdown(f"<span style='color: {color};'>- {label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: blue;'>- neutral</span>", unsafe_allow_html=True)

        with col4:
            st.write("**Price Sentiment:**")
            if price_preds:
                for pred in price_preds:
                    label = pred.replace('price_', '')
                    if label == 'positive':
                        color = 'green'
                    elif label == 'negative':
                        color = 'red'
                    elif label == 'neutral':
                        color = 'blue'
                    else:
                        color = 'white'
                    st.markdown(f"<span style='color: {color};'>- {label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: blue;'>- neutral</span>", unsafe_allow_html=True)

        with col5:
            st.write("**Part Sentiment:**")
            if part_preds:
                for pred in part_preds:
                    label = pred.replace('part_', '')
                    if label == 'positive':
                        color = 'green'
                    elif label == 'negative':
                        color = 'red'
                    elif label == 'neutral':
                        color = 'blue'
                    else:
                        color = 'white'
                    st.markdown(f"<span style='color: {color};'>- {label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: blue;'>- neutral</span>", unsafe_allow_html=True)

        with col6:
            st.write("**Others Sentiment:**")
            if others_preds:
                for pred in others_preds:
                    label = pred.replace('others_', '')
                    if label == 'positive':
                        color = 'green'
                    elif label == 'negative':
                        color = 'red'
                    elif label == 'neutral':
                        color = 'blue'
                    else:
                        color = 'white'
                    st.markdown(f"<span style='color: {color};'>- {label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: blue;'>- neutral</span>", unsafe_allow_html=True)
            
    else:
        st.warning("No labels predicted.")
