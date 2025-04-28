# pages/Model_Training.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils.feature_engineering import get_average_word2vec, train_word2vec_model
from utils.visualization import plot_multilabel_confusion_matrix
from utils.create_multilabel import create_multilabel
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain

st.set_page_config(page_title="Model Training", layout="wide")
st.title("🏋️ Model Training")

df = st.session_state.df

st.subheader("Feature Engineering")
feature_option = st.selectbox(
    "Select feature representation:", ["BoW", "Bi-gram Vector (BV)", "TF-IDF", "Word2Vec"]
)

st.subheader("Model Selection")
model_option = st.selectbox(
    "Select model:", ["Random Forest", "SVM"]
)

if feature_option != "Word2Vec":
    st.subheader("Text Vectorization Parameters")
    max_features = st.slider(
        "Max Features (for BoW/Bi-gram/TF-IDF)", min_value=1000, max_value=10000, value=5000, step=1000
    )

model_params = {}
if model_option == "Random Forest":
    model_params['n_estimators'] = st.slider(
        "Number of Trees", min_value=10, max_value=300, value=100, step=10)
elif model_option == "SVM":
    model_params['C'] = st.slider(
        "Regularization Parameter (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)

test_size = st.slider("Test Size (%)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

st.subheader("Problem Transformation")
strategy = st.selectbox("Choose problem transformator:", ["Binary Relevance", "Classifier Chain"])

if st.button("Train Model"):
    st.info("Training in progress... Please wait ⏳")

    X = df['translated']
    y = create_multilabel(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if feature_option == "BoW":
        vectorizer = CountVectorizer(max_df=1.0, min_df=0.0)
    elif feature_option == "Bi-gram Vector (BV)":
        vectorizer = CountVectorizer(ngram_range=(2,2))
    elif feature_option == "TF-IDF":
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), norm='l2', use_idf=True, smooth_idf=True)
    elif feature_option == "Word2Vec":
        train_tokens = [text.split() for text in X_train]
        w2v_model = train_word2vec_model(train_tokens, vector_size=1000)
    else:
        st.error("Invalid feature selection")
    
    if feature_option != "Word2Vec":
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
    else:
        X_train_vec = get_average_word2vec([text.split() for text in X_train], w2v_model, 1000)
        X_test_vec = get_average_word2vec([text.split() for text in X_test], w2v_model, 1000)

    if model_option == "Random Forest":
        base_model = RandomForestClassifier(**model_params, class_weight="balanced", random_state=42)
    elif model_option == "SVM":
        base_model = SVC(kernel='linear', probability=True, class_weight='balanced', **model_params)
    else:
        st.error("Invalid model selection")

    if strategy == "Binary Relevance":
        model = BinaryRelevance(base_model)
    else:
        model = ClassifierChain(base_model)

    model.fit(X_train_vec, y_train)

    st.session_state.trained_model = model
    st.session_state.vectorizer = vectorizer if feature_option != "Word2Vec" else w2v_model
    st.session_state.model_name = model_option
    st.session_state.label_columns = y.columns.tolist()
    st.session_state.feature_engineering = feature_option
    st.session_state.multi_label_strategy = strategy 

    y_pred = model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model trained successfully! Test Accuracy: {test_accuracy:.4f}")

    st.subheader("Sample Predictions")
    comparison_df = pd.DataFrame({
        "Text": X_test.reset_index(drop=True),
        "True Labels": y_test.reset_index(drop=True).values.tolist(),
        "Predicted Labels": y_pred.toarray().tolist()
    })
    st.dataframe(comparison_df.head(10))

    st.subheader("Confusion Matrices for Each Label")
    mcm = multilabel_confusion_matrix(y_test, y_pred)

    for i in range(0, len(y.columns), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(y.columns):
                with cols[j]:
                    fig = plot_multilabel_confusion_matrix(mcm[i+j], y.columns[i+j])
                    st.pyplot(fig)
                    plt.close(fig)