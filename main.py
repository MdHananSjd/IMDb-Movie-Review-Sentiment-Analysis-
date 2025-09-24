import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import sys

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv('IMDB Dataset.csv')
    except FileNotFoundError:
        st.error("Dataset not found.")
        st.stop()
    df['review'] = df['review'].apply(lambda x: re.sub(r'<br /><br />', ' ', x))
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@st.cache_resource
def get_vectorizer(X_train):
    vectorizer = CountVectorizer(stop_words='english', max_features=2000)
    vectorizer.fit(X_train)
    return vectorizer

def train_individual_model(X_train_vec, y_train, model_type, X_test_vec, y_test):
    if model_type == 'Logistic Regression':
        model = LogisticRegression(solver='liblinear', random_state=42)
    elif model_type == 'SVM':
        model = SVC(kernel='linear', C=1.0, random_state=42)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    with st.spinner(f"Training {model_type}..."):
        model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"**{model_type} trained successfully!** 🎉 Accuracy: **{accuracy:.4f}**")
    return model

def train_ensemble_model(X_train_vec, y_train, X_test_vec, y_test):
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', st.session_state['lr_model']),
            ('svm', st.session_state['svm_model']),
            ('rf', st.session_state['rf_model'])
        ],
        voting='hard'
    )
    with st.spinner("Training Voting Ensemble..."):
        ensemble_model.fit(X_train_vec, y_train)
    y_pred = ensemble_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"**Voting Ensemble trained successfully!** 🎉 Accuracy: **{accuracy:.4f}**")
    return ensemble_model

@st.cache_data
def evaluate_model(_model, _X_test_vec, y_test):
    y_pred = _model.predict(_X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return y_pred, accuracy, precision, recall, f1

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Greens, ax=ax, colorbar=False)
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)

def display_model_results(model, model_name, X_test, y_test, vectorizer):
    X_test_vec = vectorizer.transform(X_test)
    y_pred, accuracy, precision, recall, f1 = evaluate_model(model, X_test_vec, y_test)
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
    }).set_index('Metric')
    st.dataframe(metrics_df)
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_test, y_pred, f"{model_name} Confusion Matrix")

def main():
    st.set_page_config(page_title='Movie Review', page_icon="👾", layout='wide')
    st.title("IMDb Movie Review Sentiment Analysis")
    if 'data_loaded' not in st.session_state:
        st.session_state['X_train'], st.session_state['X_test'], st.session_state['y_train'], st.session_state['y_test'] = load_and_preprocess_data()
        st.session_state['vectorizer'] = get_vectorizer(st.session_state['X_train'])
        st.session_state['data_loaded'] = True
    st.header("Data Distribution")
    sentiment_counts = st.session_state['y_train'].value_counts()
    st.write("This chart shows the distribution of positive and negative reviews in the training data.")
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
    st.pyplot(fig)
    plt.close(fig)
    tab_lr, tab_rf, tab_svm, tab_ensemble, tab_predictor = st.tabs(["Logistic Regression", "Random Forest", "SVM", "Voting Ensemble", "Custom Predictor"])
    with tab_lr:
        st.header("Logistic Regression Model")
        if st.button("Train Logistic Regression", key='lr_button'):
            st.session_state['lr_model'] = train_individual_model(st.session_state['vectorizer'].transform(st.session_state['X_train']), st.session_state['y_train'], 'Logistic Regression', st.session_state['vectorizer'].transform(st.session_state['X_test']), st.session_state['y_test'])
        if 'lr_model' in st.session_state:
            display_model_results(st.session_state['lr_model'], "Logistic Regression", st.session_state['X_test'], st.session_state['y_test'], st.session_state['vectorizer'])
        else:
            st.info("Click the button above to train the model.")
    with tab_rf:
        st.header("Random Forest Model")
        if st.button("Train Random Forest", key='rf_button'):
            st.session_state['rf_model'] = train_individual_model(st.session_state['vectorizer'].transform(st.session_state['X_train']), st.session_state['y_train'], 'Random Forest', st.session_state['vectorizer'].transform(st.session_state['X_test']), st.session_state['y_test'])
        if 'rf_model' in st.session_state:
            display_model_results(st.session_state['rf_model'], "Random Forest", st.session_state['X_test'], st.session_state['y_test'], st.session_state['vectorizer'])
        else:
            st.info("Click the button above to train the model.")
    with tab_svm:
        st.header("Support Vector Machine (SVM) Model")
        if st.button("Train SVM", key='svm_button'):
            st.session_state['svm_model'] = train_individual_model(st.session_state['vectorizer'].transform(st.session_state['X_train']), st.session_state['y_train'], 'SVM', st.session_state['vectorizer'].transform(st.session_state['X_test']), st.session_state['y_test'])
        if 'svm_model' in st.session_state:
            display_model_results(st.session_state['svm_model'], "SVM", st.session_state['X_test'], st.session_state['y_test'], st.session_state['vectorizer'])
        else:
            st.info("Click the button above to train the model.")
    with tab_ensemble:
        st.header("Voting Ensemble Model")
        all_models_trained = 'lr_model' in st.session_state and 'svm_model' in st.session_state and 'rf_model' in st.session_state
        if not all_models_trained:
            st.warning("Please train the Logistic Regression, Random Forest, and SVM models first to enable the ensemble.")
        else:
            if st.button("Train Voting Ensemble", key='ensemble_button'):
                st.session_state['ensemble_model'] = train_ensemble_model(st.session_state['vectorizer'].transform(st.session_state['X_train']), st.session_state['y_train'], st.session_state['vectorizer'].transform(st.session_state['X_test']), st.session_state['y_test'])
            if 'ensemble_model' in st.session_state:
                display_model_results(st.session_state['ensemble_model'], "Voting Ensemble", st.session_state['X_test'], st.session_state['y_test'], st.session_state['vectorizer'])
            else:
                st.info("Click the button above to train the ensemble model.")
    with tab_predictor:
        st.header("Custom Review Predictor")
        if 'ensemble_model' not in st.session_state:
            st.warning("Please train the Voting Ensemble model first (in its tab) to use the predictor.")
        else:
            user_review = st.text_area("Enter a movie review to predict its sentiment:", "")
            if st.button("Predict Sentiment"):
                if user_review:
                    clean_review = re.sub(r'<br /><br />', ' ', user_review)
                    clean_review = re.sub(r'[^a-zA-Z\s]', '', clean_review).lower()
                    user_review_vec = st.session_state['vectorizer'].transform([clean_review])
                    prediction = st.session_state['ensemble_model'].predict(user_review_vec)
                    sentiment = "Positive" if prediction[0] == 1 else "Negative"
                    st.success(f"The model predicts the sentiment is: **{sentiment}**")
                else:
                    st.warning("Please enter a review to get a prediction.")

if __name__ == '__main__':
    main()