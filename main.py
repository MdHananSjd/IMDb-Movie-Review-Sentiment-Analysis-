import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import sys

# --- 1. Functions for Data Loading and Preprocessing ---

@st.cache_data
def load_and_preprocess_data():
    """
    Loads the IMDb dataset and performs all preprocessing steps.
    """
    try:
        df = pd.read_csv('IMDB Dataset.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'IMDB Dataset.csv' is in the same directory.")
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
    """Initializes and fits the CountVectorizer."""
    vectorizer = CountVectorizer(stop_words='english', max_features=2000)
    vectorizer.fit(X_train)
    return vectorizer

# --- 2. Functions for Model Training and Evaluation ---

def train_individual_model(X_train_vec, y_train, model_type, X_test_vec, y_test):
    """Trains a single model based on the model_type and returns the model and its accuracy."""
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
    """Trains the Voting Ensemble model and returns the model and its accuracy."""
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
    """Evaluates a model's performance on the test data."""
    y_pred = _model.predict(_X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return y_pred, accuracy, precision, recall, f1

# --- 3. Visualization Function ---

def plot_confusion_matrix(y_true, y_pred, title):
    """Plots a confusion matrix for model predictions."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3)) # Made smaller
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Greens, ax=ax, colorbar=False)
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)

def plot_learning_curve(model, X_train_vec, y_train, title):
    """Plots the learning curve for a model."""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_vec, y_train, cv=2, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(4, 3)) # Made smaller
    ax.grid()
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    st.pyplot(fig)
    plt.close(fig)

def display_model_results(model, model_name, X_test, y_test, vectorizer, X_train):
    """A helper function to display model metrics and plots."""
    X_test_vec = vectorizer.transform(X_test)
    y_pred, accuracy, precision, recall, f1 = evaluate_model(model, X_test_vec, y_test)
    
    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
    }).set_index('Metric')
    st.dataframe(metrics_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(y_test, y_pred, f"{model_name} Confusion Matrix")
    with col2:
        st.subheader("Learning Curve")
        plot_learning_curve(model, vectorizer.transform(X_train), y_test, f"{model_name} Learning Curve")


# --- 4. Streamlit Application UI ---

def main():
    st.set_page_config(page_title='Movie Review', page_icon="👾", layout='wide')
    st.title("IMDb Movie Review Sentiment Analysis")

    # Load and preprocess data once at the start
    if 'data_loaded' not in st.session_state:
        st.session_state['X_train'], st.session_state['X_test'], st.session_state['y_train'], st.session_state['y_test'] = load_and_preprocess_data()
        st.session_state['vectorizer'] = get_vectorizer(st.session_state['X_train'])
        st.session_state['data_loaded'] = True
    
    # Always-visible data visualization in the sidebar
    with st.sidebar:
        st.header("Data Distribution")
        sentiment_counts = st.session_state['y_train'].value_counts()
        st.write("This chart shows the distribution of positive and negative reviews in the training data.")
        fig, ax = plt.subplots(figsize=(5, 4))
        sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    # Tabs for each model and the custom predictor
    tab_lr, tab_rf, tab_svm, tab_ensemble, tab_predictor = st.tabs(["Logistic Regression", "Random Forest", "SVM", "Voting Ensemble", "Custom Predictor"])

    # --- Tab 1: Logistic Regression ---
    with tab_lr:
        st.header("Logistic Regression Model")
        if st.button("Train Logistic Regression", key='lr_button'):
            st.session_state['lr_model'] = train_individual_model(st.session_state['vectorizer'].transform(st.session_state['X_train']), st.session_state['y_train'], 'Logistic Regression', st.session_state['vectorizer'].transform(st.session_state['X_test']), st.session_state['y_test'])
        
        if 'lr_model' in st.session_state:
            display_model_results(st.session_state['lr_model'], "Logistic Regression", st.session_state['X_test'], st.session_state['y_test'], st.session_state['vectorizer'], st.session_state['X_train'])
        else:
            st.info("Click the button above to train the model.")

    # --- Tab 2: Random Forest ---
    with tab_rf:
        st.header("Random Forest Model")
        if st.button("Train Random Forest", key='rf_button'):
            st.session_state['rf_model'] = train_individual_model(st.session_state['vectorizer'].transform(st.session_state['X_train']), st.session_state['y_train'], 'Random Forest', st.session_state['vectorizer'].transform(st.session_state['X_test']), st.session_state['y_test'])
        
        if 'rf_model' in st.session_state:
            display_model_results(st.session_state['rf_model'], "Random Forest", st.session_state['X_test'], st.session_state['y_test'], st.session_state['vectorizer'], st.session_state['X_train'])
        else:
            st.info("Click the button above to train the model.")

    # --- Tab 3: SVM ---
    with tab_svm:
        st.header("Support Vector Machine (SVM) Model")
        if st.button("Train SVM", key='svm_button'):
            st.session_state['svm_model'] = train_individual_model(st.session_state['vectorizer'].transform(st.session_state['X_train']), st.session_state['y_train'], 'SVM', st.session_state['vectorizer'].transform(st.session_state['X_test']), st.session_state['y_test'])
        
        if 'svm_model' in st.session_state:
            display_model_results(st.session_state['svm_model'], "SVM", st.session_state['X_test'], st.session_state['y_test'], st.session_state['vectorizer'], st.session_state['X_train'])
        else:
            st.info("Click the button above to train the model.")
            
    # --- Tab 4: Voting Ensemble ---
    with tab_ensemble:
        st.header("Voting Ensemble Model")
        
        all_models_trained = 'lr_model' in st.session_state and 'svm_model' in st.session_state and 'rf_model' in st.session_state
        
        if not all_models_trained:
            st.warning("Please train the Logistic Regression, Random Forest, and SVM models first to enable the ensemble.")
        else:
            if st.button("Train Voting Ensemble", key='ensemble_button'):
                st.session_state['ensemble_model'] = train_ensemble_model(st.session_state['vectorizer'].transform(st.session_state['X_train']), st.session_state['y_train'], st.session_state['vectorizer'].transform(st.session_state['X_test']), st.session_state['y_test'])
            
            if 'ensemble_model' in st.session_state:
                display_model_results(st.session_state['ensemble_model'], "Voting Ensemble", st.session_state['X_test'], st.session_state['y_test'], st.session_state['vectorizer'], st.session_state['X_train'])
            else:
                st.info("Click the button above to train the ensemble model.")

    # --- Tab 5: Custom Predictor ---
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