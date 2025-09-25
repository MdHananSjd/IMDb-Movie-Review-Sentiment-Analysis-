# README: IMDb Movie Review Sentiment Classification Dashboard

## Project Summary

This project delivers a high-performance, interactive Streamlit dashboard for binary sentiment classification (Positive/Negative) using the IMDb Movie Review Dataset. The solution introduces an **Ensemble Learning** architecture to maximize predictive performance. The entire application is packaged within a clean Streamlit interface, ensuring a professional and user-friendly demonstration.

---

## Technical Approach & Methodology

### 1. Data Preparation and Feature Engineering

* **Preprocessing:** Raw reviews were cleaned by removing HTML tags (`<br /><br />`) and normalizing text to lowercase, preparing the data for model ingestion.
* **Vectorization (Feature Extraction):** **CountVectorizer** was used to convert the raw text into numerical feature vectors.
* **Dimensionality Control:** The model was limited to a vocabulary size of **2,000 features** for efficiency.
* **Data Split:** The dataset was partitioned into an **80% training set** and a **20% test set** for unbiased evaluation.

### 2. Model Architecture

The application demonstrates four classification models, accessible via dedicated tabs.

| Model | Classification Method | Key Optimization for Speed |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear Classification | Uses the highly efficient **`liblinear`** solver. |
| **Random Forest** | Tree-based Ensemble | Optimized for speed by limiting the number of decision trees (**`n_estimators=50`**). |
| **Support Vector Machine (SVM)** | Linear Classification | Uses **`LinearSVC`** instead of the slower `SVC(kernel='linear')` to ensure fast training on the large dataset. |
| **Voting Ensemble (Final Model)** | Hard Voting | Combines the predictions of LR, RF, and SVM using a **majority vote** rule (**`hard` voting**). |

---

## Evaluation and Interactive Features

### 3. Results and Visualization

The dashboard provides a complete picture of each model's performance:

* **Multiple Metrics:** We compare models using **Accuracy, Precision, Recall, and F1-Score.**
* **Confusion Matrix Plot:** Visual plots (with a **Blue** colormap) showing the count of True Positives, True Negatives, False Positives, and False Negatives.
* **Learning Curve Plot:** Plots included for each model to visualize training performance versus data size, confirming that the models are **not severely overfitting**.

### 4. Application Architecture and User Experience

* **Immediate Availability:** The entire application uses **Streamlit Caching** (`@st.cache_data` and `@st.cache_resource`) to load data and train the models only **once** upon the first run, guaranteeing instant results for all subsequent users or interactions.
* **Modular Interface:** The Streamlit application utilizes separate tabs to clearly organize the training and results for each model.
* **Interactive Predictor:** A final tab allows any user to input a review and instantly receive a sentiment prediction from the best-performing model (the Voting Ensemble).
