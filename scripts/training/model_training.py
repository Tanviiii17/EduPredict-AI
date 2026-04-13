"""
UPDATED Machine Learning Model Training Script
Now Includes:
✔ Mode-based Missing Value Imputation
✔ Feature Scaling
✔ Stratified K-Fold Cross Validation
✔ Hyperparameter Tuning (GridSearchCV)
✔ Outlier Handling (Rare Category Filtering)
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.impute import SimpleImputer
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = 'data/students_adaptability_level_online_education.csv'
MODEL_PATH = 'models/adaptability_model.pkl'
ENCODER_PATH = 'models/label_encoders.pkl'
METRICS_PATH = 'models/metrics.json'
CONFUSION_MATRIX_PATH = 'static/confusion_matrix.png'

FEATURE_COLUMNS = [
    'Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student',
    'Location', 'Load-shedding', 'Financial Condition', 'Internet Type',
    'Network Type', 'Class Duration', 'Self Lms', 'Device'
]

TARGET_COLUMN = 'Adaptivity Level'

# -------------------- NEW: MODE IMPUTATION --------------------
def handle_missing_values(df):
    logger.info("Handling missing values using mode imputation...")
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

# -------------------- NEW: RARE CATEGORY FILTER --------------------
def handle_rare_categories(df, threshold=0.01):
    logger.info("Handling rare categorical outliers...")
    df_clean = df.copy()
    for col in FEATURE_COLUMNS:
        freq = df_clean[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df_clean[col] = df_clean[col].replace(rare, 'Other')
    return df_clean

# -------------------- LOAD DATA --------------------
def load_and_preprocess_data():
    logger.info("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    df = handle_missing_values(df)
    df = handle_rare_categories(df)

    return df

# -------------------- ENCODING --------------------
def encode_features(df):
    logger.info("Encoding categorical features...")

    encoders = {}
    df_encoded = df.copy()

    columns_to_encode = FEATURE_COLUMNS + [TARGET_COLUMN]

    for column in columns_to_encode:
        if column in df.columns:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df[column].astype(str))
            encoders[column] = le

    return df_encoded, encoders

# -------------------- MAIN TRAINING --------------------
def train_model():
    logger.info("Starting Updated Training Pipeline...")

    df = load_and_preprocess_data()
    df_encoded, encoders = encode_features(df)

    X = df_encoded[FEATURE_COLUMNS]
    y = df_encoded[TARGET_COLUMN]

    # -------------------- NEW: FEATURE SCALING --------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------- NEW: HYPERPARAMETER TUNING --------------------
    param_grid = {
        'var_smoothing': np.logspace(-12, -6, num=7)
    }

    gnb = GaussianNB()
    grid = GridSearchCV(gnb, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    logger.info(f"Best Params: {grid.best_params_}")

    # -------------------- NEW: CROSS VALIDATION --------------------
    skf = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    target_names = encoders[TARGET_COLUMN].classes_
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    metrics = {
        'accuracy': round(accuracy * 100, 2),
        'cv_accuracy_mean': round(cv_scores.mean() * 100, 2),
        'cv_accuracy_std': round(cv_scores.std() * 100, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'best_params': grid.best_params_,
        'confusion_matrix': cm.tolist(),
        'class_labels': target_names.tolist(),
        'classification_report': report
    }

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    generate_visualizations(cm, target_names, accuracy)

    return metrics

# -------------------- VISUALIZATION --------------------
def generate_visualizations(cm, class_labels, accuracy):
    os.makedirs('static', exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

if __name__ == '__main__':
    train_model()
