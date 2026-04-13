"""
Advanced evaluation script for Student Adaptability project.
Includes:
- Consistent preprocessing (mode imputation + rare category handling + label encoding + scaling)
- Train/test split with stratification
- Optimized GNB (same idea as training.py)
- Baseline GNB for comparison
- Repeated stratified 5x10 CV
- Multi-class ROC curves + AUC
- Confusion matrix + macro/weighted F1 + balanced accuracy
- McNemar test (fixed: same test set for both models)
- Hardware configuration + training / inference timings
- Learning curves
- ANN (MLP) deep-learning baseline
- Optional preprocessing ablation
- Optional performance stratified by institution type
"""

import os
import time
import json
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import permutation_importance
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.preprocessing import label_binarize
import psutil

# -------------------- PATHS & CONSTANTS --------------------

DATASET_PATH = "data/students_adaptability_level_online_education.csv"

FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Education Level",
    "Institution Type",
    "IT Student",
    "Location",
    "Load-shedding",
    "Financial Condition",
    "Internet Type",
    "Network Type",
    "Class Duration",
    "Self Lms",
    "Device",
]

TARGET_COLUMN = "Adaptivity Level"

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# If you already ran training.py and stored best var_smoothing in metrics.json,
# this helper will try to read it; otherwise it falls back to a fixed value.
METRICS_JSON = "models/metrics.json"
DEFAULT_VAR_SMOOTHING = 0.01519911082952933


# -------------------- PREPROCESSING HELPERS --------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mode-based imputation on all columns."""
    imputer = SimpleImputer(strategy="most_frequent")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed


def handle_rare_categories(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Replace categories with frequency < threshold by 'Other'."""
    df_clean = df.copy()
    for col in FEATURE_COLUMNS:
        freq = df_clean[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        if len(rare) > 0:
            df_clean[col] = df_clean[col].replace(rare, "Other")
    return df_clean


def encode_features(df: pd.DataFrame):
    """Label-encode all feature columns and target."""
    encoders = {}
    df_encoded = df.copy()
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders


def load_and_prepare():
    """Full preprocessing pipeline returning encoded data and encoders."""
    df = pd.read_csv(DATASET_PATH)
    df = handle_missing_values(df)
    df = handle_rare_categories(df)
    df_encoded, encoders = encode_features(df)

    X = df_encoded[FEATURE_COLUMNS]
    y = df_encoded[TARGET_COLUMN]
    institution_raw = df["Institution Type"].values  # for stratified evaluation later

    return df, df_encoded, X, y, encoders, institution_raw


def get_best_var_smoothing():
    if os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON, "r") as f:
                metrics = json.load(f)
            return float(metrics.get("best_params", {}).get("var_smoothing", DEFAULT_VAR_SMOOTHING))
        except Exception:
            return DEFAULT_VAR_SMOOTHING
    return DEFAULT_VAR_SMOOTHING


# -------------------- CORE EVALUATION --------------------

def main():
    # ===== 0. Data & split (single split used for ALL models) =====
    df, df_encoded, X, y, encoders, institution_raw = load_and_prepare()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        inst_train,
        inst_test,
    ) = train_test_split(
        X,
        y,
        institution_raw,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Scale *only* on train, then apply to test for optimized models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===== 1. Baseline vs optimized GNB =====
    baseline_gnb = GaussianNB()
    baseline_gnb.fit(X_train, y_train)
    y_pred_base = baseline_gnb.predict(X_test)

    var_smoothing = get_best_var_smoothing()
    opt_gnb = GaussianNB(var_smoothing=var_smoothing)
    opt_gnb.fit(X_train_scaled, y_train)
    y_pred_opt = opt_gnb.predict(X_test_scaled)

    print(f"\nBest var_smoothing used for optimized GNB: {var_smoothing}")

    # Confusion matrix & basic metrics for optimized model
    cm = confusion_matrix(y_test, y_pred_opt)
    acc = accuracy_score(y_test, y_pred_opt)
    bal_acc = balanced_accuracy_score(y_test, y_pred_opt)
    macro_f1 = f1_score(y_test, y_pred_opt, average="macro")
    weighted_f1 = f1_score(y_test, y_pred_opt, average="weighted")

    target_names = encoders[TARGET_COLUMN].classes_
    clf_report = classification_report(y_test, y_pred_opt, target_names=target_names)
    print("\nOptimized GNB classification report:\n")
    print(clf_report)

    print(f"Accuracy:       {acc:.4f}")
    print(f"Balanced acc.:  {bal_acc:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")

    # Save confusion matrix figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix (Optimized GNB)\nAccuracy: {acc:.2%}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "confusion_matrix_gnb.png"))
    plt.close()

    # ===== 2. Repeated stratified 5x10 CV (optimized model) =====
    print("\nRepeated Stratified 5x10 CV for optimized GNB:")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    # Use scaled full X for CV
    X_full_scaled = scaler.fit_transform(X)  # fit on all data just for CV stability
    cv_scores = cross_val_score(opt_gnb, X_full_scaled, y, cv=rskf, n_jobs=-1)
    print("Mean CV accuracy:", cv_scores.mean())
    print("Std CV accuracy :", cv_scores.std())

    # ===== 3. Multi-class ROC curves (OvR on optimized model) =====
    y_bin = label_binarize(y_test, classes=np.unique(y))
    ovr_clf = OneVsRestClassifier(opt_gnb)
    y_score = ovr_clf.fit(X_train_scaled, y_train).predict_proba(X_test_scaled)

    plt.figure()
    auc_values = []
    for i, cls_name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        plt.plot(fpr, tpr, label=f"{cls_name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve (Optimized GNB)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "multiclass_roc_gnb.png"))
    plt.close()

    print("\nClass-wise AUCs:", dict(zip(target_names, auc_values)))
    print("Macro-avg AUC:", np.mean(auc_values))

    # ===== 4. McNemar test (fixed, same test set) =====
    table = [[0, 0], [0, 0]]
    for i in range(len(y_test)):
        base_correct = (y_pred_base[i] == y_test.iloc[i])
        opt_correct = (y_pred_opt[i] == y_test.iloc[i])
        if base_correct and opt_correct:
            table[0][0] += 1
        elif base_correct and not opt_correct:
            table[0][1] += 1
        elif (not base_correct) and opt_correct:
            table[1][0] += 1
        else:
            table[1][1] += 1

    result = mcnemar(table, exact=True)
    print("\nMcNemar contingency table [[both correct, base only], [opt only, both wrong]]:")
    print(table)
    print("McNemar p-value:", result.pvalue)

    # ===== 5. Hardware & runtime measurements =====
    print("\nHardware configuration:")
    print("CPU:", platform.processor())
    print("RAM (GB):", round(psutil.virtual_memory().total / (1024 ** 3), 2))
    print("OS:", platform.system(), platform.release())
    print("Python:", sys.version.split()[0])

    # Training time optimized GNB
    start = time.perf_counter()
    opt_tmp = GaussianNB(var_smoothing=var_smoothing)
    opt_tmp.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - start
    print(f"\nOptimized GNB training time (train set): {train_time:.6f} s")

    # Average single-sample inference time (optimized GNB)
    n_reps = 1000
    start = time.perf_counter()
    for _ in range(n_reps):
        opt_tmp.predict(X_test_scaled[:1])
    single_time = (time.perf_counter() - start) / n_reps
    print(f"Average single-sample inference time: {single_time * 1e3:.4f} ms")

    # Batch inference time (full test set)
    start = time.perf_counter()
    opt_tmp.predict(X_test_scaled)
    batch_time = time.perf_counter() - start
    print(f"Batch inference time on {len(X_test_scaled)} samples: {batch_time:.6f} s")

    # ===== 6. Learning curve =====
    print("\nComputing learning curves (this may take a bit)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        GaussianNB(var_smoothing=var_smoothing),
        X_full_scaled,
        y,
        cv=skf,
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5),
    )

    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Training accuracy")
    plt.plot(train_sizes, val_scores.mean(axis=1), marker="s", label="Validation accuracy")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves (Optimized GNB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "learning_curve_gnb.png"))
    plt.close()

    # ===== 7. Permutation importance (optional sanity-check) =====
    print("\nPermutation feature importance on test set (optimized GNB):")
    result_imp = permutation_importance(opt_gnb, X_test_scaled, y_test, n_repeats=20, random_state=42, n_jobs=-1)
    imp_means = result_imp.importances_mean
    for col, val in sorted(zip(FEATURE_COLUMNS, imp_means), key=lambda x: -x[1]):
        print(f"{col:20s}: {val:.4f}")

    # ===== 8. ANN (MLP) baseline =====
    print("\nTraining ANN (MLP) baseline...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.2,
    )

    start = time.perf_counter()
    mlp.fit(X_train_scaled, y_train)
    ann_train_time = time.perf_counter() - start
    y_pred_ann = mlp.predict(X_test_scaled)
    ann_acc = accuracy_score(y_test, y_pred_ann)
    ann_weighted_f1 = f1_score(y_test, y_pred_ann, average="weighted")

    print(f"ANN training time: {ann_train_time:.4f} s")
    print(f"ANN test accuracy: {ann_acc:.4f}")
    print(f"ANN weighted F1:  {ann_weighted_f1:.4f}")

    

    # ===== 9. Simple preprocessing ablation study =====
    print("\nPreprocessing ablation (single train/test split):")
    # Variant A: baseline GNB (no scaling, default smoothing)
    acc_A = accuracy_score(y_test, y_pred_base)
    f1_A = f1_score(y_test, y_pred_base, average="weighted")

    # Variant B: scaling only (default smoothing)
    gnb_B = GaussianNB()
    gnb_B.fit(X_train_scaled, y_train)
    y_pred_B = gnb_B.predict(X_test_scaled)
    acc_B = accuracy_score(y_test, y_pred_B)
    f1_B = f1_score(y_test, y_pred_B, average="weighted")

    # Variant C: scaling + tuned smoothing (optimized)
    acc_C = acc
    f1_C = weighted_f1

    print("Variant\t\tAccuracy\tWeighted F1")
    print(f"Baseline (A)\t{acc_A:.4f}\t\t{f1_A:.4f}")
    print(f"+Scaling (B)\t{acc_B:.4f}\t\t{f1_B:.4f}")
    print(f"+Tuned (C)\t{acc_C:.4f}\t\t{f1_C:.4f}")

    # ===== 10. Performance by institution type (optional subgroup analysis) =====
    print("\nPerformance by institution type (optimized GNB, test set):")
    # inst_test holds original string labels
    unique_insts = np.unique(inst_test)
    for inst in unique_insts:
        mask = inst_test == inst
        if mask.sum() < 5:
            continue  # too few samples
        acc_inst = accuracy_score(y_test[mask], y_pred_opt[mask])
        f1_inst = f1_score(y_test[mask], y_pred_opt[mask], average="weighted")
        print(f"{inst:15s} -> n={mask.sum():3d}, accuracy={acc_inst:.4f}, weighted F1={f1_inst:.4f}")


if __name__ == "__main__":
    main()
