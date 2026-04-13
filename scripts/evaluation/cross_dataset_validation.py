"""
optimized_cross_dataset.py  — FIXED VERSION
══════════════════════════════════════════════════════════════════
Correct SMOTE strategy per dataset type:

  Bangladesh  → ALL categorical → use SMOTEN  (not SMOTE-NC)
  xAPI        → mixed categorical + continuous → use SMOTENC

Stage 1: Per-dataset GridSearchCV (var_smoothing, scored on balanced_accuracy)
Stage 2: Correct oversampling per feature type
Stage 3: MI-based feature selection (xAPI only — fixes independence violation)

INSTALL:
    pip install imbalanced-learn
══════════════════════════════════════════════════════════════════
"""

import os, time, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold,
    GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)

# SMOTE imports
try:
    from imblearn.over_sampling import SMOTEN
    SMOTEN_AVAILABLE = True
except ImportError:
    SMOTEN_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTENC
    SMOTENC_AVAILABLE = True
except ImportError:
    SMOTENC_AVAILABLE = False

if not SMOTEN_AVAILABLE or not SMOTENC_AVAILABLE:
    print("WARNING: imbalanced-learn not fully installed.")
    print("Run:  pip install imbalanced-learn\n")

os.makedirs('static', exist_ok=True)
os.makedirs('models', exist_ok=True)


# ══════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════

def preprocess_and_encode(df, feature_cols, target_col, rare_threshold=0.01):
    imputer = SimpleImputer(strategy='most_frequent')
    df_imp  = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    for col in feature_cols:
        freq = df_imp[col].value_counts(normalize=True)
        rare = freq[freq < rare_threshold].index
        if len(rare):
            df_imp[col] = df_imp[col].replace(rare, 'Other')
    encoders, df_enc = {}, df_imp.copy()
    for col in feature_cols + [target_col]:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_imp[col].astype(str))
        encoders[col] = le
    X           = df_enc[feature_cols].values
    y           = df_enc[target_col].values
    class_names = encoders[target_col].classes_
    return X, y, class_names, encoders


# ══════════════════════════════════════════════════════════════
# STAGE 1 — var_smoothing tuning
# ══════════════════════════════════════════════════════════════

def tune_var_smoothing(X_train_sc, y_train, label):
    print(f"\n  [Stage 1] Tuning var_smoothing for {label}...")
    param_grid = {'var_smoothing': np.logspace(-12, -1, 13)}
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(GaussianNB(), param_grid, cv=skf,
                        scoring='balanced_accuracy', n_jobs=-1)
    grid.fit(X_train_sc, y_train)
    vs = grid.best_params_['var_smoothing']
    print(f"  Best var_smoothing : {vs:.2e}")
    print(f"  CV balanced acc    : {grid.best_score_:.4f}")
    return vs


# ══════════════════════════════════════════════════════════════
# STAGE 2a — SMOTEN (fully categorical — Bangladesh)
# ══════════════════════════════════════════════════════════════

def apply_smoten(X_train, y_train, label):
    """SMOTEN: all features are categorical (label-encoded integers)."""
    if not SMOTEN_AVAILABLE:
        print("  [Stage 2] SMOTEN skipped — pip install imbalanced-learn")
        return X_train, y_train
    print(f"\n  [Stage 2] Applying SMOTEN (fully-categorical) for {label}...")
    print(f"  Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    smoten = SMOTEN(random_state=42,
                    k_neighbors=min(5, min(np.bincount(y_train)) - 1))
    X_res, y_res = smoten.fit_resample(X_train, y_train)
    print(f"  After : {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════════
# STAGE 2b — SMOTENC (mixed — xAPI)
# ══════════════════════════════════════════════════════════════

def apply_smotenc(X_train, y_train, categorical_indices, label):
    """SMOTENC: mixed categorical + continuous features."""
    if not SMOTENC_AVAILABLE:
        print("  [Stage 2] SMOTENC skipped — pip install imbalanced-learn")
        return X_train, y_train
    print(f"\n  [Stage 2] Applying SMOTENC (mixed) for {label}...")
    print(f"  Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    smotenc = SMOTENC(categorical_features=categorical_indices,
                      random_state=42,
                      k_neighbors=min(5, min(np.bincount(y_train)) - 1))
    X_res, y_res = smotenc.fit_resample(X_train, y_train)
    print(f"  After : {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════════
# STAGE 3 — MI Feature Selection (xAPI only)
# ══════════════════════════════════════════════════════════════

def select_features_mi(X_tr, y_tr, X_te, X_all, feature_names, k, label):
    """Fit on train only. Transform train, test, full."""
    print(f"\n  [Stage 3] MI feature selection top-{k} for {label}...")
    selector  = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_tr, y_tr)
    sel_idx   = selector.get_support(indices=True)
    sel_names = [feature_names[i] for i in sel_idx]

    mi     = mutual_info_classif(X_tr, y_tr, random_state=42)
    ranked = sorted(zip(feature_names, mi), key=lambda x: -x[1])
    print(f"  MI scores:")
    for fname, score in ranked:
        mark = " ✓" if fname in sel_names else "  "
        print(f"  {mark} {fname:<28} {score:.4f}")
    print(f"\n  Selected: {sel_names}")

    return (selector.transform(X_tr),
            selector.transform(X_te),
            selector.transform(X_all),
            sel_names, sel_idx)


# ══════════════════════════════════════════════════════════════
# CORE EVALUATOR
# ══════════════════════════════════════════════════════════════

def evaluate(X_tr, X_te, y_tr, y_te, X_cv, y_cv,
             class_names, var_smoothing, label, tag):

    # GNB
    t0  = time.perf_counter()
    gnb = GaussianNB(var_smoothing=var_smoothing)
    gnb.fit(X_tr, y_tr)
    gnb_train_t = time.perf_counter() - t0

    n_reps = 1000
    t0 = time.perf_counter()
    for _ in range(n_reps):
        gnb.predict(X_te[:1])
    gnb_inf_ms = (time.perf_counter() - t0) / n_reps * 1000

    y_pred  = gnb.predict(X_te)
    gnb_acc = accuracy_score(y_te, y_pred)
    gnb_bal = balanced_accuracy_score(y_te, y_pred)
    gnb_mf1 = f1_score(y_te, y_pred, average='macro')
    gnb_wf1 = f1_score(y_te, y_pred, average='weighted')

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv   = cross_val_score(GaussianNB(var_smoothing=var_smoothing),
                           X_cv, y_cv, cv=rskf,
                           scoring='balanced_accuracy', n_jobs=-1)

    # ANN
    t0  = time.perf_counter()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                        solver='adam', max_iter=1000, random_state=42,
                        early_stopping=True, validation_fraction=0.2,
                        n_iter_no_change=20)
    mlp.fit(X_tr, y_tr)
    ann_train_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_reps):
        mlp.predict(X_te[:1])
    ann_inf_ms = (time.perf_counter() - t0) / n_reps * 1000

    y_pred_ann = mlp.predict(X_te)
    ann_acc    = accuracy_score(y_te, y_pred_ann)
    ann_wf1    = f1_score(y_te, y_pred_ann, average='weighted')
    speedup    = ann_train_t / max(gnb_train_t, 1e-9)
    wins       = gnb_acc > ann_acc

    print(f"\n  ┌─ {label}")
    print(f"  │  GNB accuracy      : {gnb_acc:.4f}")
    print(f"  │  GNB balanced acc  : {gnb_bal:.4f}")
    print(f"  │  GNB macro F1      : {gnb_mf1:.4f}")
    print(f"  │  GNB weighted F1   : {gnb_wf1:.4f}")
    print(f"  │  GNB CV bal.acc    : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"  │  GNB train time    : {gnb_train_t:.6f} s")
    print(f"  │  GNB inf/sample    : {gnb_inf_ms:.4f} ms")
    print(f"  │  ANN accuracy      : {ann_acc:.4f}")
    print(f"  │  ANN weighted F1   : {ann_wf1:.4f}")
    print(f"  │  ANN train time    : {ann_train_t:.4f} s")
    print(f"  │  Speedup vs ANN    : {speedup:.0f}x")
    print(f"  └  Result            : {'GNB WINS' if wins else 'ANN wins'}")
    print(f"\n  Classification Report (GNB):")
    print(classification_report(y_te, y_pred, target_names=class_names))

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{label}\nAcc: {gnb_acc:.2%}  Bal.Acc: {gnb_bal:.2%}')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'static/cm_{tag}.png'); plt.close()

    return {
        'label'        : label,
        'var_smoothing': float(var_smoothing),
        'gnb_accuracy' : round(gnb_acc  * 100, 2),
        'gnb_bal_acc'  : round(gnb_bal  * 100, 2),
        'gnb_macro_f1' : round(gnb_mf1  * 100, 2),
        'gnb_wf1'      : round(gnb_wf1  * 100, 2),
        'cv_bal_mean'  : round(cv.mean() * 100, 2),
        'cv_bal_std'   : round(cv.std()  * 100, 2),
        'gnb_train_s'  : round(gnb_train_t, 6),
        'gnb_inf_ms'   : round(gnb_inf_ms,  4),
        'ann_accuracy' : round(ann_acc  * 100, 2),
        'ann_wf1'      : round(ann_wf1  * 100, 2),
        'ann_train_s'  : round(ann_train_t, 4),
        'ann_inf_ms'   : round(ann_inf_ms,  4),
        'gnb_beats_ann': bool(wins),
        'speedup'      : round(speedup, 1),
    }


# ══════════════════════════════════════════════════════════════
# DATASET LOADERS
# ══════════════════════════════════════════════════════════════

def load_bangladesh():
    FEATURES = [
        'Gender', 'Age', 'Education Level', 'Institution Type',
        'IT Student', 'Location', 'Load-shedding', 'Financial Condition',
        'Internet Type', 'Network Type', 'Class Duration', 'Self Lms', 'Device'
    ]
    TARGET = 'Adaptivity Level'
    df = pd.read_csv(
        'data/students_adaptability_level_online_education.csv'
    )
    X, y, cls, enc = preprocess_and_encode(df, FEATURES, TARGET)
    return X, y, cls, FEATURES


def load_xapi():
    FEATURES = [
        'gender',                    # 0  cat
        'NationalITy',               # 1  cat
        'StageID',                   # 2  cat
        'GradeID',                   # 3  cat
        'SectionID',                 # 4  cat
        'Topic',                     # 5  cat
        'Semester',                  # 6  cat
        'Relation',                  # 7  cat
        'raisedhands',               # 8  continuous 0-100
        'VisITedResources',          # 9  continuous 0-100
        'AnnouncementsView',         # 10 continuous 0-100
        'Discussion',                # 11 continuous 0-100
        'ParentAnsweringSurvey',     # 12 cat
        'ParentschoolSatisfaction',  # 13 cat
        'StudentAbsenceDays',        # 14 cat
    ]
    CAT_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14]
    TARGET  = 'Class'
    path = 'data/xAPI-Edu-Data.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(
            "\n  Place xAPI-Edu-Data.csv in data/\n"
            "  https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data\n"
        )
    df = pd.read_csv(path)[FEATURES + [TARGET]].dropna(subset=[TARGET])
    X, y, cls, enc = preprocess_and_encode(df, FEATURES, TARGET)
    return X, y, cls, FEATURES, CAT_IDX


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    results = {}

    # ── BANGLADESH ────────────────────────────────────────────
    print("\n" + "="*65)
    print("  BANGLADESH — PRIMARY DATASET")
    print("="*65)

    X1, y1, cls1, feat1 = load_bangladesh()
    scaler1 = StandardScaler()
    X1_tr_raw, X1_te_raw, y1_tr, y1_te = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1
    )
    X1_tr_sc  = scaler1.fit_transform(X1_tr_raw)
    X1_te_sc  = scaler1.transform(X1_te_raw)
    X1_all_sc = scaler1.transform(X1)

    vs_bd = tune_var_smoothing(X1_tr_sc, y1_tr, "Bangladesh")

    print("\n  -- Config A: Original (1e-12, no SMOTE) --")
    results['BD_A'] = evaluate(
        X1_tr_sc, X1_te_sc, y1_tr, y1_te, X1_all_sc, y1,
        cls1, 1e-12, "Bangladesh Original", "bd_A"
    )

    print(f"\n  -- Config B: Tuned ({vs_bd:.2e}), no SMOTE --")
    results['BD_B'] = evaluate(
        X1_tr_sc, X1_te_sc, y1_tr, y1_te, X1_all_sc, y1,
        cls1, vs_bd, f"Bangladesh Tuned ({vs_bd:.2e})", "bd_B"
    )

    if SMOTEN_AVAILABLE:
        print(f"\n  -- Config C: Tuned ({vs_bd:.2e}) + SMOTEN --")
        X1_tr_sm, y1_tr_sm = apply_smoten(X1_tr_raw, y1_tr, "Bangladesh")
        scaler1b     = StandardScaler()
        X1_tr_sm_sc  = scaler1b.fit_transform(X1_tr_sm)
        X1_te_sm_sc  = scaler1b.transform(X1_te_raw)
        X1_all_sm_sc = scaler1b.transform(X1)
        results['BD_C'] = evaluate(
            X1_tr_sm_sc, X1_te_sm_sc, y1_tr_sm, y1_te,
            X1_all_sm_sc, y1,
            cls1, vs_bd, "Bangladesh Tuned + SMOTEN", "bd_C"
        )
    else:
        print("\n  Config C skipped: pip install imbalanced-learn")

    # ── xAPI ──────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  xAPI — SECONDARY DATASET (Jordan / Kuwait)")
    print("="*65)

    X2, y2, cls2, feat2, cat2 = load_xapi()
    scaler2 = StandardScaler()
    X2_tr_raw, X2_te_raw, y2_tr, y2_te = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )
    X2_tr_sc  = scaler2.fit_transform(X2_tr_raw)
    X2_te_sc  = scaler2.transform(X2_te_raw)
    X2_all_sc = scaler2.transform(X2)

    vs_xapi = tune_var_smoothing(X2_tr_sc, y2_tr, "xAPI")

    print("\n  -- Config D: Original (1e-12, 15 features) --")
    results['xAPI_D'] = evaluate(
        X2_tr_sc, X2_te_sc, y2_tr, y2_te, X2_all_sc, y2,
        cls2, 1e-12, "xAPI Original (1e-12)", "xapi_D"
    )

    print(f"\n  -- Config E: Tuned ({vs_xapi:.2e}), 15 features --")
    results['xAPI_E'] = evaluate(
        X2_tr_sc, X2_te_sc, y2_tr, y2_te, X2_all_sc, y2,
        cls2, vs_xapi, f"xAPI Tuned ({vs_xapi:.2e})", "xapi_E"
    )

    print(f"\n  -- Config F: Tuned + Feature Selection (top 10 MI) --")
    X2_tr_fs, X2_te_fs, X2_all_fs, feat_sel, sel_idx = select_features_mi(
        X2_tr_sc, y2_tr, X2_te_sc, X2_all_sc, feat2, k=10, label="xAPI"
    )
    results['xAPI_F'] = evaluate(
        X2_tr_fs, X2_te_fs, y2_tr, y2_te, X2_all_fs, y2,
        cls2, vs_xapi, "xAPI Tuned + FS top-10", "xapi_F"
    )

    if SMOTENC_AVAILABLE:
        print(f"\n  -- Config G: Tuned + FS + SMOTENC (full optimization) --")
        # Apply selection on RAW features so SMOTENC gets unscaled values
        sel_raw = SelectKBest(mutual_info_classif, k=10)
        sel_raw.fit(X2_tr_raw, y2_tr)
        sel_raw_idx   = sel_raw.get_support(indices=True)
        X2_tr_raw_fs  = sel_raw.transform(X2_tr_raw)
        X2_te_raw_fs  = sel_raw.transform(X2_te_raw)
        X2_all_raw_fs = sel_raw.transform(X2)

        # Remap categorical indices after feature selection
        new_cat_idx = [i for i, orig_i in enumerate(sel_raw_idx)
                       if orig_i in cat2]
        print(f"  Categorical indices after FS: {new_cat_idx}")

        if 0 < len(new_cat_idx) < 10:
            X2_tr_sm, y2_tr_sm = apply_smotenc(
                X2_tr_raw_fs, y2_tr, new_cat_idx, "xAPI post-FS"
            )
            scaler_g     = StandardScaler()
            X2_tr_sm_sc  = scaler_g.fit_transform(X2_tr_sm)
            X2_te_g_sc   = scaler_g.transform(X2_te_raw_fs)
            X2_all_g_sc  = scaler_g.transform(X2_all_raw_fs)
            results['xAPI_G'] = evaluate(
                X2_tr_sm_sc, X2_te_g_sc, y2_tr_sm, y2_te,
                X2_all_g_sc, y2,
                cls2, vs_xapi,
                "xAPI Tuned + FS + SMOTENC", "xapi_G"
            )
        else:
            print(f"  Config G skipped: need mixed features after FS.")
    else:
        print("\n  Config G skipped: pip install imbalanced-learn")

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "="*75)
    print("  OPTIMIZATION SUMMARY")
    print("="*75)
    print(f"  {'Config':<42} {'Acc%':>6} {'Bal%':>6} {'WF1%':>6} {'Beat ANN':>9}")
    print("  " + "-"*71)

    order = [
        ('A  Bangladesh Original',          'BD_A'),
        ('B  Bangladesh Tuned',             'BD_B'),
        ('C  Bangladesh Tuned + SMOTEN',    'BD_C'),
        ('D  xAPI Original',                'xAPI_D'),
        ('E  xAPI Tuned',                   'xAPI_E'),
        ('F  xAPI Tuned + Feat.Sel.',       'xAPI_F'),
        ('G  xAPI Tuned + FS + SMOTENC',    'xAPI_G'),
    ]
    for lbl, key in order:
        if key not in results:
            print(f"  {lbl:<42} {'[skipped]':>30}")
            continue
        r   = results[key]
        win = '   YES' if r['gnb_beats_ann'] else '    NO'
        print(f"  {lbl:<42} "
              f"{r['gnb_accuracy']:>6.2f} "
              f"{r['gnb_bal_acc']:>6.2f} "
              f"{r['gnb_wf1']:>6.2f}"
              f"{win:>9}")

    xapi_valid = {k: results[k] for k in ['xAPI_D','xAPI_E','xAPI_F','xAPI_G']
                  if k in results}
    if xapi_valid:
        best = max(xapi_valid, key=lambda k: xapi_valid[k]['gnb_accuracy'])
        br   = xapi_valid[best]
        gain = round(br['gnb_accuracy'] - 61.46, 2)
        print(f"\n  BEST xAPI config : {br['label']}")
        print(f"  GNB accuracy     : {br['gnb_accuracy']}%  (was 61.46%, gain: +{gain}%)")
        print(f"  GNB balanced acc : {br['gnb_bal_acc']}%")
        print(f"  GNB weighted F1  : {br['gnb_wf1']}%")
        print(f"  Beats ANN        : {'YES' if br['gnb_beats_ann'] else 'NO'}")
        print(f"  ANN accuracy     : {br['ann_accuracy']}%")

    with open('models/optimized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved -> models/optimized_results.json")
    return results


if __name__ == '__main__':
    main()