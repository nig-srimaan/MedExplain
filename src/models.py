"""
Classification Models, Optuna TPE Hyperparameter Optimization, Stratified 5-Fold Cross-Validation,
and Wilcoxon Signed-Rank Statistical Significance Testing (Sections 3.5, 4.2-4.5 of Paper).

Models Benchmarked:
1. Logistic Regression (ElasticNet L1+L2 regularization)
2. Decision Tree (CART with Gini impurity)
3. Random Forest (Bagged decorrelated ensemble)
4. XGBoost (Second-order regularized gradient boosting)
5. CatBoost (Oblivious trees with ordered boosting)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import wilcoxon

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

from src.preprocessing import ClinicalPreprocessor


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns initialized classification models with tuned hyperparameters matching Section 4.3.
    """
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=2000,
            random_state=random_state,
        ),
        "Decision Tree (CART)": DecisionTreeClassifier(
            criterion="gini",
            max_depth=6,
            min_samples_split=5,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.2,
            eval_metric="logloss",
            random_state=random_state,
        )

    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            verbose=0,
            random_seed=random_state,
        )

    return models


def optimize_hyperparameters_optuna(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Bayesian Optimization with Tree-structured Parzen Estimators (TPE) via Optuna over 50 iterations,
    targeting validation ROC-AUC (Section 4.3).
    """
    if optuna is None:
        raise RuntimeError("Optuna is required for Bayesian hyperparameter tuning.")

    def objective(trial):
        if model_name == "Logistic Regression":
            c_val = trial.suggest_float("C", 0.001, 100.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", C=c_val, l1_ratio=l1_ratio,
                max_iter=2000, random_state=random_state
            )
        elif model_name == "Random Forest":
            n_est = trial.suggest_int("n_estimators", 100, 500)
            max_d = trial.suggest_int("max_depth", 3, 12)
            min_split = trial.suggest_int("min_samples_split", 2, 10)
            clf = RandomForestClassifier(
                n_estimators=n_est, max_depth=max_d, min_samples_split=min_split,
                random_state=random_state, n_jobs=-1
            )
        elif model_name == "XGBoost" and XGBClassifier is not None:
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            max_d = trial.suggest_int("max_depth", 3, 8)
            sub = trial.suggest_float("subsample", 0.6, 1.0)
            col = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            gamma = trial.suggest_float("gamma", 0.0, 5.0)
            clf = XGBClassifier(
                n_estimators=300, learning_rate=lr, max_depth=max_d,
                subsample=sub, colsample_bytree=col, gamma=gamma,
                eval_metric="logloss", random_state=random_state
            )
        elif model_name == "CatBoost" and CatBoostClassifier is not None:
            iters = trial.suggest_int("iterations", 200, 800)
            lr = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
            depth = trial.suggest_int("depth", 4, 8)
            l2 = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
            clf = CatBoostClassifier(
                iterations=iters, learning_rate=lr, depth=depth,
                l2_leaf_reg=l2, verbose=0, random_seed=random_state
            )
        else:
            max_d = trial.suggest_int("max_depth", 3, 10)
            min_split = trial.suggest_int("min_samples_split", 2, 10)
            clf = DecisionTreeClassifier(
                max_depth=max_d, min_samples_split=min_split, random_state=random_state
            )

        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, probs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def evaluate_domain_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    zero_missing_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Executes Stratified 5-Fold Cross-Validation with leak-free preprocessing (MICE, Winsorizer,
    RobustScaler, and SMOTE-Tomek fitted strictly inside each training fold).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    candidate_models = get_candidate_models(random_state=random_state)

    fold_auc_scores: Dict[str, List[float]] = {m: [] for m in candidate_models}
    summary_rows = []

    for model_name, base_model in candidate_models.items():
        accs, precs, recs, f1s, aucs, briers = [], [], [], [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
            y_train_raw, y_test_raw = y.iloc[train_idx], y.iloc[test_idx]

            preproc = ClinicalPreprocessor(
                zero_missing_cols=zero_missing_cols,
                mice_iterations=10,
                random_state=random_state,
            )
            X_train_proc, y_train_proc = preproc.fit_transform_train(
                X_train_raw, y_train_raw, apply_smote_tomek=True
            )
            X_test_proc = preproc.transform_test(X_test_raw)

            # Clone or fit fresh instance
            from sklearn.base import clone
            clf = clone(base_model)
            clf.fit(X_train_proc, y_train_proc)

            y_pred = clf.predict(X_test_proc)
            y_prob = clf.predict_proba(X_test_proc)[:, 1]

            accs.append(accuracy_score(y_test_raw, y_pred))
            precs.append(precision_score(y_test_raw, y_pred, zero_division=0))
            recs.append(recall_score(y_test_raw, y_pred, zero_division=0))
            f1s.append(f1_score(y_test_raw, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_test_raw, y_prob))
            briers.append(brier_score_loss(y_test_raw, y_prob))

        fold_auc_scores[model_name] = aucs
        summary_rows.append({
            "Classifier": model_name,
            "Accuracy_Mean": np.mean(accs),
            "Accuracy_Std": np.std(accs),
            "Precision_Mean": np.mean(precs),
            "Precision_Std": np.std(precs),
            "Recall_Mean": np.mean(recs),
            "Recall_Std": np.std(recs),
            "F1_Mean": np.mean(f1s),
            "F1_Std": np.std(f1s),
            "ROC_AUC_Mean": np.mean(aucs),
            "ROC_AUC_Std": np.std(aucs),
            "Brier_Score_Mean": np.mean(briers),
        })

    return pd.DataFrame(summary_rows), fold_auc_scores


def run_wilcoxon_significance_test(
    fold_scores_a: List[float], fold_scores_b: List[float]
) -> Tuple[float, float]:
    """
    Pair-wise two-tailed Wilcoxon signed-rank test across validation folds (Section 4.5).
    """
    stat, p_val = wilcoxon(fold_scores_a, fold_scores_b, zero_method="zsplit", alternative="two-sided")
    return float(stat), float(p_val)
