"""
Dataset Loader and Domain Metadata for MedExplain.
Supports:
1. Cardiovascular Domain: Cleveland Heart Disease (N = 303, 13 features)
2. Metabolic Domain: Pima Indians Diabetes (N = 768, 8 features)
3. Oncological Domain: Wisconsin Breast Cancer Diagnostic - WDBC (N = 569, 30 features)
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any

DOMAIN_CONFIG: Dict[str, Dict[str, Any]] = {
    "heart": {
        "name": "Cardiovascular (Cleveland Heart Disease)",
        "filename": "cleveland_heart_disease.csv",
        "target_col": "target",
        "positive_label": "Coronary Artery Disease (>=50% stenosis)",
        "negative_label": "Normal / No Significant Stenosis",
        "zero_missing_cols": [],
        "categorical_cols": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "description": "UCI Cleveland Heart Disease dataset (N=303, 13 clinical & angiographic attributes).",
    },
    "diabetes": {
        "name": "Metabolic (Pima Indians Diabetes)",
        "filename": "pima_indians_diabetes.csv",
        "target_col": "Outcome",
        "positive_label": "Type 2 Diabetes Onset (Positive)",
        "negative_label": "Non-Diabetic Control (Negative)",
        "zero_missing_cols": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "categorical_cols": [],
        "description": "NIDDK Pima Indians Diabetes cohort (N=768, 8 metabolic & physiological attributes).",
    },
    "breast_cancer": {
        "name": "Oncology (Wisconsin Breast Cancer - WDBC)",
        "filename": "wisconsin_breast_cancer.csv",
        "target_col": "target",
        "positive_label": "Malignant (M)",
        "negative_label": "Benign (B)",
        "zero_missing_cols": [],
        "categorical_cols": [],
        "description": "Wisconsin Breast Cancer Diagnostic (WDBC) FNA biopsy dataset (N=569, 30 nuclear morphology features).",
    },
}


def load_clinical_dataset(domain: str, data_dir: str = "data") -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Loads the specified clinical dataset and splits into features (X) and binary target (y).
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain '{domain}'. Valid options: {list(DOMAIN_CONFIG.keys())}")

    cfg = DOMAIN_CONFIG[domain]
    csv_path = os.path.join(data_dir, cfg["filename"])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    target_col = cfg["target_col"]
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()
    return X, y, cfg
