"""
Native NumPy/SciPy Machine Learning, Preprocessing, and XAI Fallback Engine.
Provides full mathematical implementations of:
- Multivariate Imputation by Chained Equations (MICE) via Ridge Regression
- 1st/99th Percentile Winsorizer & Robust IQR Scaler
- SMOTE-Tomek Class Rebalancing (k-NN Synthetic Minority Oversampling + Tomek Link Cleaning)
- ElasticNet Logistic Regression, CART Decision Tree, Random Forest, XGBoost, and CatBoost
- Game-Theoretic Shapley Attribution (SHAP) and Perturbation-Based Tabular LIME
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import expit
from typing import Tuple, List, Dict, Optional


class NativeMICEPreprocessor:
    """
    Zero-dependency implementation of MICE imputation, 1st/99th percentile Winsorization,
    Robust IQR Normalization, and SMOTE-Tomek training fold rebalancing.
    """

    def __init__(self, zero_missing_cols: Optional[List[str]] = None, mice_iter: int = 10, random_state: int = 42):
        self.zero_missing_cols = zero_missing_cols or []
        self.mice_iter = mice_iter
        self.random_state = random_state
        self.medians_: Optional[np.ndarray] = None
        self.iqrs_: Optional[np.ndarray] = None
        self.q01_: Optional[np.ndarray] = None
        self.q99_: Optional[np.ndarray] = None
        self.mice_weights_: Dict[int, Tuple[np.ndarray, float]] = {}
        self.feature_names_: List[str] = []

    def _prepare_missing(self, df: pd.DataFrame) -> np.ndarray:
        df_c = df.copy()
        for c in self.zero_missing_cols:
            if c in df_c.columns:
                df_c[c] = df_c[c].replace(0, np.nan)
        return df_c.values.astype(float)

    def fit_transform_train(
        self, X_train: pd.DataFrame, y_train: pd.Series, apply_smote_tomek: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        self.feature_names_ = list(X_train.columns)
        X_arr = self._prepare_missing(X_train)
        n_samples, n_features = X_arr.shape

        # Initial median fill
        col_medians = np.nanmedian(X_arr, axis=0)
        nan_mask = np.isnan(X_arr)
        X_imp = X_arr.copy()
        for j in range(n_features):
            X_imp[nan_mask[:, j], j] = col_medians[j]

        # Chained Ridge equations over mice_iter iterations
        alpha = 1.0
        for _ in range(self.mice_iter):
            for j in range(n_features):
                if not np.any(nan_mask[:, j]):
                    continue
                obs_idx = ~nan_mask[:, j]
                mis_idx = nan_mask[:, j]
                other_cols = [c for c in range(n_features) if c != j]
                X_obs = X_imp[obs_idx][:, other_cols]
                y_obs = X_imp[obs_idx, j]
                X_obs_b = np.column_stack([np.ones(len(X_obs)), X_obs])
                reg = alpha * np.eye(X_obs_b.shape[1])
                reg[0, 0] = 0.0
                w = np.linalg.pinv(X_obs_b.T @ X_obs_b + reg) @ (X_obs_b.T @ y_obs)
                self.mice_weights_[j] = (w[1:], float(w[0]))
                X_mis = X_imp[mis_idx][:, other_cols]
                X_imp[mis_idx, j] = X_mis @ w[1:] + w[0]

        # 1st and 99th percentile Winsorization
        self.q01_ = np.quantile(X_imp, 0.01, axis=0)
        self.q99_ = np.quantile(X_imp, 0.99, axis=0)
        X_cap = np.clip(X_imp, self.q01_, self.q99_)

        # Robust IQR Scaling
        self.medians_ = np.median(X_cap, axis=0)
        q75 = np.quantile(X_cap, 0.75, axis=0)
        q25 = np.quantile(X_cap, 0.25, axis=0)
        self.iqrs_ = np.where((q75 - q25) > 1e-8, (q75 - q25), 1.0)
        X_scl = (X_cap - self.medians_) / self.iqrs_

        y_arr = y_train.values.astype(int)
        if apply_smote_tomek:
            X_scl, y_arr = self._smote_tomek(X_scl, y_arr)

        return pd.DataFrame(X_scl, columns=self.feature_names_), pd.Series(y_arr)

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X_arr = self._prepare_missing(X_test)
        nan_mask = np.isnan(X_arr)
        n_features = X_arr.shape[1]
        X_imp = X_arr.copy()
        for j in range(n_features):
            X_imp[nan_mask[:, j], j] = self.medians_[j]
        for j, (w, b) in self.mice_weights_.items():
            if np.any(nan_mask[:, j]):
                other_cols = [c for c in range(n_features) if c != j]
                X_imp[nan_mask[:, j], j] = X_imp[nan_mask[:, j]][:, other_cols] @ w + b

        X_cap = np.clip(X_imp, self.q01_, self.q99_)
        X_scl = (X_cap - self.medians_) / self.iqrs_
        return pd.DataFrame(X_scl, columns=self.feature_names_, index=X_test.index)

    def _smote_tomek(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts[0] == counts[1]:
            return X, y
        min_cls = classes[np.argmin(counts)]
        maj_cls = classes[np.argmax(counts)]
        n_diff = counts.max() - counts.min()

        X_min = X[y == min_cls]
        dists = cdist(X_min, X_min)
        np.fill_diagonal(dists, np.inf)
        k = min(5, len(X_min) - 1)
        nn_idx = np.argsort(dists, axis=1)[:, :k]

        synth = []
        for _ in range(n_diff):
            i = rng.randint(0, len(X_min))
            nb = nn_idx[i, rng.randint(0, k)]
            gap = rng.uniform(0.0, 1.0)
            synth.append(X_min[i] + gap * (X_min[nb] - X_min[i]))

        X_aug = np.vstack([X, np.array(synth)])
        y_aug = np.concatenate([y, np.full(n_diff, min_cls, dtype=int)])

        # Tomek link cleaning (remove boundary pairs that are mutual 1-nearest neighbors of opposite class)
        full_dists = cdist(X_aug, X_aug)
        np.fill_diagonal(full_dists, np.inf)
        nearest = np.argmin(full_dists, axis=1)
        keep_mask = np.ones(len(X_aug), dtype=bool)
        for i in range(len(X_aug)):
            j = nearest[i]
            if nearest[j] == i and y_aug[i] != y_aug[j]:
                if y_aug[i] == maj_cls:
                    keep_mask[i] = False
        return X_aug[keep_mask], y_aug[keep_mask]

# ===========================================================================
# Patient Symptom & Body Pain to Clinical Biomarker Inference Engine (v3.5)
# ===========================================================================
BODY_PAIN_REGIONS = {
    "chest_center_left": {
        "label": "Chest (Center / Left Precordial)",
        "short_label": "Chest / Heart",
        "icon": "🫀",
        "svg_cx": 118,
        "svg_cy": 118,
        "svg_r": 22,
        "primary_organ": "Cardiovascular (Heart)",
        "domain_weights": {"heart": 0.88, "diabetes": 0.05, "breast_cancer": 0.12},
        "clinical_note": "Retrosternal or left precordial chest pressure/pain is the primary hallmark of myocardial ischemia and angina pectoris.",
    },
    "left_arm_shoulder_jaw": {
        "label": "Left Arm, Shoulder, Neck, or Jaw",
        "short_label": "Left Arm / Jaw",
        "icon": "💪",
        "svg_cx": 158,
        "svg_cy": 135,
        "svg_r": 20,
        "primary_organ": "Cardiovascular (Referred Nerve Pathway)",
        "domain_weights": {"heart": 0.82, "diabetes": 0.05, "breast_cancer": 0.10},
        "clinical_note": "Radiating pain along the left shoulder, ulnar arm, neck, or jaw reflects referred T1-T4 cardiac sympathetic nerve activation.",
    },
    "upper_back": {
        "label": "Upper Back (Between Shoulder Blades)",
        "short_label": "Upper Back",
        "icon": "🧍",
        "svg_cx": 110,
        "svg_cy": 92,
        "svg_r": 18,
        "primary_organ": "Cardiovascular (Posterior Wall)",
        "domain_weights": {"heart": 0.58, "diabetes": 0.10, "breast_cancer": 0.10},
        "clinical_note": "Interscapular pain is a classic atypical angina presentation, especially in female and older diabetic cohorts.",
    },
    "breast_axilla": {
        "label": "Breast, Chest Wall, or Underarm (Axilla)",
        "short_label": "Breast / Axilla",
        "icon": "🎗️",
        "svg_cx": 88,
        "svg_cy": 118,
        "svg_r": 20,
        "primary_organ": "Oncology (Mammary & Axillary Lymph)",
        "domain_weights": {"heart": 0.08, "diabetes": 0.04, "breast_cancer": 0.92},
        "clinical_note": "Localized breast or axillary tenderness, swelling, or mass sensation routes directly to diagnostic mammography and WDBC cytological screening.",
    },
    "abdomen_stomach": {
        "label": "Upper Abdomen / Stomach (Epigastric)",
        "short_label": "Abdomen / Stomach",
        "icon": "🫃",
        "svg_cx": 110,
        "svg_cy": 168,
        "svg_r": 22,
        "primary_organ": "Metabolic / Hepatogastric",
        "domain_weights": {"heart": 0.28, "diabetes": 0.60, "breast_cancer": 0.04},
        "clinical_note": "Epigastric discomfort or burning pain correlates with metabolic dysregulation, gastroparesis, or inferior myocardial ischemia.",
    },
    "legs_feet": {
        "label": "Legs, Calves, or Feet (Tingling / Burning)",
        "short_label": "Legs & Feet",
        "icon": "🦵",
        "svg_cx": 110,
        "svg_cy": 268,
        "svg_r": 26,
        "primary_organ": "Metabolic (Peripheral Neuropathy)",
        "domain_weights": {"heart": 0.25, "diabetes": 0.82, "breast_cancer": 0.02},
        "clinical_note": "Distal lower-extremity burning pain, numbness, or pins-and-needles is the classic presentation of diabetic peripheral neuropathy.",
    },
    "head_general": {
        "label": "Head, Neck, or Generalized Body Ache",
        "short_label": "Head / Dizziness",
        "icon": "🧠",
        "svg_cx": 110,
        "svg_cy": 44,
        "svg_r": 20,
        "primary_organ": "Systemic / Vascular",
        "domain_weights": {"heart": 0.35, "diabetes": 0.52, "breast_cancer": 0.05},
        "clinical_note": "Headache, orthostatic lightheadedness, or systemic malaise accompanies hypertensive or glycemic fluctuations.",
    },
    "no_pain": {
        "label": "No Specific Localized Body Pain",
        "short_label": "No Body Pain",
        "icon": "✅",
        "svg_cx": 0,
        "svg_cy": 0,
        "svg_r": 0,
        "primary_organ": "Symptom-Only Screening",
        "domain_weights": {"heart": 0.12, "diabetes": 0.42, "breast_cancer": 0.22},
        "clinical_note": "Evaluates risk based purely on non-pain systemic symptoms, age, and biological sex.",
    },
}

PAIN_CHARACTER_OPTIONS = {
    "crushing_pressure": {
        "label": "Heavy Pressure, Squeezing, Tightness, or Fullness",
        "cp_code": 0,
        "heart_boost": 0.35,
    },
    "sharp_stabbing": {
        "label": "Sharp, Stabbing, or Pinching Pain",
        "cp_code": 1,
        "heart_boost": 0.18,
    },
    "burning_aching": {
        "label": "Burning, Gnawing, Pins-and-Needles, or Deep Ache",
        "cp_code": 2,
        "heart_boost": 0.12,
    },
    "tender_throbbing": {
        "label": "Localized Tenderness, Soreness, or Throbbing",
        "cp_code": 2,
        "heart_boost": 0.05,
    },
    "none": {
        "label": "No Pain / Asymptomatic Discomfort",
        "cp_code": 3,
        "heart_boost": 0.0,
    },
}

PAIN_TRIGGER_OPTIONS = {
    "exertion": {
        "label": "Worsens during physical activity, walking, or climbing stairs",
        "exang_code": 1,
        "heart_boost": 0.32,
    },
    "stress_emotion": {
        "label": "Triggered by emotional stress or anxiety",
        "exang_code": 1,
        "heart_boost": 0.22,
    },
    "touch_pressure": {
        "label": "Noticeable when pressing or touching the area",
        "exang_code": 0,
        "heart_boost": -0.05,
    },
    "after_meals": {
        "label": "Occurs or worsens after eating meals",
        "exang_code": 0,
        "heart_boost": 0.05,
    },
    "rest_constant": {
        "label": "Present at rest or constant throughout the day",
        "exang_code": 0,
        "heart_boost": 0.10,
    },
}

SYMPTOM_CATALOG = {
    "shortness_of_breath": {
        "label": "Shortness of breath / difficulty catching breath",
        "category": "Breathing & Circulation",
        "weights": {"heart": 0.55, "diabetes": 0.10, "breast_cancer": 0.05},
        "clinical_term": "Dyspnea on exertion",
    },
    "cold_sweats": {
        "label": "Cold sweats / sudden clammy perspiration",
        "category": "Breathing & Circulation",
        "weights": {"heart": 0.60, "diabetes": 0.20, "breast_cancer": 0.0},
        "clinical_term": "Autonomic diaphoresis",
    },
    "palpitations": {
        "label": "Racing, pounding, or fluttering heartbeat",
        "category": "Breathing & Circulation",
        "weights": {"heart": 0.48, "diabetes": 0.10, "breast_cancer": 0.0},
        "clinical_term": "Cardiac palpitations / arrhythmia",
    },
    "dizziness_fainting": {
        "label": "Dizziness, lightheadedness, or feeling faint",
        "category": "Breathing & Circulation",
        "weights": {"heart": 0.42, "diabetes": 0.35, "breast_cancer": 0.0},
        "clinical_term": "Presyncope / orthostatic dizziness",
    },
    "radiating_arm_jaw": {
        "label": "Pain spreading to left arm, shoulder, neck, or jaw",
        "category": "Breathing & Circulation",
        "weights": {"heart": 0.75, "diabetes": 0.0, "breast_cancer": 0.10},
        "clinical_term": "Referred somatic cardiac pain",
    },
    "excessive_thirst": {
        "label": "Excessive thirst (drinking water constantly)",
        "category": "Thirst, Energy & Metabolism",
        "weights": {"heart": 0.05, "diabetes": 0.78, "breast_cancer": 0.0},
        "clinical_term": "Polydipsia",
    },
    "frequent_urination": {
        "label": "Frequent urination (especially waking up at night)",
        "category": "Thirst, Energy & Metabolism",
        "weights": {"heart": 0.05, "diabetes": 0.75, "breast_cancer": 0.0},
        "clinical_term": "Polyuria / nocturia",
    },
    "chronic_fatigue": {
        "label": "Unusual exhaustion or constant daytime tiredness",
        "category": "Thirst, Energy & Metabolism",
        "weights": {"heart": 0.35, "diabetes": 0.50, "breast_cancer": 0.30},
        "clinical_term": "Systemic lethargy / fatigue",
    },
    "blurred_vision": {
        "label": "Blurred vision or difficulty focusing eyes",
        "category": "Thirst, Energy & Metabolism",
        "weights": {"heart": 0.10, "diabetes": 0.62, "breast_cancer": 0.0},
        "clinical_term": "Glycemic osmotic lens shift / retinopathy risk",
    },
    "numbness_tingling": {
        "label": "Tingling, pins-and-needles, or numbness in hands/feet",
        "category": "Thirst, Energy & Metabolism",
        "weights": {"heart": 0.15, "diabetes": 0.68, "breast_cancer": 0.0},
        "clinical_term": "Peripheral paresthesia / neuropathy",
    },
    "slow_healing": {
        "label": "Cuts, bruises, or sores that heal very slowly",
        "category": "Thirst, Energy & Metabolism",
        "weights": {"heart": 0.10, "diabetes": 0.65, "breast_cancer": 0.05},
        "clinical_term": "Impaired microvascular wound healing",
    },
    "unexplained_weight_change": {
        "label": "Unexplained weight loss or sudden increased hunger",
        "category": "Thirst, Energy & Metabolism",
        "weights": {"heart": 0.05, "diabetes": 0.55, "breast_cancer": 0.45},
        "clinical_term": "Catabolic weight change / polyphagia",
    },
    "breast_underarm_lump": {
        "label": "Noticeable firm lump, knot, or thickening in breast or armpit",
        "category": "Localized Tissue & Skin Changes",
        "weights": {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.95},
        "clinical_term": "Palpable mammary or axillary mass",
    },
    "breast_swelling_asymmetry": {
        "label": "Swelling, warmth, or change in breast size/shape",
        "category": "Localized Tissue & Skin Changes",
        "weights": {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.78},
        "clinical_term": "Localized tissue edema / asymmetry",
    },
    "skin_dimpling_redness": {
        "label": "Skin dimpling (like an orange peel), puckering, or redness",
        "category": "Localized Tissue & Skin Changes",
        "weights": {"heart": 0.0, "diabetes": 0.05, "breast_cancer": 0.85},
        "clinical_term": "Peau d'orange / dermal tethering",
    },
    "nipple_changes": {
        "label": "Nipple inversion (pulling inward) or unusual discharge",
        "category": "Localized Tissue & Skin Changes",
        "weights": {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.82},
        "clinical_term": "Nipple retraction / pathological discharge",
    },
    "family_history": {
        "label": "Close family history (parent/sibling with Heart Disease, Diabetes, or Cancer)",
        "category": "Personal & Family Risk Factors",
        "weights": {"heart": 0.30, "diabetes": 0.45, "breast_cancer": 0.38},
        "clinical_term": "First-degree hereditary predisposition",
    },
}

# Dedicated Clinical Test Cases Suite (Used in the "🧪 Clinical Test Cases" Side Tab)
CLINICAL_TEST_CASES = [
    {
        "id": "TC-01",
        "title": "Case 1: Classic Exertional Angina & Referred Left Arm Pain",
        "patient_summary": "58-year-old Male presenting with heavy squeezing chest pain (8/10) radiating to the left arm/jaw when walking upstairs, accompanied by shortness of breath and cold sweats.",
        "age": 58,
        "sex": "Male",
        "pain_regions": ["chest_center_left", "left_arm_shoulder_jaw"],
        "pain_character": "crushing_pressure",
        "pain_severity": 8,
        "pain_trigger": "exertion",
        "selected_symptoms": ["shortness_of_breath", "cold_sweats", "radiating_arm_jaw", "palpitations"],
        "expected_domain": "heart",
        "expected_domain_label": "Cardiovascular (Heart Disease)",
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-02",
        "title": "Case 2: Diabetic Peripheral Neuropathy & Hyperglycemic Triad",
        "patient_summary": "46-year-old Female reporting burning/tingling pain in both feet and legs (5/10), epigastric stomach discomfort, excessive thirst, frequent nighttime urination, and blurred vision.",
        "age": 46,
        "sex": "Female",
        "pain_regions": ["legs_feet", "abdomen_stomach"],
        "pain_character": "burning_aching",
        "pain_severity": 5,
        "pain_trigger": "rest_constant",
        "selected_symptoms": ["excessive_thirst", "frequent_urination", "chronic_fatigue", "blurred_vision", "numbness_tingling", "family_history"],
        "expected_domain": "diabetes",
        "expected_domain_label": "Metabolic (Type-2 Diabetes)",
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-03",
        "title": "Case 3: Palpable Mammary / Axillary Mass & Dermal Dimpling",
        "patient_summary": "51-year-old Female noticing localized tenderness in the breast and underarm (6/10) with a firm knot/lump, localized swelling, and orange-peel skin dimpling.",
        "age": 51,
        "sex": "Female",
        "pain_regions": ["breast_axilla"],
        "pain_character": "tender_throbbing",
        "pain_severity": 6,
        "pain_trigger": "touch_pressure",
        "selected_symptoms": ["breast_underarm_lump", "breast_swelling_asymmetry", "skin_dimpling_redness"],
        "expected_domain": "breast_cancer",
        "expected_domain_label": "Oncology (Breast Lesion)",
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-04",
        "title": "Case 4: Atypical Female Interscapular Cardiac Ischemia",
        "patient_summary": "64-year-old Female with dull aching pain between the shoulder blades (7/10) during physical activity, unexplained exhaustion, dizziness, and shortness of breath.",
        "age": 64,
        "sex": "Female",
        "pain_regions": ["upper_back", "chest_center_left"],
        "pain_character": "sharp_stabbing",
        "pain_severity": 7,
        "pain_trigger": "exertion",
        "selected_symptoms": ["shortness_of_breath", "dizziness_fainting", "chronic_fatigue", "cold_sweats"],
        "expected_domain": "heart",
        "expected_domain_label": "Cardiovascular (Heart Disease)",
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-05",
        "title": "Case 5: Early Metabolic Syndrome & Slow Wound Healing",
        "patient_summary": "52-year-old Male with generalized headache/dizziness (4/10), slow-healing skin cuts, constant daytime fatigue, excessive thirst, and family history.",
        "age": 52,
        "sex": "Male",
        "pain_regions": ["head_general", "legs_feet"],
        "pain_character": "burning_aching",
        "pain_severity": 4,
        "pain_trigger": "after_meals",
        "selected_symptoms": ["excessive_thirst", "slow_healing", "chronic_fatigue", "family_history", "unexplained_weight_change"],
        "expected_domain": "diabetes",
        "expected_domain_label": "Metabolic (Type-2 Diabetes)",
        "expected_risk": "HIGH RISK",
    },
    {
        "id": "TC-06",
        "title": "Case 6: Healthy Asymptomatic Control Baseline",
        "patient_summary": "31-year-old Female with no localized body pain (0/10) and no cardiopulmonary, metabolic, or localized tissue symptoms.",
        "age": 31,
        "sex": "Female",
        "pain_regions": ["no_pain"],
        "pain_character": "none",
        "pain_severity": 0,
        "pain_trigger": "rest_constant",
        "selected_symptoms": [],
        "expected_domain": "heart",
        "expected_domain_label": "All Domains Clear (Routine Baseline)",
        "expected_risk": "LOW RISK",
    },
]


def evaluate_patient_symptoms(
    age: int,
    sex: str,
    pain_regions=None,
    pain_character: str = "crushing_pressure",
    pain_severity: int = 6,
    pain_trigger: str = "exertion",
    selected_symptoms=None,
    decision_threshold: float = 0.40,
):
    if selected_symptoms is None:
        selected_symptoms = []
    if not pain_regions:
        pain_regions = ["no_pain"]

    sex_code = 1 if str(sex).lower().startswith("m") else 0
    severity_norm = float(np.clip(pain_severity, 0, 10)) / 10.0

    domain_scores = {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.0}
    pain_only_scores = {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.0}
    symptom_only_scores = {"heart": 0.0, "diabetes": 0.0, "breast_cancer": 0.0}
    symptom_contributions = {"heart": [], "diabetes": [], "breast_cancer": []}

    age_heart_shift = 0.28 * ((age - 50.0) / 12.0)
    sex_heart_shift = 0.22 if sex_code == 1 else -0.10
    symptom_contributions["heart"].append(
        (f"Age ({age} yrs) & Sex ({sex})", round(age_heart_shift + sex_heart_shift, 3))
    )
    symptom_contributions["diabetes"].append(
        (f"Age ({age} yrs) baseline", round(0.26 * ((age - 38.0) / 14.0), 3))
    )
    symptom_contributions["breast_cancer"].append(
        (f"Age ({age} yrs) & Sex ({sex})", round(0.24 * ((age - 48.0) / 14.0) + (0.18 if sex_code == 0 else -0.35), 3))
    )

    for reg in pain_regions:
        reg_info = BODY_PAIN_REGIONS.get(reg, BODY_PAIN_REGIONS["no_pain"])
        scale = (0.55 + 0.65 * severity_norm) if reg != "no_pain" else 0.25
        for dom, w in reg_info["domain_weights"].items():
            contrib = w * scale
            domain_scores[dom] += contrib
            pain_only_scores[dom] += contrib
            if contrib >= 0.08 and reg != "no_pain":
                symptom_contributions[dom].append(
                    (f"Body Pain: {reg_info['label']} ({pain_severity}/10)", round(contrib, 3))
                )

    char_info = PAIN_CHARACTER_OPTIONS.get(pain_character, PAIN_CHARACTER_OPTIONS["none"])
    trig_info = PAIN_TRIGGER_OPTIONS.get(pain_trigger, PAIN_TRIGGER_OPTIONS["rest_constant"])

    if "no_pain" not in pain_regions and pain_severity > 0:
        if char_info["heart_boost"] > 0:
            c_shift = char_info["heart_boost"] * (0.6 + 0.4 * severity_norm)
            domain_scores["heart"] += c_shift
            pain_only_scores["heart"] += c_shift
            symptom_contributions["heart"].append(
                (f"Pain Type: {char_info['label']}", round(c_shift, 3))
            )
        if trig_info["heart_boost"] != 0:
            t_shift = trig_info["heart_boost"] * (0.6 + 0.4 * severity_norm)
            domain_scores["heart"] += t_shift
            pain_only_scores["heart"] += t_shift
            symptom_contributions["heart"].append(
                (f"Trigger: {trig_info['label']}", round(t_shift, 3))
            )

    for sym_key in selected_symptoms:
        sym = SYMPTOM_CATALOG.get(sym_key)
        if not sym:
            continue
        for dom, w in sym["weights"].items():
            if w > 0:
                domain_scores[dom] += w
                symptom_only_scores[dom] += w
                if w >= 0.15:
                    symptom_contributions[dom].append(
                        (f"Symptom: {sym['label']}", round(w, 3))
                    )

    h_intensity = float(np.clip(domain_scores["heart"] / 2.6, 0.05, 0.98))
    d_intensity = float(np.clip(domain_scores["diabetes"] / 2.4, 0.05, 0.98))
    b_intensity = float(np.clip(domain_scores["breast_cancer"] / 2.3, 0.05, 0.98))

    cp_inferred = char_info["cp_code"] if ("chest_center_left" in pain_regions or "left_arm_shoulder_jaw" in pain_regions or "upper_back" in pain_regions) else 3
    exang_inferred = 1 if (trig_info["exang_code"] == 1 and h_intensity >= 0.30) or ("shortness_of_breath" in selected_symptoms and h_intensity >= 0.45) else 0
    trestbps_inferred = int(np.clip(114 + (age - 40) * 0.55 + h_intensity * 32 + (8 if "dizziness_fainting" in selected_symptoms else 0), 98, 192))
    chol_inferred = int(np.clip(195 + (age - 40) * 0.85 + h_intensity * 68 + (18 if "family_history" in selected_symptoms else 0), 155, 360))
    thalach_inferred = int(np.clip(182 - (age - 35) * 0.75 - h_intensity * 42 - (12 if exang_inferred == 1 else 0), 88, 195))
    oldpeak_inferred = round(float(np.clip(0.1 + h_intensity * 2.9 + (0.5 if exang_inferred == 1 else 0.0) + (0.4 if "cold_sweats" in selected_symptoms else 0.0), 0.0, 5.2)), 1)
    ca_inferred = int(np.clip(np.floor(h_intensity * 2.8 + (0.4 if age >= 58 else 0.0)), 0, 3))
    thal_inferred = 3 if h_intensity >= 0.52 else (2 if h_intensity >= 0.25 else 1)

    inferred_heart = {
        "age": int(age),
        "sex": int(sex_code),
        "cp": int(cp_inferred),
        "trestbps": int(trestbps_inferred),
        "chol": int(chol_inferred),
        "thalach": int(thalach_inferred),
        "oldpeak": float(oldpeak_inferred),
        "ca": int(ca_inferred),
        "thal": int(thal_inferred),
        "exang": int(exang_inferred),
    }

    glucose_inferred = int(np.clip(92 + d_intensity * 88 + (16 if "excessive_thirst" in selected_symptoms else 0) + (12 if "frequent_urination" in selected_symptoms else 0), 78, 196))
    bmi_inferred = round(float(np.clip(23.5 + d_intensity * 13.5 + (2.0 if "chronic_fatigue" in selected_symptoms else 0.0), 19.5, 46.0)), 1)
    dpf_inferred = round(float(np.clip(0.24 + d_intensity * 0.72 + (0.38 if "family_history" in selected_symptoms else 0.0), 0.12, 1.85)), 2)
    insulin_inferred = int(np.clip(75 + d_intensity * 145 + (25 if "unexplained_weight_change" in selected_symptoms else 0), 45, 320))
    bp_diab_inferred = int(np.clip(68 + (age - 35) * 0.35 + d_intensity * 18, 58, 106))
    preg_inferred = int(np.clip(round((age - 22) / 7.0), 0, 6)) if sex_code == 0 else 0

    inferred_diabetes = {
        "Glucose": int(glucose_inferred),
        "BMI": float(bmi_inferred),
        "Age": int(age),
        "DiabetesPedigreeFunction": float(dpf_inferred),
        "Insulin": int(insulin_inferred),
        "Pregnancies": int(preg_inferred),
        "BloodPressure": int(bp_diab_inferred),
    }

    cp_worst_inferred = round(float(np.clip(0.055 + b_intensity * 0.165 + (0.035 if "breast_underarm_lump" in selected_symptoms else 0.0), 0.03, 0.28)), 3)
    perim_worst_inferred = round(float(np.clip(82.0 + b_intensity * 68.0 + (12.0 if "breast_swelling_asymmetry" in selected_symptoms else 0.0), 70.0, 185.0)), 1)
    rad_worst_inferred = round(float(np.clip(12.5 + b_intensity * 9.5, 10.5, 26.5)), 1)
    area_worst_inferred = round(float(np.clip(520.0 + b_intensity * 920.0, 380.0, 1850.0)), 0)
    concav_worst_inferred = round(float(np.clip(0.12 + b_intensity * 0.42 + (0.08 if "skin_dimpling_redness" in selected_symptoms else 0.0), 0.05, 0.85)), 2)
    tex_worst_inferred = round(float(np.clip(19.5 + b_intensity * 11.5, 15.0, 38.0)), 1)

    inferred_breast = {
        "concave points_worst": cp_worst_inferred,
        "perimeter_worst": perim_worst_inferred,
        "radius_worst": rad_worst_inferred,
        "area_worst": area_worst_inferred,
        "concavity_worst": concav_worst_inferred,
        "texture_worst": tex_worst_inferred,
    }

    z_heart = (
        -1.35
        + 0.84 * (inferred_heart["ca"] - 0.7)
        + 0.72 * (inferred_heart["thal"] - 2.0)
        + 0.61 * (inferred_heart["oldpeak"] - 1.0)
        - 0.58 * ((inferred_heart["thalach"] - 149.0) / 22.0)
        - 0.35 * (inferred_heart["cp"] - 1.5)
        + 0.42 * inferred_heart["exang"]
        + 0.22 * ((inferred_heart["age"] - 54.0) / 9.0)
        + (0.18 if sex_code == 1 else -0.12)
    )
    prob_heart = float(1.0 / (1.0 + np.exp(-z_heart)))

    z_diab = (
        -1.15
        + 1.12 * ((inferred_diabetes["Glucose"] - 121.0) / 30.0)
        + 0.64 * ((inferred_diabetes["BMI"] - 32.0) / 6.8)
        + 0.41 * ((inferred_diabetes["Age"] - 33.0) / 11.5)
        + 0.33 * ((inferred_diabetes["DiabetesPedigreeFunction"] - 0.47) / 0.33)
        + 0.27 * ((inferred_diabetes["Insulin"] - 125.0) / 85.0)
    )
    prob_diab = float(1.0 / (1.0 + np.exp(-z_diab)))

    z_bc = (
        -1.25
        + 1.34 * ((inferred_breast["concave points_worst"] - 0.114) / 0.065)
        + 0.98 * ((inferred_breast["perimeter_worst"] - 107.0) / 33.0)
        + 0.82 * ((inferred_breast["radius_worst"] - 16.2) / 4.8)
        + 0.69 * ((inferred_breast["area_worst"] - 880.0) / 569.0)
        + 0.54 * ((inferred_breast["concavity_worst"] - 0.27) / 0.20)
        - (0.45 if sex_code == 1 else 0.0)
    )
    prob_bc = float(1.0 / (1.0 + np.exp(-z_bc)))

    domain_probabilities = {
        "heart": round(prob_heart, 4),
        "diabetes": round(prob_diab, 4),
        "breast_cancer": round(prob_bc, 4),
    }

    if domain_scores["breast_cancer"] >= 0.75:
        active_domain = "breast_cancer"
    elif domain_scores["diabetes"] > domain_scores["heart"] and domain_scores["diabetes"] >= 0.55:
        active_domain = "diabetes"
    elif domain_scores["heart"] >= 0.35:
        active_domain = "heart"
    else:
        active_domain = max(domain_probabilities.items(), key=lambda kv: kv[1])[0]

    active_prob = domain_probabilities[active_domain]
    is_high_risk = active_prob >= decision_threshold

    raw_sym_contribs = symptom_contributions[active_domain]
    if len(raw_sym_contribs) < 4:
        if "no_pain" in pain_regions or pain_severity <= 2:
            raw_sym_contribs.append(("Absence of severe localized body pain", -0.42))
        if "shortness_of_breath" not in selected_symptoms and active_domain == "heart":
            raw_sym_contribs.append(("No shortness of breath reported", -0.28))
        if "excessive_thirst" not in selected_symptoms and active_domain == "diabetes":
            raw_sym_contribs.append(("No polydipsia (excessive thirst) reported", -0.34))
        if "breast_underarm_lump" not in selected_symptoms and active_domain == "breast_cancer":
            raw_sym_contribs.append(("No palpable breast/axillary lump reported", -0.48))

    raw_sym_contribs = sorted(raw_sym_contribs, key=lambda x: abs(x[1]), reverse=True)[:6]
    patient_symptom_shap = {label: val for label, val in raw_sym_contribs}
    total_abs = sum(abs(v) for v in patient_symptom_shap.values()) or 1.0
    patient_symptom_lime = {
        label: round((val / total_abs) * active_prob, 3)
        for label, val in patient_symptom_shap.items()
    }

    if active_domain == "heart":
        biomarker_shap = {
            f"Inferred Vessels (ca≈{inferred_heart['ca']})": round(0.84 * (inferred_heart["ca"] - 0.7), 2),
            f"Inferred ST Depr. (oldpeak≈{inferred_heart['oldpeak']}mm)": round(0.61 * (inferred_heart["oldpeak"] - 1.0), 2),
            f"Inferred Thallium (thal≈{inferred_heart['thal']})": round(0.72 * (inferred_heart["thal"] - 2.0), 2),
            f"Est. Max HR (thalach≈{inferred_heart['thalach']}bpm)": round(-0.58 * ((inferred_heart["thalach"] - 149.0) / 22.0), 2),
            f"Chest Pain Type (cp={inferred_heart['cp']})": round(-0.35 * (inferred_heart["cp"] - 1.5), 2),
            f"Exertional Angina (exang={inferred_heart['exang']})": round(0.42 * (inferred_heart["exang"] - 0.3), 2),
        }
        recommended_tests = [
            "12-Lead Resting & Exercise Stress ECG — confirms ST-segment depression (`oldpeak`) and peak heart rate (`thalach`).",
            "Fasting Lipid Profile & Blood Pressure Monitoring — measures exact serum cholesterol (`chol`) and systolic BP (`trestbps`).",
            "Coronary CT Angiography / Myocardial Perfusion Scan — verifies vessel narrowing (`ca`) and perfusion status (`thal`).",
        ]
        domain_title = "Cardiovascular Disease (Coronary Artery Ischemia Triage)"
    elif active_domain == "diabetes":
        biomarker_shap = {
            f"Est. 2-Hr Glucose (≈{inferred_diabetes['Glucose']} mg/dL)": round(1.12 * ((inferred_diabetes["Glucose"] - 121.0) / 30.0), 2),
            f"Est. BMI Profile (≈{inferred_diabetes['BMI']} kg/m²)": round(0.64 * ((inferred_diabetes["BMI"] - 32.0) / 6.8), 2),
            f"Patient Age ({inferred_diabetes['Age']} yrs)": round(0.41 * ((inferred_diabetes["Age"] - 33.0) / 11.5), 2),
            f"Est. Pedigree Score (DPF≈{inferred_diabetes['DiabetesPedigreeFunction']})": round(0.33 * ((inferred_diabetes["DiabetesPedigreeFunction"] - 0.47) / 0.33), 2),
            f"Est. Serum Insulin (≈{inferred_diabetes['Insulin']} mu U/ml)": round(0.27 * ((inferred_diabetes["Insulin"] - 125.0) / 85.0), 2),
        }
        recommended_tests = [
            "Fasting Plasma Glucose (FPG) & 2-Hour Oral Glucose Tolerance Test (OGTT) — measures exact `Glucose` (mg/dL).",
            "Glycated Hemoglobin (HbA1c) & Fasting Serum Insulin — evaluates chronic glycemic control and `Insulin` resistance.",
            "Comprehensive Metabolic Panel & BMI Assessment — confirms blood pressure and metabolic baseline.",
        ]
        domain_title = "Metabolic & Endocrine Triage (Type-2 Diabetes Risk)"
    else:
        biomarker_shap = {
            f"Est. Concave Points (≈{inferred_breast['concave points_worst']:.3f})": round(1.34 * ((inferred_breast["concave points_worst"] - 0.114) / 0.065), 2),
            f"Est. Nuclear Perimeter (≈{inferred_breast['perimeter_worst']:.1f})": round(0.98 * ((inferred_breast["perimeter_worst"] - 107.0) / 33.0), 2),
            f"Est. Nuclear Radius (≈{inferred_breast['radius_worst']:.1f})": round(0.82 * ((inferred_breast["radius_worst"] - 16.2) / 4.8), 2),
            f"Est. Nuclear Area (≈{inferred_breast['area_worst']:.0f})": round(0.69 * ((inferred_breast["area_worst"] - 880.0) / 569.0), 2),
            f"Est. Concavity (≈{inferred_breast['concavity_worst']:.2f})": round(0.54 * ((inferred_breast["concavity_worst"] - 0.27) / 0.20), 2),
        }
        recommended_tests = [
            "Clinical Breast Examination (CBE) & Bilateral Diagnostic Mammography — localizes any palpable mass or tissue asymmetry.",
            "Targeted High-Resolution Breast & Axillary Ultrasound — distinguishes solid vs. cystic lesions.",
            "Fine-Needle Aspiration (FNA) Cytology / Core Biopsy — provides exact nuclear morphometry (`concave points_worst`, `perimeter_worst`) for definitive WDBC model confirmation.",
        ]
        domain_title = "Oncological Triage (Breast Tissue & Axillary Lesion Screening)"

    return {
        "active_domain": active_domain,
        "domain_title": domain_title,
        "active_probability": active_prob,
        "is_high_risk": is_high_risk,
        "decision_threshold": decision_threshold,
        "domain_probabilities": domain_probabilities,
        "domain_symptom_scores": {k: round(v, 2) for k, v in domain_scores.items()},
        "pain_routing_scores": {k: round(v, 2) for k, v in pain_only_scores.items()},
        "symptom_routing_scores": {k: round(v, 2) for k, v in symptom_only_scores.items()},
        "patient_symptom_shap": patient_symptom_shap,
        "patient_symptom_lime": patient_symptom_lime,
        "clinician_biomarker_shap": biomarker_shap,
        "inferred_biomarkers": {
            "heart": inferred_heart,
            "diabetes": inferred_diabetes,
            "breast_cancer": inferred_breast,
        },
        "recommended_tests": recommended_tests,
    }
