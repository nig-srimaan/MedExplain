"""
Dual-Layer Explainable Artificial Intelligence (XAI) Framework (Sections 3.6 & 5.4 of Paper):
1. Global Explainability: TreeSHAP & KernelSHAP cohort-wide feature attribution (E[|phi_i|]).
2. Local Explainability: Patient-specific SHAP Waterfall plots and Tabular LIME linear surrogates.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    import lime.lime_tabular
except ImportError:
    lime = None


class MedExplainXAI:
    """
    Unified SHAP + LIME explainability wrapper for tree ensembles and linear classifiers.
    """

    def __init__(self, model: Any, X_background: pd.DataFrame, feature_names: Optional[list] = None):
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names or list(X_background.columns)
        self.shap_explainer = None
        self.lime_explainer = None
        self._init_explainers()

    def _init_explainers(self):
        if shap is not None:
            model_type = type(self.model).__name__
            if any(k in model_type for k in ["XGB", "CatBoost", "RandomForest", "DecisionTree"]):
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                bg_sample = shap.sample(self.X_background, min(100, len(self.X_background)))
                self.shap_explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1], bg_sample
                )

        if lime is not None:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.asarray(self.X_background),
                feature_names=self.feature_names,
                class_names=["Negative (Control)", "Positive (Diagnosis)"],
                mode="classification",
                random_state=42,
            )

    def compute_global_shap_importance(self, X_eval: pd.DataFrame) -> pd.DataFrame:
        """
        Computes mean absolute Shapley values E[|phi_i|] across the evaluation cohort.
        """
        if self.shap_explainer is None:
            raise RuntimeError("SHAP library is not installed.")

        shap_values = self.shap_explainer.shap_values(X_eval)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(np.shape(shap_values)) == 3:
            shap_values = shap_values[:, :, 1]

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs_shap,
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        return df_imp

    def explain_patient_local(
        self, patient_row: pd.Series, num_features: int = 8
    ) -> Dict[str, Any]:
        """
        Generates both local TreeSHAP/KernelSHAP attributions and LIME surrogate weights
        for an individual patient record.
        """
        x_df = pd.DataFrame([patient_row.values], columns=self.feature_names)
        prob_pos = float(self.model.predict_proba(x_df)[0, 1])

        result: Dict[str, Any] = {
            "predicted_probability": prob_pos,
            "predicted_class": int(prob_pos >= 0.5),
            "shap_contributions": {},
            "lime_contributions": [],
        }

        if self.shap_explainer is not None:
            sv = self.shap_explainer.shap_values(x_df)
            if isinstance(sv, list):
                sv = sv[1][0]
            elif len(np.shape(sv)) == 3:
                sv = sv[0, :, 1]
            else:
                sv = sv[0]
            for feat, val in zip(self.feature_names, sv):
                result["shap_contributions"][feat] = float(val)

        if self.lime_explainer is not None:
            exp = self.lime_explainer.explain_instance(
                data_row=np.asarray(patient_row.values, dtype=float),
                predict_fn=self.model.predict_proba,
                num_features=num_features,
            )
            result["lime_contributions"] = exp.as_list()

        return result
