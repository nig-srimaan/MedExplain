"""
Automated Unit Tests for MedExplain v3.0 datasets, MICE + SMOTE-Tomek preprocessor,
Patient Symptom & Body Pain Triage Engine (`src/symptom_mapper.py`),
12-figure ML model graph suite, and benchmark metrics.
"""

import os
import unittest
import pandas as pd
from src.data_loader import load_clinical_dataset
from src.native_engine import NativeMICEPreprocessor
from src.native_engine import evaluate_patient_symptoms


class TestMedExplainV3Repository(unittest.TestCase):

    def test_datasets_exist_and_match_paper_dimensions(self):
        df_heart = pd.read_csv("data/cleveland_heart_disease.csv")
        df_diab = pd.read_csv("data/pima_indians_diabetes.csv")
        df_wdbc = pd.read_csv("data/wisconsin_breast_cancer.csv")

        self.assertEqual(len(df_heart), 303)
        self.assertEqual(df_heart.shape[1], 14)

        self.assertEqual(len(df_diab), 768)
        self.assertEqual(df_diab.shape[1], 9)

        self.assertEqual(len(df_wdbc), 569)
        self.assertEqual(df_wdbc.shape[1], 31)

    def test_native_mice_and_smote_tomek_preprocessor(self):
        X, y, cfg = load_clinical_dataset("diabetes", data_dir="data")
        pre = NativeMICEPreprocessor(zero_missing_cols=cfg["zero_missing_cols"], mice_iter=5, random_state=42)
        X_tr, y_tr = pre.fit_transform_train(X.iloc[:600], y.iloc[:600], apply_smote_tomek=True)
        X_te = pre.transform_test(X.iloc[600:])
        self.assertFalse(X_tr.isna().any().any())
        self.assertFalse(X_te.isna().any().any())
        self.assertEqual(X_tr.shape[1], 8)
        self.assertEqual(len(X_te), 168)

    def test_patient_symptom_and_body_pain_triage_engine(self):
        # 1. Cardiac presentation: 58yo Male with Chest & Left Arm pain + dyspnea + sweating
        res_heart = evaluate_patient_symptoms(
            age=58,
            sex="Male",
            pain_regions=["chest_center_left", "left_arm_shoulder_jaw"],
            pain_character="crushing_pressure",
            pain_severity=8,
            pain_trigger="exertion",
            selected_symptoms=["shortness_of_breath", "cold_sweats", "radiating_arm_jaw"],
        )
        self.assertEqual(res_heart["active_domain"], "heart")
        self.assertTrue(res_heart["is_high_risk"])
        self.assertGreater(res_heart["active_probability"], 0.65)
        self.assertGreaterEqual(len(res_heart["patient_symptom_shap"]), 3)
        self.assertGreaterEqual(len(res_heart["recommended_tests"]), 2)

        # 2. Metabolic presentation: 46yo Female with thirst, urination, tingling feet
        res_diab = evaluate_patient_symptoms(
            age=46,
            sex="Female",
            pain_regions=["legs_feet", "abdomen_stomach"],
            pain_character="burning_aching",
            pain_severity=5,
            pain_trigger="rest_constant",
            selected_symptoms=["excessive_thirst", "frequent_urination", "numbness_tingling", "blurred_vision"],
        )
        self.assertEqual(res_diab["active_domain"], "diabetes")
        self.assertTrue(res_diab["is_high_risk"])

        # 3. Oncology presentation: 51yo Female with breast/underarm lump & skin dimpling
        res_bc = evaluate_patient_symptoms(
            age=51,
            sex="Female",
            pain_regions=["breast_axilla"],
            pain_character="tender_throbbing",
            pain_severity=6,
            pain_trigger="touch_pressure",
            selected_symptoms=["breast_underarm_lump", "skin_dimpling_redness", "breast_swelling_asymmetry"],
        )
        self.assertEqual(res_bc["active_domain"], "breast_cancer")
        self.assertTrue(res_bc["is_high_risk"])

        # 4. Healthy baseline: 31yo Female with no pain and no symptoms
        res_healthy = evaluate_patient_symptoms(
            age=31,
            sex="Female",
            pain_regions=["no_pain"],
            pain_character="none",
            pain_severity=0,
            pain_trigger="rest_constant",
            selected_symptoms=[],
        )
        self.assertFalse(res_healthy["is_high_risk"])

    def test_all_12_ml_model_figures_exist(self):
        expected_figs = [
            "fig1_multidomain_eda.png",
            "fig2_model_benchmark_comparison.png",
            "fig3_global_shap_rankings.png",
            "fig4_patient42_local_xai.png",
            "fig5_multimodel_roc_curves.png",
            "fig6_precision_recall_curves.png",
            "fig7_confusion_matrices_grid.png",
            "fig8_calibration_reliability_curves.png",
            "fig9_threshold_sensitivity_tradeoff.png",
            "fig10_learning_curves_and_optuna.png",
            "fig11_model_radar_and_latency.png",
            "fig12_correlation_heatmaps.png",
        ]
        for fname in expected_figs:
            fpath = os.path.join("results", "figures", fname)
            self.assertTrue(os.path.exists(fpath), f"Missing figure: {fpath}")
            self.assertGreater(os.path.getsize(fpath), 1000)


if __name__ == "__main__":
    unittest.main()
