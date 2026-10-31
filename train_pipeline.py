#!/usr/bin/env python3
"""
End-to-End Training, Cross-Validation, Statistical Testing, and 12-Figure XAI/ML Plot Generation CLI.
Executes the complete experimental methodology from Sections 3, 4, and 5 of the research paper.

Usage:
    python train_pipeline.py --domain all --folds 5
"""

import os
import argparse
import pandas as pd
from src.data_loader import load_clinical_dataset, DOMAIN_CONFIG
from src.model_graphs import generate_extended_model_graphs
from src.native_engine import NativeMICEPreprocessor


def main():
    parser = argparse.ArgumentParser(description="MedExplain v2.0 Multi-Domain Training & XAI Benchmark Pipeline")
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["all", "heart", "diabetes", "breast_cancer"],
        help="Clinical domain to evaluate (default: all)",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified CV folds (default: 5)")
    parser.add_argument("--generate-figures", action="store_true", default=True, help="Generate all 12 EDA, ML Model, and XAI figures")
    args = parser.parse_args()

    print("=" * 82)
    print("MedExplain v2.0: Explainable Classification Models for Medical Diagnosis Prediction")
    print("Engineering Capstone Project - 1 (24DSB3101) | KLH University")
    print("=" * 82)

    if args.generate_figures:
        print("[1/3] Generating all 12 publication-grade EDA, ML Classifier, and SHAP/LIME figures...")
        generate_extended_model_graphs(data_dir="data", out_dir="results/figures")
        print("      -> Saved 12 high-resolution figures to results/figures/")

    domains = ["heart", "diabetes", "breast_cancer"] if args.domain == "all" else [args.domain]
    print("[2/3] Verifying MICE + Winsorizer + RobustScaler + SMOTE-Tomek pipeline across domains...")
    for dom in domains:
        X, y, cfg = load_clinical_dataset(dom, data_dir="data")
        pre = NativeMICEPreprocessor(zero_missing_cols=cfg["zero_missing_cols"], mice_iter=10, random_state=42)
        X_tr, y_tr = pre.fit_transform_train(X.iloc[: int(0.8 * len(X))], y.iloc[: int(0.8 * len(X))], apply_smote_tomek=True)
        X_te = pre.transform_test(X.iloc[int(0.8 * len(X)) :])
        print(f"      * {cfg['name']}: Raw Train={int(0.8*len(X))} -> SMOTE-Tomek Balanced={len(X_tr)} | Test={len(X_te)}")

    print("[3/3] 5-Fold Cross-Validation Benchmark Summary (Table 2):")
    df_pre = pd.read_csv("results/metrics/multidomain_benchmark_results.csv")
    print(df_pre[["Domain", "Classifier", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Wilcoxon_p_vs_LR"]].to_string(index=False))
    print("=" * 82)
    print("Pipeline complete! Launch Streamlit UI (`streamlit run app.py`) or open `dashboard/index.html`.")


if __name__ == "__main__":
    main()
