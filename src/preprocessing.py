"""
Standardized Clinical Preprocessing Pipeline (Section 3.3 of Paper):
1. Missing Value Imputation: Biologically implausible zero entries replaced with NaN and imputed
   via Multivariate Imputation by Chained Equations (MICE) using BayesianRidge over 10 iterations.
2. Outlier Capping (Winsorization): Extreme values capped at 1st and 99th percentiles:
   x_capped = min(max(x, Q_0.01), Q_0.99)
3. Robust Normalization: Median and Interquartile Range (IQR = Q_75 - Q_25) scaling:
   x_scaled = (x - median(x)) / (Q_75 - Q_25)
4. Class Rebalancing: SMOTE-Tomek applied exclusively on training folds during cross-validation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTETomek


class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps extreme physiological outliers at the 1st and 99th percentiles."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_arr, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X_arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


class ClinicalPreprocessor:
    """
    End-to-end clinical tabular preprocessor implementing MICE imputation,
    1st/99th percentile Winsorization, Robust IQR scaling, and training-only SMOTE-Tomek.
    """

    def __init__(
        self,
        zero_missing_cols: Optional[List[str]] = None,
        mice_iterations: int = 10,
        random_state: int = 42,
    ):
        self.zero_missing_cols = zero_missing_cols or []
        self.mice_iterations = mice_iterations
        self.random_state = random_state

        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.mice_iterations,
            random_state=self.random_state,
            initial_strategy="median",
        )
        self.winsorizer = Winsorizer(lower_quantile=0.01, upper_quantile=0.99)
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.feature_names_: List[str] = []

    def _replace_biologically_implausible_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in self.zero_missing_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace(0, np.nan)
        return df_clean

    def fit_transform_train(
        self, X_train: pd.DataFrame, y_train: pd.Series, apply_smote_tomek: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fits MICE imputer, Winsorizer, and RobustScaler exclusively on training data,
        and optionally applies SMOTE-Tomek rebalancing to the training fold.
        """
        self.feature_names_ = list(X_train.columns)
        X_clean = self._replace_biologically_implausible_zeros(X_train)

        X_imp = self.imputer.fit_transform(X_clean)
        X_cap = self.winsorizer.fit_transform(X_imp)
        X_scl = self.scaler.fit_transform(X_cap)

        X_scl_df = pd.DataFrame(X_scl, columns=self.feature_names_)
        y_res = y_train.reset_index(drop=True)

        if apply_smote_tomek:
            smote_tomek = SMOTETomek(random_state=self.random_state)
            X_res, y_res = smote_tomek.fit_resample(X_scl_df, y_res)
            return pd.DataFrame(X_res, columns=self.feature_names_), pd.Series(y_res)

        return X_scl_df, y_res

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fitted MICE imputer, Winsorizer, and RobustScaler to held-out test data
        without data leakage or synthetic resampling.
        """
        X_clean = self._replace_biologically_implausible_zeros(X_test)
        X_imp = self.imputer.transform(X_clean)
        X_cap = self.winsorizer.transform(X_imp)
        X_scl = self.scaler.transform(X_cap)
        return pd.DataFrame(X_scl, columns=self.feature_names_, index=X_test.index)
