"""
Data Cleaning, Feature Engineering, and Preprocessing Pipeline
for House Price Prediction System.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer that performs:
    1. Imputation of domain-specific missing values (NA means 'None' vs median/mode).
    2. Feature engineering (TotalSF, TotalBath, HouseAge, RemodelAge, etc.).
    3. Handles both full train/test data and partial single-sample user inputs.
    """

    def __init__(self):
        # Learned parameters during fit
        self.median_values_ = {}
        self.mode_values_ = {}
        self.neighborhood_lotfrontage_ = {}
        self.feature_columns_ = []
        self.numeric_features_ = []
        self.categorical_features_ = []

    def fit(self, X, y=None):
        df = X.copy()
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])
        if "SalePrice" in df.columns:
            df = df.drop(columns=["SalePrice"])

        # Compute Neighborhood-based median LotFrontage
        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            self.neighborhood_lotfrontage_ = (
                df.groupby("Neighborhood")["LotFrontage"].median().to_dict()
            )
            overall_lf_median = df["LotFrontage"].median()
            self.median_values_["LotFrontage"] = overall_lf_median

        # Numerical medians
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.median_values_[col] = float(df[col].median(skipna=True))

        # Categorical modes
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        for col in cat_cols:
            mode_series = df[col].mode(dropna=True)
            self.mode_values_[col] = mode_series.iloc[0] if not mode_series.empty else "None"

        # Fit engineered features on sample to determine column names
        transformed_df = self._engineer_features(df)
        self.feature_columns_ = transformed_df.columns.tolist()
        self.numeric_features_ = transformed_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = [
            c for c in self.feature_columns_ if c not in self.numeric_features_
        ]

        return self

    def _engineer_features(self, data):
        df = data.copy()

        # Fill domain-specific "NA as None" categories
        none_categories = [
            "Alley", "BsmtQual", "BsmtCond", "BsmtExposure",
            "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "PoolQC", "Fence", "MiscFeature", "MasVnrType"
        ]
        for col in none_categories:
            if col in df.columns:
                df[col] = df[col].fillna("None").astype(str)

        # Fill domain-specific numerical zeroes (features where NA means 0 area/count)
        zero_cols = [
            "GarageYrBlt", "GarageArea", "GarageCars",
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
            "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
        ]
        for col in zero_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(float)

        # LotFrontage by Neighborhood median
        if "LotFrontage" in df.columns:
            if "Neighborhood" in df.columns and self.neighborhood_lotfrontage_:
                df["LotFrontage"] = df.apply(
                    lambda r: self.neighborhood_lotfrontage_.get(
                        r["Neighborhood"], self.median_values_.get("LotFrontage", 69.0)
                    )
                    if pd.isna(r["LotFrontage"])
                    else r["LotFrontage"],
                    axis=1,
                )
            else:
                df["LotFrontage"] = df["LotFrontage"].fillna(
                    self.median_values_.get("LotFrontage", 69.0)
                )

        # Impute remaining numerical columns with median
        for col, med in self.median_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        # Impute remaining categorical columns with mode
        for col, mode_val in self.mode_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val).astype(str)

        # Feature Engineering calculations
        # 1. Total Square Footage
        bsmt_sf = df.get("TotalBsmtSF", 0).fillna(0) if hasattr(df.get("TotalBsmtSF", 0), "fillna") else df.get("TotalBsmtSF", 0)
        first_flr = df.get("1stFlrSF", 0).fillna(0) if hasattr(df.get("1stFlrSF", 0), "fillna") else df.get("1stFlrSF", 0)
        sec_flr = df.get("2ndFlrSF", 0).fillna(0) if hasattr(df.get("2ndFlrSF", 0), "fillna") else df.get("2ndFlrSF", 0)
        df["TotalSF"] = bsmt_sf + first_flr + sec_flr

        # 2. Total Bathrooms
        full_bath = df.get("FullBath", 0)
        half_bath = df.get("HalfBath", 0)
        bsmt_full = df.get("BsmtFullBath", 0)
        bsmt_half = df.get("BsmtHalfBath", 0)
        df["TotalBath"] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

        # 3. Ages and Remodel info
        yr_sold = df.get("YrSold", 2010)
        yr_built = df.get("YearBuilt", 1970)
        yr_remod = df.get("YearRemodAdd", yr_built)
        df["HouseAge"] = (yr_sold - yr_built).clip(lower=0)
        df["RemodelAge"] = (yr_sold - yr_remod).clip(lower=0)
        df["IsRemodeled"] = (yr_remod != yr_built).astype(int)

        # 4. Total Porch Square Footage
        wood_deck = df.get("WoodDeckSF", 0)
        open_porch = df.get("OpenPorchSF", 0)
        enc_porch = df.get("EnclosedPorch", 0)
        three_ssn = df.get("3SsnPorch", 0)
        screen_porch = df.get("ScreenPorch", 0)
        df["TotalPorchSF"] = wood_deck + open_porch + enc_porch + three_ssn + screen_porch

        # 5. Combined Quality Score
        qual = df.get("OverallQual", 5)
        cond = df.get("OverallCond", 5)
        df["OverallScore"] = qual * cond

        # 6. Binary feature flags
        df["HasGarage"] = (df.get("GarageCars", 0) > 0).astype(int)
        df["HasBsmt"] = (df.get("TotalBsmtSF", 0) > 0).astype(int)
        df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)
        df["HasPool"] = (df.get("PoolArea", 0) > 0).astype(int)
        df["Has2ndFlr"] = (df.get("2ndFlrSF", 0) > 0).astype(int)

        # Convert MSSubClass to string (categorical concept)
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)

        # Drop Id column if present
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])
        if "SalePrice" in df.columns:
            df = df.drop(columns=["SalePrice"])

        return df

    def transform(self, X):
        df = X.copy()
        # If any columns are missing (e.g. inference with partial fields), fill defaults
        for col, med in self.median_values_.items():
            if col not in df.columns:
                df[col] = med
        for col, mode_val in self.mode_values_.items():
            if col not in df.columns:
                df[col] = mode_val

        transformed = self._engineer_features(df)
        
        # Ensure identical column order
        missing_cols = [c for c in self.feature_columns_ if c not in transformed.columns]
        for c in missing_cols:
            transformed[c] = 0
            
        return transformed[self.feature_columns_]


def create_preprocessor(numeric_features, categorical_features):
    """
    Builds a ColumnTransformer that applies RobustScaler to numeric columns
    and OneHotEncoder (ignoring unseen categories) to categorical columns.
    """
    return ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def load_and_clean_data(train_csv_path="train.csv", remove_outliers=True):
    """
    Loads training data, applies outlier removal recommended for Ames housing.
    Returns X (DataFrame), y (pd.Series), and y_log (pd.Series).
    """
    df = pd.read_csv(train_csv_path)

    if remove_outliers:
        # Standard Ames outlier recommendation: GrLivArea > 4000 and SalePrice < 300000
        outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
        df = df[~outlier_mask].reset_index(drop=True)

    y = df["SalePrice"]
    y_log = np.log1p(y)
    X = df.drop(columns=["SalePrice"])

    return X, y, y_log
