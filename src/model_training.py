"""
Model Training, Cross-Validation, Evaluation, and Serialization Suite
for House Price Prediction System.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
import joblib

from src.data_preprocessing import (
    HousingFeatureEngineer,
    create_preprocessor,
    load_and_clean_data,
)


def get_candidate_models():
    """
    Returns a dictionary of candidate regression models.
    """
    return {
        "Ridge": Ridge(alpha=10.0, random_state=42),
        "Lasso": Lasso(alpha=0.0005, max_iter=10000, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=10000, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=350, learning_rate=0.04, max_depth=3, subsample=0.8, random_state=42
        ),
        "EnsembleBlend": VotingRegressor(
            estimators=[
                ("ridge", Ridge(alpha=10.0, random_state=42)),
                ("lasso", Lasso(alpha=0.0005, max_iter=10000, random_state=42)),
                (
                    "gbr",
                    GradientBoostingRegressor(
                        n_estimators=350, learning_rate=0.04, max_depth=3, subsample=0.8, random_state=42
                    ),
                ),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=150, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1
                    ),
                ),
            ],
            weights=[0.25, 0.25, 0.35, 0.15],
        ),
    }


def evaluate_models(X, y, y_log, feature_engineer, preprocessor, cv_splits=5):
    """
    Cross-validates all candidate models, calculates RMSLE, RMSE ($), MAE ($), and R2.
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    models = get_candidate_models()
    metrics = {}

    # Pre-transform features for rapid cross-validation
    print("[1/5] Engineering features and fitting preprocessor...")
    X_eng = feature_engineer.fit_transform(X)
    X_proc = preprocessor.fit_transform(X_eng)

    print(f"[2/5] Running {cv_splits}-Fold Cross-Validation on {len(models)} models...")
    oof_predictions = {}

    for name, model in models.items():
        print(f"   -> Evaluating {name}...")
        # Out-of-fold log predictions
        pred_log = cross_val_predict(model, X_proc, y_log, cv=kf, n_jobs=-1)
        # Convert back to actual dollar scale: exp(x) - 1
        pred_dollar = np.expm1(pred_log)

        # Calculate metrics
        rmsle = float(np.sqrt(mean_squared_error(y_log, pred_log)))
        rmse_dollar = float(np.sqrt(mean_squared_error(y, pred_dollar)))
        mae_dollar = float(mean_absolute_error(y, pred_dollar))
        r2 = float(r2_score(y, pred_dollar))

        metrics[name] = {
            "RMSLE": round(rmsle, 4),
            "RMSE_Dollar": round(rmse_dollar, 2),
            "MAE_Dollar": round(mae_dollar, 2),
            "R2_Score": round(r2, 4),
        }
        oof_predictions[name] = pred_dollar

    return metrics, oof_predictions, X_proc, y.values


def plot_visualizations(metrics, oof_predictions, y_actual, feature_engineer, preprocessor, best_model, output_dir="static/plots"):
    """
    Generates high-resolution, visually polished evaluation charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Model Comparison Chart
    df_metrics = pd.DataFrame(metrics).T
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R2 Score comparison
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#06b6d4", "#ec4899"]
    bars1 = axes[0].bar(df_metrics.index, df_metrics["R2_Score"], color=colors[:len(df_metrics)], width=0.55)
    axes[0].set_title("Model Comparison: R² Score (Higher is Better)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].set_ylim([max(0.8, df_metrics["R2_Score"].min() - 0.05), 1.0])
    axes[0].tick_params(axis="x", rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # RMSLE comparison
    bars2 = axes[1].bar(df_metrics.index, df_metrics["RMSLE"], color=colors[:len(df_metrics)], width=0.55)
    axes[1].set_title("Model Comparison: RMSLE (Lower is Better)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Log Error (RMSLE)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=25)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"Saved: {comparison_path}")

    # 2. Actual vs Predicted Plot (for EnsembleBlend or best model)
    best_pred = oof_predictions.get("EnsembleBlend", list(oof_predictions.values())[0])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_actual, best_pred, alpha=0.5, color="#2563eb", edgecolors="none", s=28, label="Predictions")
    min_val = min(y_actual.min(), best_pred.min())
    max_val = max(y_actual.max(), best_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
    ax.set_title("Actual vs. Predicted House Prices ($)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Actual SalePrice ($)", fontsize=11)
    ax.set_ylabel("Predicted SalePrice ($)", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    act_pred_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(act_pred_path, dpi=300)
    plt.close()
    print(f"Saved: {act_pred_path}")

    # 3. Residuals Distribution Plot
    residuals = y_actual - best_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(best_pred, residuals, alpha=0.5, color="#059669", s=25)
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_title("Residuals vs. Fitted Values", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Fitted SalePrice ($)", fontsize=11)
    axes[0].set_ylabel("Residual ($)", fontsize=11)

    axes[1].hist(residuals, bins=40, color="#10b981", edgecolor="#047857", alpha=0.75)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title("Residuals Distribution Histogram", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual Error ($)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    residuals_path = os.path.join(output_dir, "residuals_distribution.png")
    plt.savefig(residuals_path, dpi=300)
    plt.close()
    print(f"Saved: {residuals_path}")

    # 4. Feature Importance Plot (Gradient Boosting)
    try:
        # Extract feature names from ColumnTransformer
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_features_out = cat_encoder.get_feature_names_out(feature_engineer.categorical_features_).tolist()
        all_feature_names = feature_engineer.numeric_features_ + cat_features_out

        gbr = GradientBoostingRegressor(n_estimators=350, learning_rate=0.04, max_depth=3, subsample=0.8, random_state=42)
        X_proc = preprocessor.transform(feature_engineer.transform(pd.read_csv("train.csv").drop(columns=["SalePrice", "Id"], errors="ignore")))
        y_log_train = np.log1p(pd.read_csv("train.csv")["SalePrice"])
        gbr.fit(X_proc, y_log_train)

        importances = gbr.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_names = [all_feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_importances[::-1], color="#4f46e5", alpha=0.85, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=10)
        ax.set_xlabel("Relative Importance Score", fontsize=11)
        ax.set_title("Top 15 Most Influential Property Features (Gradient Boosting)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        feat_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"Saved: {feat_path}")
        return top_names, top_importances.tolist()
    except Exception as e:
        print(f"Warning: Could not plot feature importance: {e}")
        return [], []


def train_and_save_pipeline(train_csv_path="train.csv", model_dir="models"):
    """
    Executes complete training workflow and serializes best pipeline to disk.
    """
    os.makedirs(model_dir, exist_ok=True)
    X, y, y_log = load_and_clean_data(train_csv_path, remove_outliers=True)

    fe = HousingFeatureEngineer()
    fe.fit(X)
    X_eng = fe.transform(X)

    preprocessor = create_preprocessor(fe.numeric_features_, fe.categorical_features_)
    preprocessor.fit(X_eng)

    # Cross-validation
    metrics, oof_predictions, X_proc, y_actual = evaluate_models(X, y, y_log, fe, preprocessor)

    # Select best model based on R2 / RMSLE
    best_model_name = "EnsembleBlend" if "EnsembleBlend" in metrics else max(metrics, key=lambda k: metrics[k]["R2_Score"])
    best_estimator = get_candidate_models()[best_model_name]
    print(f"[3/5] Selected best model: {best_model_name} (R² = {metrics[best_model_name]['R2_Score']})")

    # Fit best model on complete training set (using log target)
    print("[4/5] Fitting final pipeline on complete training data...")
    final_pipeline = Pipeline([
        ("feature_engineer", fe),
        ("preprocessor", preprocessor),
        ("regressor", best_estimator),
    ])
    final_pipeline.fit(X, y_log)

    # Save visualization plots
    top_feature_names, top_weights = plot_visualizations(
        metrics, oof_predictions, y_actual, fe, preprocessor, best_estimator
    )

    # Save model pipeline
    pipeline_path = os.path.join(model_dir, "house_price_pipeline.joblib")
    joblib.dump(final_pipeline, pipeline_path)
    print(f"[5/5] Saved serialized pipeline to: {pipeline_path}")

    # Save metrics metadata
    metadata = {
        "best_model": best_model_name,
        "metrics": metrics,
        "sample_count": len(X),
        "engineered_feature_count": len(fe.feature_columns_),
        "encoded_feature_count": X_proc.shape[1],
        "top_features": [
            {"name": name, "importance": round(weight, 4)}
            for name, weight in zip(top_feature_names, top_weights)
        ],
        "target_summary": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "median": float(y.median()),
            "std": float(y.std()),
        },
        "neighborhoods": sorted(X["Neighborhood"].dropna().unique().tolist()) if "Neighborhood" in X.columns else [],
    }

    metadata_path = os.path.join(model_dir, "model_metrics.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    return final_pipeline, metadata


if __name__ == "__main__":
    train_and_save_pipeline()
