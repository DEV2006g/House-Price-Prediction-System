"""
Prediction and Inference Engine for House Price Prediction System.
Supports single-record predictions (with confidence intervals) and batch CSV processing.
"""

import os
import joblib
import numpy as np
import pandas as pd


class HousePricePredictor:
    """
    Wrapper for trained pipeline to predict house prices for new observations.
    """

    def __init__(self, model_path="models/house_price_pipeline.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please train the model first using run_pipeline.py"
            )
        self.pipeline = joblib.load(model_path)
        # Approximate standard error of residuals in log scale for confidence intervals (~0.12)
        self.rmse_log = 0.125

    def predict(self, df_input):
        """
        Takes a pandas DataFrame and returns array of predicted prices on dollar scale.
        """
        pred_log = self.pipeline.predict(df_input)
        pred_dollar = np.expm1(pred_log)
        return np.maximum(pred_dollar, 10000.0)

    def predict_single(self, feature_dict):
        """
        Takes a single dictionary of property features and returns detailed valuation.
        """
        df = pd.DataFrame([feature_dict])
        pred_dollar = self.predict(df)[0]

        # Calculate estimated 90% confidence range (+- 1.645 * standard error on log scale)
        log_val = np.log1p(pred_dollar)
        lower_bound = np.expm1(log_val - 1.645 * self.rmse_log)
        upper_bound = np.expm1(log_val + 1.645 * self.rmse_log)

        # Total sqft calculation for price/sqft metric
        gr_liv_area = float(feature_dict.get("GrLivArea", 1500))
        total_bsmt = float(feature_dict.get("TotalBsmtSF", 800))
        total_sf = gr_liv_area + total_bsmt
        price_per_sqft = (pred_dollar / total_sf) if total_sf > 0 else 0

        return {
            "predicted_price": round(float(pred_dollar), 2),
            "formatted_price": f"${pred_dollar:,.0f}",
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "formatted_range": f"${lower_bound:,.0f} - ${upper_bound:,.0f}",
            "price_per_sqft": round(float(price_per_sqft), 2),
            "formatted_sqft_price": f"${price_per_sqft:,.2f}/sq ft",
        }

    def predict_batch_and_save(self, test_csv_path="test.csv", output_path="submission_predicted.csv"):
        """
        Predicts house prices on test.csv and saves standard Kaggle submission CSV.
        """
        if not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

        df_test = pd.read_csv(test_csv_path)
        ids = df_test["Id"].values if "Id" in df_test.columns else np.arange(len(df_test))

        print(f"Generating predictions for {len(df_test)} test properties...")
        predictions = self.predict(df_test)

        submission_df = pd.DataFrame({
            "Id": ids,
            "SalePrice": np.round(predictions, 2),
        })

        submission_df.to_csv(output_path, index=False)
        print(f"Batch predictions successfully saved to: {output_path}")
        return submission_df


if __name__ == "__main__":
    predictor = HousePricePredictor()
    predictor.predict_batch_and_save()
