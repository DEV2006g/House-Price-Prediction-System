"""
Master Pipeline Runner for House Price Prediction System.
Executes end-to-end:
1. Data loading & cleaning
2. Feature engineering & preprocessing
3. Multi-model training and 5-fold cross-validation
4. Visualization generation (charts saved to static/plots/)
5. Model serialization to models/house_price_pipeline.joblib
6. Batch predictions on test.csv -> submission_predicted.csv
"""

import sys
import time
from src.model_training import train_and_save_pipeline
from src.predict import HousePricePredictor


def main():
    print("=" * 70)
    print("       HOUSE PRICE PREDICTION SYSTEM - MASTER PIPELINE RUNNER       ")
    print("=" * 70)
    start_time = time.time()

    print("\n>>> Phase 1: Model Training, Evaluation & Plot Generation...")
    pipeline, metadata = train_and_save_pipeline("train.csv", "models")

    print("\n>>> Phase 2: Generating Batch Predictions on test.csv...")
    predictor = HousePricePredictor("models/house_price_pipeline.joblib")
    predictor.predict_batch_and_save("test.csv", "submission_predicted.csv")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("                   PIPELINE COMPLETED SUCCESSFULLY!                  ")
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    print(f"Selected Best Model: {metadata['best_model']}")
    print("\nCross-Validation Performance Summary:")
    for model_name, score_dict in metadata["metrics"].items():
        print(f"  * {model_name:<16}: R² = {score_dict['R2_Score']:.4f} | RMSLE = {score_dict['RMSLE']:.4f} | MAE = ${score_dict['MAE_Dollar']:,.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
