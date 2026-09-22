"""
Unit and Integration Tests for House Price Prediction System.
"""

import os
import unittest
import numpy as np
import pandas as pd
from src.predict import HousePricePredictor


class TestHousePricePrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = HousePricePredictor("models/house_price_pipeline.joblib")

    def test_pipeline_loaded(self):
        self.assertIsNotNone(self.predictor.pipeline)

    def test_single_prediction_standard_home(self):
        sample_input = {
            "OverallQual": 7,
            "GrLivArea": 1710,
            "TotalBsmtSF": 856,
            "1stFlrSF": 856,
            "2ndFlrSF": 854,
            "GarageCars": 2,
            "GarageArea": 548,
            "YearBuilt": 2003,
            "YearRemodAdd": 2003,
            "FullBath": 2,
            "HalfBath": 1,
            "BedroomAbvGr": 3,
            "KitchenQual": "Gd",
            "Neighborhood": "CollgCr",
            "LotArea": 8450,
            "Fireplaces": 0,
        }
        res = self.predictor.predict_single(sample_input)
        self.assertIn("predicted_price", res)
        self.assertIn("formatted_price", res)
        self.assertIn("lower_bound", res)
        self.assertIn("upper_bound", res)
        self.assertIn("price_per_sqft", res)

        price = res["predicted_price"]
        self.assertTrue(100000 < price < 400000, f"Expected reasonable price, got {price}")
        self.assertTrue(res["lower_bound"] < price < res["upper_bound"])

    def test_luxury_vs_budget_comparison(self):
        luxury_house = {
            "OverallQual": 10,
            "GrLivArea": 3500,
            "TotalBsmtSF": 2000,
            "1stFlrSF": 2000,
            "2ndFlrSF": 1500,
            "GarageCars": 3,
            "GarageArea": 900,
            "YearBuilt": 2008,
            "YearRemodAdd": 2009,
            "FullBath": 3,
            "Neighborhood": "NridgHt",
            "KitchenQual": "Ex",
        }
        budget_house = {
            "OverallQual": 4,
            "GrLivArea": 900,
            "TotalBsmtSF": 600,
            "1stFlrSF": 900,
            "2ndFlrSF": 0,
            "GarageCars": 1,
            "GarageArea": 250,
            "YearBuilt": 1950,
            "YearRemodAdd": 1950,
            "FullBath": 1,
            "Neighborhood": "OldTown",
            "KitchenQual": "TA",
        }
        lux_res = self.predictor.predict_single(luxury_house)
        bud_res = self.predictor.predict_single(budget_house)

        self.assertGreater(lux_res["predicted_price"], bud_res["predicted_price"])

    def test_partial_features_graceful_handling(self):
        # Only 3 features provided, rest should be safely imputed by FeatureEngineer
        minimal_input = {
            "OverallQual": 6,
            "GrLivArea": 1400,
            "YearBuilt": 1990,
        }
        res = self.predictor.predict_single(minimal_input)
        self.assertGreater(res["predicted_price"], 50000)


if __name__ == "__main__":
    unittest.main()
