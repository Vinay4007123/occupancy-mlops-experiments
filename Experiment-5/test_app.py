import unittest
from pathlib import Path

import pandas as pd

from app import app, FEATURES


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "Occupancy_Estimation.csv"
)


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):

        self.client = app.test_client()

        self.data = pd.read_csv(
            DATASET_PATH
        )


    def test_root_endpoint(self):

        response = self.client.get("/")

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertEqual(
            response.get_json()["status"],
            "ok"
        )


    def test_health_endpoint(self):

        response = self.client.get("/health")

        self.assertEqual(
            response.status_code,
            200
        )

        self.assertEqual(
            response.get_json()["status"],
            "healthy"
        )


    def test_prediction_endpoint(self):

        row = self.data[
            FEATURES
        ].iloc[0]

        sample = {
            feature: float(row[feature])
            for feature in FEATURES
        }

        response = self.client.post(
            "/predict",
            json=sample
        )

        self.assertEqual(
            response.status_code,
            200
        )

        result = response.get_json()

        self.assertIn(
            "prediction",
            result
        )

        self.assertIn(
            int(result["prediction_code"]),
            [0, 1, 2, 3]
        )


    def test_missing_field_validation(self):

        response = self.client.post(
            "/predict",
            json={
                "S1_Temp": 25.0,
                "S2_Temp": 26.0
            }
        )

        self.assertEqual(
            response.status_code,
            400
        )

        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
