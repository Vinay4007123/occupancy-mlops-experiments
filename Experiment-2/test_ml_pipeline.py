import json
import os
import unittest
from pathlib import Path

import joblib
import pandas as pd

from train_model import (
    DATASET_PATH,
    FEATURES,
    TARGET
)


EXPERIMENT_DIR = Path(
    __file__
).resolve().parent

MODEL_PATH = (
    EXPERIMENT_DIR / "occupancy_model.pkl"
)

METRICS_PATH = (
    EXPERIMENT_DIR / "metrics.json"
)


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            DATASET_PATH.exists()
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists(MODEL_PATH)
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists(METRICS_PATH)
        )

    def test_accuracy_is_valid(self):

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(
            accuracy,
            0.0
        )

        self.assertLessEqual(
            accuracy,
            1.0
        )

    def test_model_prediction(self):

        model = joblib.load(
            MODEL_PATH
        )

        data = pd.read_csv(
            DATASET_PATH
        )

        sample = data[
            FEATURES
        ].iloc[[0]]

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1, 2, 3]
        )

    def test_expected_classes(self):

        data = pd.read_csv(
            DATASET_PATH
        )

        classes = sorted(
            data[TARGET].unique().tolist()
        )

        self.assertEqual(
            classes,
            [0, 1, 2, 3]
        )

    def test_required_features_exist(self):

        data = pd.read_csv(
            DATASET_PATH
        )

        for feature in FEATURES:

            self.assertIn(
                feature,
                data.columns
            )


if __name__ == "__main__":
    unittest.main()
