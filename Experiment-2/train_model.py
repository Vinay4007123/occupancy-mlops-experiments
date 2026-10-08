import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Dataset is stored in the repository root
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "Occupancy_Estimation.csv"
)

TARGET = "Room_Occupancy_Count"


FEATURES = [
    "S1_Temp",
    "S2_Temp",
    "S3_Temp",
    "S4_Temp",
    "S1_Light",
    "S2_Light",
    "S3_Light",
    "S4_Light",
    "S1_Sound",
    "S2_Sound",
    "S3_Sound",
    "S4_Sound",
    "S5_CO2",
    "S5_CO2_Slope",
    "S6_PIR",
    "S7_PIR"
]


def train_model():

    print("Loading occupancy dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    X = data[FEATURES]
    y = data[TARGET]

    print("Target classes:", sorted(y.unique()))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])

    print("Training model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print()
    print("Model Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("Confusion Matrix:")
    print(matrix)

    # Save model inside Experiment-2
    model_path = (
        Path(__file__).resolve().parent
        / "occupancy_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Model saved as:",
        model_path.name
    )

    metrics = {
        "accuracy": float(accuracy),
        "total_records": int(len(data)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "target": TARGET,
        "classes": [
            int(value)
            for value in sorted(y.unique())
        ],
        "features": FEATURES,
        "confusion_matrix": matrix.tolist()
    }

    metrics_path = (
        Path(__file__).resolve().parent
        / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        "Metrics saved as:",
        metrics_path.name
    )


if __name__ == "__main__":
    train_model()
