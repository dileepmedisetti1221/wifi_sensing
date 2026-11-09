
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

DATA_FILE = "data/features.csv"
MODEL_DIR = "models"
RESULT_DIR = "results"

FEATURES = [
    "mean_rssi",
    "variance",
    "std_rssi",
    "min_rssi",
    "max_rssi",
    "range_rssi"
]

TARGET = "label"
GROUP = "session_id"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading feature dataset...")

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + [TARGET, GROUP]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required)

    print("\nDataset size:", df.shape)
    print("\nClass distribution:")
    print(df[TARGET].value_counts())

    # Hold out one complete recording session per class.
    # This prevents overlapping windows from the same session
    # appearing in both training and testing.
    test_sessions = []

    for label in sorted(df[TARGET].unique()):
        sessions = sorted(
            df.loc[df[TARGET] == label, GROUP].unique()
        )

        if len(sessions) < 2:
            raise ValueError(
                f"Need at least 2 sessions for class {label}"
            )

        test_sessions.append(sessions[-1])

    test_df = df[df[GROUP].isin(test_sessions)].copy()
    train_df = df[~df[GROUP].isin(test_sessions)].copy()

    if set(train_df[TARGET].unique()) != set(df[TARGET].unique()):
        raise ValueError("Training data is missing one or more classes.")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\nTraining sessions:")
    print(sorted(train_df[GROUP].unique()))

    print("\nTesting sessions:")
    print(test_sessions)

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n===== MODEL RESULTS =====")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save trained model and feature information
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "classes": list(model.classes_)
        },
        os.path.join(MODEL_DIR, "occupancy_model.joblib")
    )

    # Save test predictions
    output = test_df[
        ["timestamp", "session_id", "manual_headcount", TARGET]
    ].copy()

    output["predicted_label"] = predictions

    output.to_csv(
        os.path.join(RESULT_DIR, "test_predictions.csv"),
        index=False
    )

    # Generate confusion matrix
    labels = sorted(df[TARGET].unique())

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Wi-Fi Occupancy Classification")
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULT_DIR, "confusion_matrix.png"),
        dpi=150
    )

    plt.close()

    print("\nSaved files:")
    print("models/occupancy_model.joblib")
    print("results/test_predictions.csv")
    print("results/confusion_matrix.png")

    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
