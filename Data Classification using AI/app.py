"""Data Classification Using AI - Project 2.

This script builds a supervised learning classifier on the Iris dataset using
K-Nearest Neighbors, feature scaling, and a train-test split. It reports
accuracy, confusion matrix, classification metrics, and K-value comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


APP_TITLE = "Data Classification Using AI - Project 2"
OUTPUT_DIR = Path(__file__).with_name("artifacts")
RANDOM_STATE = 42
TEST_SIZE = 0.2
K_VALUES = [1, 3, 5, 7, 9, 11, 15]


@dataclass(slots=True)
class ClassificationResults:
    """Container for the final evaluation outputs."""

    best_k: int
    accuracy: float
    weighted_f1: float
    confusion: np.ndarray
    classification_report_text: str
    k_summary: pd.DataFrame


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the Iris benchmark dataset."""

    iris = load_iris()
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)
    return iris.data, iris.target, feature_names, target_names


def preprocess_data(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split the data and apply standard scaling."""

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_k_values(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Train KNN models over multiple K values and collect scores."""

    rows: list[dict[str, float]] = []

    for k in K_VALUES:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        rows.append(
            {
                "k": float(k),
                "train_accuracy": float(model.score(X_train, y_train)),
                "test_accuracy": float(model.score(X_test, y_test)),
            }
        )

    return pd.DataFrame(rows)


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
) -> ClassificationResults:
    """Fit the final model using the best observed K value."""

    k_summary = evaluate_k_values(X_train, X_test, y_train, y_test)
    best_row = k_summary.loc[k_summary["test_accuracy"].idxmax()]
    best_k = int(best_row["k"])

    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train, y_train)
    predictions = final_model.predict(X_test)

    confusion = confusion_matrix(y_test, predictions)
    report_text = classification_report(y_test, predictions, target_names=target_names)
    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    return ClassificationResults(
        best_k=best_k,
        accuracy=accuracy,
        weighted_f1=weighted_f1,
        confusion=confusion,
        classification_report_text=report_text,
        k_summary=k_summary,
    )


def save_confusion_matrix(confusion: np.ndarray, target_names: list[str], best_k: int) -> Path:
    """Render and save the confusion matrix heatmap."""

    OUTPUT_DIR.mkdir(exist_ok=True)
    figure_path = OUTPUT_DIR / f"confusion_matrix_knn_k{best_k}.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix - KNN (K={best_k})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    return figure_path


def save_k_performance_plot(k_summary: pd.DataFrame) -> Path:
    """Render and save the K-value performance chart."""

    OUTPUT_DIR.mkdir(exist_ok=True)
    figure_path = OUTPUT_DIR / "knn_performance_by_k.png"

    plt.figure(figsize=(10, 6))
    plt.plot(k_summary["k"], k_summary["train_accuracy"], marker="o", label="Training Accuracy", linewidth=2)
    plt.plot(k_summary["k"], k_summary["test_accuracy"], marker="s", label="Testing Accuracy", linewidth=2)
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.title("KNN Performance vs K Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    return figure_path


def print_report(
    features: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    results: ClassificationResults,
    confusion_path: Path,
    k_plot_path: Path,
) -> None:
    """Print the project report to the console."""

    print(APP_TITLE)
    print("=" * len(APP_TITLE))
    print(f"Dataset shape: {features.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_names}")
    print(f"Training samples: {y_train.shape[0]}")
    print(f"Testing samples: {y_test.shape[0]}")
    print(f"Class counts: {np.bincount(np.concatenate([y_train, y_test]))}")
    print(f"Scaler mean (first 4 features): {np.round(scaler.mean_, 4)}")
    print(f"Scaler variance (first 4 features): {np.round(scaler.var_, 4)}")
    print()

    print("K-value comparison:")
    print(results.k_summary.to_string(index=False, formatters={"k": "{:.0f}".format, "train_accuracy": "{:.4f}".format, "test_accuracy": "{:.4f}".format}))
    print()

    print(f"Best K value: {results.best_k}")
    print()
    print("Confusion Matrix:")
    print(results.confusion)
    print()
    print("Classification Report:")
    print(results.classification_report_text)
    print(f"Weighted F1 Score: {results.weighted_f1:.4f}")
    print(f"Test Accuracy: {results.accuracy:.4f}")
    print()
    print(f"Saved confusion matrix plot: {confusion_path}")
    print(f"Saved K-value plot: {k_plot_path}")


def main() -> None:
    """Run the full Iris classification workflow."""

    features, labels, feature_names, target_names = load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(features, labels)
    results = train_final_model(X_train, y_train, X_test, y_test, target_names)
    confusion_path = save_confusion_matrix(results.confusion, target_names, results.best_k)
    k_plot_path = save_k_performance_plot(results.k_summary)
    print_report(
        features,
        feature_names,
        target_names,
        y_train,
        y_test,
        scaler,
        results,
        confusion_path,
        k_plot_path,
    )


if __name__ == "__main__":
    main()
