import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import joblib
import logging
import time
import csv
from typing import List, Dict, Tuple

# Configure logging to display timestamp, log level, and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataLoader:
    """Handles data loading and preprocessing operations."""

    @staticmethod
    def load_and_preprocess(
        file_path: str, selected_features: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess the data from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
            selected_features (List[str]): List of feature names to select. If None, all features are used.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: X (features) and y (target)
        """
        logging.info("Loading data from %s", file_path)
        df = pd.read_csv(file_path)

        # Select features based on input, or use all features except 'target'
        X = df[selected_features] if selected_features else df.drop("target", axis=1)
        y = df["target"]
        logging.info("Data loaded. Shape: %s", X.shape)
        return X, y


class ModelTrainer:
    """Class for training and evaluating multiple regression models."""

    def __init__(self):
        # Initialize a dictionary of models to train
        self.models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        }
        # Initialize the scaler for feature normalization
        self.scaler = StandardScaler()

    def train(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[object, object, float, List[Dict]]:
        """
        Train multiple models and return the best one along with the scaler.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.

        Returns:
            Tuple[object, object, float, List[Dict]]: best_model, scaler, best_rmse, and results list.
        """
        logging.info("Starting model training")
        start_time = time.time()

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logging.info(
            "Train set shape: %s, Validation set shape: %s", X_train.shape, X_val.shape
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        logging.info("Data scaled")

        # Evaluate all models and get the best one
        best_model, best_rmse, results = self._evaluate_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )

        end_time = time.time()
        logging.info("Total training time: %.2f seconds", end_time - start_time)

        return best_model, self.scaler, best_rmse, results

    def _evaluate_models(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
    ) -> Tuple[object, float, List[Dict]]:
        best_model = None
        best_rmse = float("inf")
        results = []

        for name, model in self.models.items():
            logging.info("Training %s", name)
            # Perform cross-validation
            scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
            )
            rmse_scores = np.sqrt(-scores)
            mean_rmse = rmse_scores.mean()
            std_rmse = rmse_scores.std()
            logging.info(
                "%s - Mean RMSE: %.4f (+/- %.4f)", name, mean_rmse, std_rmse * 2
            )

            # Fit the model and evaluate on validation set
            model.fit(X_train, y_train)
            val_predictions = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

            # Store results
            results.append(
                {
                    "Model": name,
                    "Mean CV RMSE": mean_rmse,
                    "Std CV RMSE": std_rmse,
                    "Validation RMSE": val_rmse,
                    "Description": "Using selected features",
                }
            )

            # Update best model if current model performs better
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model = model

        logging.info("Best model: %s", type(best_model).__name__)
        logging.info("Best Validation RMSE: %.4f", best_rmse)

        return best_model, best_rmse, results


class ModelSaver:
    """Utility class for saving model and scaler objects."""

    @staticmethod
    def save_model_and_scaler(model: object, scaler: object):
        """Save the trained model and scaler to disk."""
        logging.info("Saving model and scaler")
        joblib.dump(model, "model.joblib")
        joblib.dump(scaler, "scaler.joblib")
        logging.info("Model and scaler saved successfully")

    @staticmethod
    def save_results_to_csv(results: List[Dict], filename: str = "model_results.csv"):
        """Save the training results to a CSV file."""
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logging.info("Results saved to %s", filename)


def main():
    """Start the training process."""

    logging.info("Starting the training process")

    # Initialize objects
    data_loader = DataLoader()
    model_trainer = ModelTrainer()
    model_saver = ModelSaver()

    # Train with all features
    X_all, y = data_loader.load_and_preprocess("data/train.csv")
    model_all, scaler_all, val_rmse_all, results_all = model_trainer.train(X_all, y)
    for result in results_all:
        result["Description"] = "Using all features"

    # Train with only features 6 and 7
    X_selected, y = data_loader.load_and_preprocess(
        "data/train.csv", selected_features=["6", "7"]
    )
    model_selected, scaler_selected, val_rmse_selected, results_selected = (
        model_trainer.train(X_selected, y)
    )
    for result in results_selected:
        result["Description"] = "Using only features 6 and 7"

    # Combine and save results from both feature sets
    all_results = results_all + results_selected
    model_saver.save_results_to_csv(all_results)

    # Save the best model (choosing based on validation RMSE)
    if val_rmse_selected < val_rmse_all:
        model_saver.save_model_and_scaler(model_selected, scaler_selected)
        logging.info(
            "Final Validation RMSE (selected features): %.4f", val_rmse_selected
        )
    else:
        model_saver.save_model_and_scaler(model_all, scaler_all)
        logging.info("Final Validation RMSE (all features): %.4f", val_rmse_all)

    logging.info("Model training completed and saved.")


if __name__ == "__main__":
    main()
