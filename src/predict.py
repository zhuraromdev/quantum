import pandas as pd
import numpy as np
import joblib
import logging
from typing import Tuple

# Configure logging to display timestamp, log level, and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelLoader:
    """Handles loading of trained model and scaler."""

    @staticmethod
    def load_model_and_scaler() -> Tuple[object, object]:
        """
        Load the trained model and scaler from disk.

        Returns:
            Tuple[object, object]: Loaded model and scaler
        """
        logging.info("Loading model and scaler")
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        logging.info("Model and scaler loaded successfully")
        return model, scaler


class DataProcessor:
    """Handles data loading and preprocessing for prediction."""

    @staticmethod
    def load_and_preprocess(file_path: str, scaler: object) -> np.ndarray:
        """
        Load and preprocess the test data.

        Args:
            file_path (str): Path to the CSV file containing the test dataset.
            scaler (object): Fitted StandardScaler object.

        Returns:
            np.ndarray: Scaled test data
        """
        logging.info("Loading test data from %s", file_path)
        test_data = pd.read_csv(file_path)
        logging.info("Test data loaded. Shape: %s", test_data.shape)

        # Select only features 6 and 7
        selected_features = ["6", "7"]
        X_test = test_data[selected_features].values

        logging.info("Selected features for prediction: %s", selected_features)
        logging.info("Shape of selected features: %s", X_test.shape)

        logging.info("Scaling test data")
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled


class Predictor:
    """Handles making predictions using the loaded model."""

    @staticmethod
    def make_predictions(model: object, X_test_scaled: np.ndarray) -> np.ndarray:
        """
        Make predictions on the scaled test data.

        Args:
            model (object): Trained model object.
            X_test_scaled (np.ndarray): Scaled test data.

        Returns:
            np.ndarray: Predictions
        """
        logging.info("Making predictions")
        predictions = model.predict(X_test_scaled)
        logging.info("Predictions made. Shape: %s", predictions.shape)
        return predictions


class ResultSaver:
    """Handles saving of prediction results."""

    @staticmethod
    def save_predictions(predictions: np.ndarray, output_path: str):
        """
        Save the predictions to a CSV file.

        Args:
            predictions (np.ndarray): Array of predictions.
            output_path (str): Path to save the CSV file.
        """
        logging.info("Saving predictions to %s", output_path)
        pd.DataFrame({"prediction": predictions}).to_csv(output_path, index=False)
        logging.info("Predictions saved successfully")


def main():
    """Orchestrate the prediction process."""
    logging.info("Starting prediction process")

    # Initialize objects
    model_loader = ModelLoader()
    data_processor = DataProcessor()
    predictor = Predictor()
    result_saver = ResultSaver()

    # Load model and scaler
    model, scaler = model_loader.load_model_and_scaler()
    logging.info("Model type: %s", type(model).__name__)
    logging.info("Scaler type: %s", type(scaler).__name__)
    logging.info("Number of features the scaler expects: %d", scaler.n_features_in_)

    # Load and preprocess test data
    X_test_scaled = data_processor.load_and_preprocess("data/hidden_test.csv", scaler)

    # Make predictions
    predictions = predictor.make_predictions(model, X_test_scaled)

    # Save predictions
    result_saver.save_predictions(predictions, "predictions.csv")

    logging.info("Prediction process completed")


if __name__ == "__main__":
    main()
