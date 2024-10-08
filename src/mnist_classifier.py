from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random


class DigitClassificationInterface(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """Predict the digit in the given image."""
        pass

    @abstractmethod
    def evaluate(self, X, y) -> float:
        """Evaluate the model and return the accuracy score."""
        pass


class CNNModel(DigitClassificationInterface):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(28, 28, 1)),
                keras.layers.Conv2D(32, (3, 3), activation="relu"),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(self, X, y, validation_data=None, epochs=5, batch_size=64):
        """Train the model on the given data."""
        self.model.fit(
            X,
            y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
        )

    def predict(self, image: np.ndarray) -> int:
        """Predict the digit in the given image."""
        # Ensure the image has the correct shape (28, 28, 1)
        if image.shape != (28, 28, 1):
            image = image.reshape((28, 28, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = self.model.predict(image)
        return np.argmax(prediction[0])

    def evaluate(self, X, y) -> float:
        """Evaluate the model and return the accuracy score."""
        predictions = self.model.predict(X)
        y_pred = np.argmax(predictions, axis=1)
        return accuracy_score(y, y_pred)


class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_fitted = False

    def fit(self, X, y):
        """Train the model on the given data."""
        # Flatten the images for RandomForest
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
        self.is_fitted = True

    def predict(self, image: np.ndarray) -> int:
        if not self.is_fitted:
            raise RuntimeError("The model has not been fitted yet.")

        flat_image = image.flatten()
        return self.model.predict(flat_image.reshape(1, -1))[0]

    def evaluate(self, X, y) -> float:
        """Evaluate the model and return the accuracy score."""
        X_flat = X.reshape(X.shape[0], -1)
        y_pred = self.model.predict(X_flat)
        return accuracy_score(y, y_pred)


class RandomModel(DigitClassificationInterface):
    def fit(self, X, y):
        """No training needed for random model."""
        pass

    def predict(self, image: np.ndarray) -> int:
        """Predict a random digit."""
        return random.randint(0, 9)

    def evaluate(self, X, y) -> float:
        """Evaluate the model and return the accuracy score."""
        y_pred = [random.randint(0, 9) for _ in range(len(y))]
        return accuracy_score(y, y_pred)


class DigitClassifier:
    def __init__(self, algorithm: str):
        if algorithm == "cnn":
            self.model = CNNModel()
        elif algorithm == "rf":
            self.model = RandomForestModel()
        elif algorithm == "rand":
            self.model = RandomModel()
        else:
            raise ValueError("Unsupported algorithm")

    def fit(self, X, y, validation_data=None):
        if isinstance(self.model, CNNModel):
            self.model.fit(X, y, validation_data=validation_data)
        else:
            self.model.fit(X, y)

    def predict(self, image: np.ndarray) -> int:
        return self.model.predict(image)

    def evaluate(self, X, y) -> float:
        """Evaluate the model and return the accuracy score."""
        return self.model.evaluate(X, y)


# Add new functions for data handling and preprocessing
def load_and_preprocess_data():
    """Load and preprocess the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape images to (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    for algo in ["cnn", "rf", "rand"]:
        print(f"\nTraining and evaluating {algo.upper()} classifier:")
        classifier = DigitClassifier(algo)
        if algo == "cnn":
            classifier.fit(x_train, y_train, validation_data=(x_val, y_val))
        else:
            classifier.fit(x_train, y_train)

        accuracy = classifier.evaluate(x_test, y_test)
        print(f"{algo.upper()} accuracy: {accuracy:.4f}")

        test_image = x_test[0]
        prediction = classifier.predict(test_image)
        print(f"{algo.upper()} prediction for first test image: {prediction}")

    print(f"\nTrue label of first test image: {y_test[0]}")
