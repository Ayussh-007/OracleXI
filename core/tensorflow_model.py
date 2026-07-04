"""
OracleXI – TensorFlow Neural Network Model
=============================================
Lightweight neural network for sports prediction using TensorFlow.

Demonstrates:
    - tf.keras.Sequential model building
    - Dense layers with activation functions
    - Dropout regularization
    - Model compilation, training, evaluation
    - Model saving and loading (.keras format)
    - Prediction with confidence scores
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from utils.constants import (
    CRICKET_MODEL_PATH,
    FOOTBALL_MODEL_PATH,
    MODEL_CONFIG,
    MODEL_DIR,
)
from utils.helper import setup_logging

logger = setup_logging("TFModel")


class TensorFlowPredictor:
    """
    TensorFlow-based neural network for sports match prediction.

    Builds, trains, saves, loads, and predicts using a lightweight
    Sequential model with Dense layers.
    """

    def __init__(self) -> None:
        """Initialize the TensorFlow predictor."""
        self.football_model: Optional[object] = None
        self.cricket_model: Optional[object] = None
        self._models_loaded: bool = False

        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. ML predictions disabled.")
        else:
            logger.info("TensorFlowPredictor initialized")
            os.makedirs(MODEL_DIR, exist_ok=True)
            self._load_existing_models()

    # ─────────────────────────────────────────
    # Model Building
    # ─────────────────────────────────────────

    def build_model(
        self,
        sport: str = "football",
    ) -> object:
        """
        Build a lightweight Sequential neural network.

        Architecture:
            Input → Dense(64, ReLU) → Dropout(0.2) →
            Dense(32, ReLU) → Dropout(0.2) →
            Dense(16, ReLU) →
            Dense(output, Softmax)

        Args:
            sport: 'football' or 'cricket'.

        Returns:
            Compiled Keras model.
        """
        if not TENSORFLOW_AVAILABLE:
            return None

        config = MODEL_CONFIG[sport]
        input_dim = config["input_features"] * 2  # Combined features for both teams

        # TensorFlow: Build Sequential model
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(input_dim,)),

            # Hidden layer 1
            tf.keras.layers.Dense(
                config["hidden_layers"][0],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(config["dropout_rate"]),

            # Hidden layer 2
            tf.keras.layers.Dense(
                config["hidden_layers"][1],
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(config["dropout_rate"]),

            # Hidden layer 3
            tf.keras.layers.Dense(
                config["hidden_layers"][2],
                activation="relu",
            ),

            # Output layer with softmax for probability distribution
            tf.keras.layers.Dense(
                config["output_classes"],
                activation="softmax",
            ),
        ])

        # TensorFlow: Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config["learning_rate"]
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(
            f"Built {sport} model: "
            f"input={input_dim}, "
            f"layers={config['hidden_layers']}, "
            f"output={config['output_classes']}"
        )

        return model

    # ─────────────────────────────────────────
    # Model Training
    # ─────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sport: str = "football",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        validation_split: float = 0.2,
    ) -> Dict[str, any]:
        """
        Train the neural network on historical data.

        Demonstrates:
            - model.fit() with validation split
            - Training history tracking
            - Early stopping callback

        Args:
            X_train: Feature matrix (n_samples, n_features).
            y_train: Label vector (n_samples,).
            sport: 'football' or 'cricket'.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            validation_split: Fraction for validation.

        Returns:
            Dictionary with training history and metrics.
        """
        if not TENSORFLOW_AVAILABLE:
            return {"accuracy": 0.0, "loss": 0.0, "trained": False}

        if len(X_train) == 0 or len(y_train) == 0:
            logger.warning("Empty training data provided")
            return {"accuracy": 0.0, "loss": 0.0, "trained": False}

        config = MODEL_CONFIG[sport]
        epochs = epochs or config["epochs"]
        batch_size = batch_size or config["batch_size"]

        # Build model
        model = self.build_model(sport)
        if model is None:
            return {"accuracy": 0.0, "loss": 0.0, "trained": False}

        # TensorFlow: Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        logger.info(
            f"Training {sport} model: "
            f"{len(X_train)} samples, {epochs} epochs"
        )

        # TensorFlow: Model training
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0,
        )

        # Save model
        if sport == "football":
            self.football_model = model
            self.save_model(model, FOOTBALL_MODEL_PATH)
        else:
            self.cricket_model = model
            self.save_model(model, CRICKET_MODEL_PATH)

        # Extract training metrics
        final_acc = history.history["accuracy"][-1]
        final_loss = history.history["loss"][-1]
        val_acc = history.history.get("val_accuracy", [0.0])[-1]
        val_loss = history.history.get("val_loss", [0.0])[-1]

        logger.info(
            f"{sport} model trained: "
            f"acc={final_acc:.4f}, val_acc={val_acc:.4f}"
        )

        return {
            "accuracy": round(float(final_acc), 4),
            "val_accuracy": round(float(val_acc), 4),
            "loss": round(float(final_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "epochs_trained": len(history.history["loss"]),
            "trained": True,
        }

    # ─────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────

    def predict(
        self,
        features: np.ndarray,
        sport: str = "football",
    ) -> Dict[str, float]:
        """
        Predict match outcome using the trained model.

        Demonstrates:
            - model.predict() for inference
            - Softmax probability interpretation
            - Confidence score calculation

        Args:
            features: Combined feature vector (1, n_features).
            sport: 'football' or 'cricket'.

        Returns:
            Dictionary with prediction probabilities and confidence.
        """
        model = (
            self.football_model if sport == "football"
            else self.cricket_model
        )

        if model is None or not TENSORFLOW_AVAILABLE:
            # Return uniform probabilities as fallback
            if sport == "football":
                return {
                    "team_a_win": 0.40,
                    "draw": 0.20,
                    "team_b_win": 0.40,
                    "confidence": 0.0,
                }
            else:
                return {
                    "team_a_win": 0.50,
                    "team_b_win": 0.50,
                    "confidence": 0.0,
                }

        try:
            # Reshape for single prediction
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # TensorFlow: Model prediction
            probabilities = model.predict(features, verbose=0)[0]

            # Calculate confidence as max probability deviation from uniform
            n_classes = len(probabilities)
            uniform = 1.0 / n_classes
            confidence = float(np.max(np.abs(probabilities - uniform)) / uniform)
            confidence = min(confidence, 1.0)

            if sport == "football":
                return {
                    "team_a_win": round(float(probabilities[0]), 4),
                    "draw": round(float(probabilities[1]), 4),
                    "team_b_win": round(float(probabilities[2]), 4),
                    "confidence": round(confidence, 4),
                }
            else:
                return {
                    "team_a_win": round(float(probabilities[0]), 4),
                    "team_b_win": round(float(probabilities[1]), 4),
                    "confidence": round(confidence, 4),
                }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            if sport == "football":
                return {
                    "team_a_win": 0.40,
                    "draw": 0.20,
                    "team_b_win": 0.40,
                    "confidence": 0.0,
                }
            else:
                return {
                    "team_a_win": 0.50,
                    "team_b_win": 0.50,
                    "confidence": 0.0,
                }

    # ─────────────────────────────────────────
    # Model Persistence
    # ─────────────────────────────────────────

    def save_model(self, model: object, path: str) -> bool:
        """
        Save a trained Keras model to disk.

        Args:
            model: Trained Keras model.
            path: File path for saving.

        Returns:
            True if save was successful.
        """
        if not TENSORFLOW_AVAILABLE or model is None:
            return False

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> Optional[object]:
        """
        Load a trained Keras model from disk.

        Args:
            path: File path to load from.

        Returns:
            Loaded Keras model or None.
        """
        if not TENSORFLOW_AVAILABLE:
            return None

        if not os.path.exists(path):
            logger.info(f"No saved model found at {path}")
            return None

        try:
            model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return None

    def _load_existing_models(self) -> None:
        """Attempt to load previously saved models."""
        self.football_model = self.load_model(FOOTBALL_MODEL_PATH)
        self.cricket_model = self.load_model(CRICKET_MODEL_PATH)
        if self.football_model or self.cricket_model:
            self._models_loaded = True

    def is_model_trained(self, sport: str = "football") -> bool:
        """Check if a model is trained and ready for prediction."""
        if sport == "football":
            return self.football_model is not None
        return self.cricket_model is not None

    def get_model_summary(self, sport: str = "football") -> str:
        """Get a string summary of the model architecture."""
        model = (
            self.football_model if sport == "football"
            else self.cricket_model
        )
        if model is None or not TENSORFLOW_AVAILABLE:
            return "Model not available"

        try:
            lines: List[str] = []
            model.summary(print_fn=lambda x: lines.append(x))
            return "\n".join(lines)
        except Exception:
            return "Model summary unavailable"
