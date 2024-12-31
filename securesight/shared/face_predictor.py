import numpy as np
import joblib

class FacePredictor:
    """
    A face predictor class that uses a pre-trained model to predict labels for face embeddings.

    Attributes:
        model: The machine learning model loaded from a joblib file.
    """

    def __init__(self, weights='./shared/weights/knn.joblib'):
        """
        Initializes the FacePredictor with the given model weights.

        Args:
            weights (str): Path to the joblib file containing the pre-trained model.
        """
        self.model = joblib.load(weights)

    def __call__(self, X: np.ndarray):
        """
        Predicts labels for the provided face embeddings.

        Args:
            X (np.ndarray): An array of face embeddings.

        Returns:
            np.ndarray: The predicted labels for the embeddings.
        """
        return self.model.predict(X)
