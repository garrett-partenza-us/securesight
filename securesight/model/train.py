import os
import json
import copy
import cv2
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from shared.face_detector import FaceDetector
from shared.face_encoder import FaceEncoder

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATA_DIR = "./model/data/images"  # Training data


def load_and_process_images(data_dir):
    """
    Loads and processes images to extract embeddings and labels.

    Args:
        data_dir (str): Directory containing the training images.

    Returns:
        tuple: Two numpy arrays, embeddings (X) and labels (y).
    """
    detector = FaceDetector()
    encoder = FaceEncoder()
    X, y = [], []

    for name in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, name)):
            for number in os.listdir(os.path.join(data_dir, name)):
                try:
                    path = os.path.join(data_dir, name, number)
                    logging.info(f"\nProcessing player: {name}, image: {number}")

                    frame = cv2.imread(path)

                    boxes, nms, scores, scale = detector(copy.deepcopy(frame))
                    embeddings = encoder(copy.deepcopy(frame), boxes, nms, scale)

                    for vec in embeddings:
                        X.append(vec)
                        y.append(name)
                except Exception as e:
                    logging.error(f"Failed for image: {path}", exc_info=True)

    pca = PCA(n_components=27)
    X = pca.fit_transform(X)
    joblib.dump(pca, './shared/weights/pca.joblib')
    # Save the principal components and mean as separate JSON file
    pca_components = pca.components_.tolist()
    mean = pca.mean_.tolist()
    with open('./shared/weights/pca_components.json', 'w') as f:
        json.dump({'components': pca_components, 'mean': mean}, f)

    return np.array(X), np.array(y)


def train_knn_classifier(X, y, model_path="./shared/weights/knn.joblib"):
    """
    Trains a K-Nearest Neighbors (KNN) classifier and saves the model.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        model_path (str): Path to save the trained model.
    """
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    joblib.dump(knn, model_path)
    logging.info(f"KNN model saved to {model_path}")


def visualize_embeddings_with_pca(X, y, output_path="./model/plots/PCA.png"):
    """
    Visualizes embeddings using PCA and saves the scatter plot.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        output_path (str): Path to save the PCA plot.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', alpha=0.7)
    plt.title('2D PCA Scatter Plot of Face Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Name Labels')

    plt.savefig(output_path)
    logging.info(f"PCA scatter plot saved to {output_path}")
    plt.show()


def main():
    """
    Main function to process images, train the model, and visualize results.
    """
    X, y = load_and_process_images(DATA_DIR)
    print(X.shape, y.shape)
    np.savetxt("./shared/weights/knn.csv", np.append(X.astype(str), y.reshape(-1, 1), axis=1),
               delimiter=",", fmt="%s")
    train_knn_classifier(X, y)
    visualize_embeddings_with_pca(X, y)


if __name__ == "__main__":
    main()
