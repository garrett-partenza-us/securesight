import os
import cv2
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO
from run import crop_image_by_box


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATA_DIR = "data/images" # Training data
YOLO = cv2.dnn.readNetFromONNX('weights/yolov11n-face.onnx') # Face detector
VGG = cv2.dnn.readNetFromONNX('weights/inception_resnet_v1.onnx') # Face encoder

# Initialize lists to store features and labels
X, y = [], []

# Iterate through names and associated images
for name in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, name)):
        for number in os.listdir(os.path.join(DATA_DIR, name)):

            try:

                # Construct full image path
                path = os.path.join(DATA_DIR, name, number)
                logging.info(f"\nProcessing player: {name}, image: {number}")

                # Load image
                frame = cv2.imread(path)

                # Pad image to square and calculate scale
                [height, width, _] = frame.shape
                length = max((height, width))
                image = np.zeros((length, length, 3), np.float32)
                image[0:height, 0:width] = frame
                scale = length / 640

                # Create blob from image with RGB and normalized pixel values
                blob = cv2.dnn.blobFromImage(image, scalefactor=1.0 / 255, size=(640, 640), swapRB=True)

                # Predict faces
                YOLO.setInput(blob)
                outputs = YOLO.forward()
                outputs = np.array([cv2.transpose(outputs[0])])
                rows = outputs.shape[1]

                boxes = []
                scores = []

                # Iterate through output to collect bounding boxes and confidence scores
                for i in range(rows):
                    classes_scores = outputs[0][i][4:]
                    (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                    if maxScore >= 0.25:
                        # Convert x_center, y_center, w, h to x_min, y_min, w, h
                        box = [
                            outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                            outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                            outputs[0][i][2],
                            outputs[0][i][3],
                        ]
                        boxes.append(box)
                        scores.append(maxScore)

                # Apply NMS (Non-maximum suppression) to remove duplicate boxes
                result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

                embeddings = []

                # Iterate through the NMS results for VGG embeddings
                for i in range(len(result_boxes)):
                    index = result_boxes[i]
                    box = boxes[index]
                    # Get subframe of face with x_top_left, y_top_left, w, h
                    subframe = crop_image_by_box(
                        frame,
                        round(box[0] * scale),
                        round(box[1] * scale),
                        round(box[2] * scale),
                        round(box[3] * scale)
                    )
                    # Convert image to square``
                    [height, width, _] = subframe.shape
                    length = max((height, width))
                    subimage = np.zeros((length, length, 3), np.float32)
                    subimage[0:height, 0:width] = subframe
                    subimage = cv2.resize(subimage, (224, 224))
                    subimage = cv2.cvtColor(subimage, cv2.COLOR_BGR2RGB)
                    blob = cv2.dnn.blobFromImage(subimage, scalefactor=1.0/255, mean=(0.485, 0.456, 0.406))
                    # Predict faces
                    VGG.setInput(blob)
                    outputs = VGG.forward()
                    embeddings.append(outputs.flatten())

                # Flatten the features and append to the list
                for vec in embeddings: # TODO: Multiple of the same person in an image?
                    X.append(vec)
                    y.append(name)

            except Exception as e:
                print(f"Failed for image: {path}")
                print(e)

# Convert features and labels into numpy arrays
X, y = np.array(X), np.array(y)

# Perform KNN classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Save the KNN model
joblib.dump(knn, "knn.joblib")

# Perform PCA to reduce the features to 2D for visualization
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

# Show the plot
plt.show()
plt.savefig("PCA.png")
