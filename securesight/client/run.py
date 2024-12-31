import cv2
import json
import numpy as np
import joblib
import requests
from shared.face_detector import FaceDetector
from shared.face_encoder import FaceEncoder

def draw_bounding_box(img, label, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        label (string): The label of the box.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{label}, ({confidence:.2f})"
    color = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == '__main__':
    detector = FaceDetector()
    encoder = FaceEncoder()
    KNN = joblib.load("./shared/weights/knn.joblib")

    cap = cv2.VideoCapture("./model/data/video.mp4")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        # Detect faces
        boxes, nms, scores, scale = detector(frame)

        # Encode faces
        embeddings = encoder(frame, boxes, nms, scale)

        if len(embeddings) > 0:
            # Predict names using KNN
            predictions = KNN.predict(embeddings)
            # Encrypt embeddings and call API
            #encrypted_embeddings = encryptor.encrypt(embeddings)
            response = requests.post(
                "http://localhost:8080/api/knn",
                json={"embeddings": json.dumps(list(vec.tolist() for vec in embeddings))}
            )
            if response.status_code == 200:
                print(response.text)
            else:
                print(f"Error: {response.status_code}")

        detections = []

        # Iterate through NMS results to prepare detections
        for i, index in enumerate(nms):
            box = boxes[index]
            detection = {
                "label": predictions[i],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)

            draw_bounding_box(
                frame,
                detection['label'],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        # Display the image with bounding boxes
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
