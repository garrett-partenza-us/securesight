import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import joblib


def crop_image_by_box(image, x, y, w, h):
    # Crop the image using numpy array slicing
    x2 = x + w
    y2 = y + h
    cropped_image = image[y:y2, x:x2] 
    return cropped_image

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

    VGG = cv2.dnn.readNetFromONNX('weights/inception_resnet_v1.onnx')
    YOLO = cv2.dnn.readNetFromONNX('weights/yolov11n-face.onnx')
    KNN = joblib.load("knn.joblib")

    cap = cv2.VideoCapture("data/video.mp4")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        [height, width, _] = frame.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = frame
        scale = length / 640

        # Convert to RGB and normalize the image
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0 / 255, size=(640, 640), swapRB=True)

        # Predict faces
        YOLO.setInput(blob)
        outputs = YOLO.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)

        # Apply NMS (Non-maximum suppression)
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
        if embeddings:
            embeddings = np.array(embeddings)
            predictions = KNN.predict(embeddings)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
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

        continue


        boxes = results.boxes.xywh.tolist()

        resized_imgs = []

        for xywh in boxes:

            face = crop_image_by_box(frame, xywh)

            resized_img = resize_transform(face) 

            tensor_img = transforms.ToTensor()(resized_img)

            resized_imgs.append(tensor_img)

        if resized_imgs:

            resized_imgs_tensor = torch.stack(resized_imgs, dim=0, out=None)

            features = vgg(resized_imgs_tensor)

            features = features.view(features.size(0), 512).numpy()

            names = knn.predict(features)

            for idx, name in enumerate(names):

                results.names[idx] = name
    
        for idx, xywh in enumerate(boxes):

            # Draw the bounding box on the original image using OpenCV
            x, y, w, h = xywh
            xmin = int(x - w / 2)
            ymin = int(y - h / 2)
            xmax = int(x + w / 2)
            ymax = int(y + h / 2)

            # Draw rectangle around the face
            color = (0, 255, 0)  # Green color for the rectangle
            thickness = 2
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)

            # Get the name for the label from the KNN prediction
            label = names[idx]

            # Add label text above the rectangle (using cv2.putText)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = xmin
            text_y = ymin - 10  # Position text above the rectangle

            # Draw the label background
            label_background = (255, 255, 255)  # White background for the label
            cv2.rectangle(frame, (xmin, text_y - text_size[1]), 
                          (xmin + text_size[0], text_y + 5), label_background, -1)

            # Draw the label text
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

        cv2.imshow("YOLO Face Detection", frame)

    cap.release()
    cap.destroyAllWindows()
