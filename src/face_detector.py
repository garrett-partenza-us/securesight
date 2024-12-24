import cv2
import numpy as np


class FaceDetector:
    """
    A face detection class using a YOLO-based ONNX model.

    Attributes:
        model (cv2.dnn_Net): The neural network model loaded from an ONNX file.
        imsize (int): The size to which input images will be resized.
        pixelscale (float): Scaling factor for pixel values.
    """

    def __init__(self, onnx='../weights/yolov11n-face.onnx', imsize=640, pixelscale=1.0/255):
        """
        Initializes the FaceDetector with the given ONNX model and preprocessing parameters.

        Args:
            onnx (str): Path to the ONNX model file.
            imsize (int): Size to resize input images (default: 640).
            pixelscale (float): Scaling factor for image pixel values (default: 1.0/255).
        """
        self.model = cv2.dnn.readNetFromONNX(onnx)
        self.imsize = imsize
        self.pixelscale = pixelscale

    def __call__(self, image: np.ndarray):
        """
        Detects faces in the provided image.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            tuple: A tuple containing:
                - boxes (list): List of bounding boxes for detected faces.
                - nms (list): Indices of boxes kept after non-maximum suppression.
                - scale (float): The scaling factor used for preprocessing.
        """
        blob, scale = self.preprocess(image)
        self.model.setInput(blob)
        output = self.model.forward()
        boxes, nms, scores = self.format_output(output)
        return boxes, nms, scores, scale

    def preprocess(self, image: np.ndarray):
        """
        Preprocesses the input image for the neural network.

        Args:
            image (np.ndarray): The input image.

        Returns:
            tuple: A tuple containing:
                - blob (np.ndarray): The preprocessed image blob.
                - scale (float): The scaling factor for resizing.
        """
        height, width, channels = image.shape
        size = max(height, width)
        square = np.zeros((size, size, 3), np.uint8)
        square[0:height, 0:width] = image
        scale = size / self.imsize
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=self.pixelscale,
            size=(self.imsize, self.imsize),
            swapRB=True
        )
        return blob, scale

    def format_output(self, output: np.ndarray):
        """
        Processes the raw model output to extract bounding boxes and apply non-maximum suppression.

        Args:
            output (np.ndarray): The raw output from the neural network.

        Returns:
            tuple: A tuple containing:
                - boxes (list): List of bounding boxes as [x_min, y_min, width, height].
                - nms (list): Indices of boxes kept after non-maximum suppression.
        """
        output = np.array([cv2.transpose(output[0])])
        rows = output.shape[1]
        boxes = []
        scores = []

        # Iterate through output to collect bounding boxes and confidence scores
        for i in range(rows):
            classes_scores = output[0][i][4:]
            (_, max_score, _, _) = cv2.minMaxLoc(classes_scores)
            if max_score >= 0.25:
                # Convert to x_min, y_min, width, height
                box = [
                    output[0][i][0] - (0.5 * output[0][i][2]),
                    output[0][i][1] - (0.5 * output[0][i][3]),
                    output[0][i][2],
                    output[0][i][3],
                ]
                boxes.append(box)
                scores.append(max_score)

        # Apply NMS (Non-maximum suppression)
        nms = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        return boxes, nms, scores

