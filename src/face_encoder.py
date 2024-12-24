import cv2
import numpy as np

class FaceEncoder:
    """
    A face encoder class using an ONNX model to generate embeddings for detected faces.

    Attributes:
        model (cv2.dnn_Net): The neural network model loaded from an ONNX file.
        imsize (int): The size to which cropped face images will be resized.
        pixelscale (float): Scaling factor for pixel values.
    """

    def __init__(self, onnx='../weights/inception_resnet_v1.onnx', imsize=224, pixelscale=1.0/255):
        """
        Initializes the FaceEncoder with the given ONNX model and preprocessing parameters.

        Args:
            onnx (str): Path to the ONNX model file.
            imsize (int): Size to resize cropped face images (default: 224).
            pixelscale (float): Scaling factor for image pixel values (default: 1.0/255).
        """
        self.model = cv2.dnn.readNetFromONNX(onnx)
        self.imsize = imsize
        self.pixelscale = pixelscale

    def __call__(self, image: np.ndarray, boxes: np.ndarray, nms: list, scale: float):
        """
        Generates face embeddings for the provided image and detected face bounding boxes.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            boxes (np.ndarray): Array of bounding boxes for detected faces.
            nms (list): Indices of boxes kept after non-maximum suppression.
            scale (float): The scaling factor used for preprocessing.

        Returns:
            list: A list of embeddings for the detected faces.
        """
        embeddings = []

        for i in range(len(nms)):
            index = nms[i]
            box = boxes[index]
            subframe = self.crop_image_by_box(
                image,
                round(box[0] * scale),
                round(box[1] * scale),
                round(box[2] * scale),
                round(box[3] * scale)
            )
            blob = self.preprocess(subframe)
            self.model.setInput(blob)
            output = self.model.forward()
            embeddings.append(output.flatten())

        return embeddings

    def preprocess(self, image: np.ndarray):
        """
        Preprocesses the cropped face image for the neural network.

        Args:
            image (np.ndarray): The cropped face image.

        Returns:
            np.ndarray: The preprocessed image blob.
        """
        height, width, _ = image.shape
        size = max(height, width)
        square = np.zeros((size, size, 3), np.float32)
        square[0:height, 0:width] = image
        square = cv2.resize(square, (self.imsize, self.imsize))
        square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(square, scalefactor=self.pixelscale, mean=(0.485, 0.456, 0.406))
        return blob

    def crop_image_by_box(self, image: np.ndarray, x: int, y: int, w: int, h: int):
        """
        Crops a region from the image based on the provided bounding box.

        Args:
            image (np.ndarray): The input image.
            x (int): X-coordinate of the top-left corner of the box.
            y (int): Y-coordinate of the top-left corner of the box.
            w (int): Width of the bounding box.
            h (int): Height of the bounding box.

        Returns:
            np.ndarray: The cropped image.
        """
        x2 = x + w
        y2 = y + h
        cropped_image = image[y:y2, x:x2]
        cv2.imwrite(f"faces/{x}{y}.jpg", cropped_image)
        return cropped_image
