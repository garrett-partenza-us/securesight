package main

import (
	"gocv.io/x/gocv"
	"image"
)

// Detector struct holds the network used for object detection.
type Detector struct {
	Net gocv.Net // Deep learning model for inference
}

// NewDetector initializes and returns a new Detector instance with the provided network.
func NewDetector(net gocv.Net) Detector {
	return Detector{
		Net: net,
	}
}

// Detect performs object detection on an input image (Mat).
// It returns the bounding boxes, confidence scores, and indices of detected objects.
func (d *Detector) Detect(i *gocv.Mat) ([]image.Rectangle, []float32, []int) {
	// Get the height and width of the input image
	height, width := i.Rows(), i.Cols()

	// Calculate max dimension to create a square image for resizing
	maxDim := max(height, width)

	// Create a square Mat with size based on the max dimension
	square := gocv.NewMatWithSize(maxDim, maxDim, gocv.MatTypeCV8UC3)

	// Copy the input image into the square region
	roi := square.Region(image.Rect(0, 0, width, height))
	i.CopyTo(&roi)
	roi.Close()

	// Calculate the scaling factor to resize the image to 640x640 for inference
	scale := float32(maxDim) / 640.0

	// Create a blob from the image for input to the network
	blob := gocv.BlobFromImage(square, 1.0/255.0, image.Pt(640, 640), gocv.NewScalar(0, 0, 0, 0), true, false)

	// Set the blob as the input to the network
	d.Net.SetInput(blob, "")

	// Perform inference (forward pass)
	results := d.Net.Forward("")

	// Format the results into bounding boxes, scores, and indices
	boxes, scores, indices := FormatResultsYOLO(&results, scale)

	return boxes, scores, indices
}

// FormatResultsYOLO processes the YOLO model's output into bounding boxes, confidence scores, and indices.
// It uses a threshold to filter out low-confidence detections.
func FormatResultsYOLO(m *gocv.Mat, scale float32) ([]image.Rectangle, []float32, []int) {
	var boxes []image.Rectangle // List of bounding boxes for detected objects
	var scores []float32        // Confidence scores corresponding to each box

	// Iterate over each detected object (assuming YOLO v3 output format)
	for boxInfo := 0; boxInfo < 5; boxInfo++ {
		for subSection := 0; subSection < 8400; subSection++ {
			// Extract confidence score from model output
			confidence := m.GetFloatAt3(0, 4, subSection)

			// If confidence exceeds threshold, process the detection
			if confidence >= 0.25 {
				// Extract bounding box coordinates from the model output
				x := m.GetFloatAt3(0, 0, subSection)
				y := m.GetFloatAt3(0, 1, subSection)
				w := m.GetFloatAt3(0, 2, subSection)
				h := m.GetFloatAt3(0, 3, subSection)

				// Convert YOLO box format to image.Rectangle (scaled to original size)
				x1 := int((x - w/2) * scale)
				y1 := int((y - h/2) * scale)
				x2 := int((x + w/2) * scale)
				y2 := int((y + h/2) * scale)

				// Append the bounding box and its confidence score
				box := image.Rect(x1, y1, x2, y2)
				boxes = append(boxes, box)
				scores = append(scores, float32(confidence))
			}
		}
	}

	// Perform Non-Maximum Suppression (NMS) to eliminate overlapping boxes
	indices := make([]int, len(boxes))
	if len(boxes) > 0 {
		indices = gocv.NMSBoxes(boxes, scores, 0.5, 0.4)
	}

	return boxes, scores, indices
}

