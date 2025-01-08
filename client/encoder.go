package main

import (
	"gocv.io/x/gocv"
	"image"
)

// Encoder struct holds the network used for encoding images into embeddings.
type Encoder struct {
	Net gocv.Net // Deep learning model for feature extraction
}

// NewEncoder initializes and returns a new Encoder instance with the provided network.
func NewEncoder(net gocv.Net) Encoder {
	return Encoder{
		Net: net,
	}
}

// Encode processes a list of bounding boxes and extracts embeddings for each detected object in the image.
// It returns a slice of embeddings, each representing a detected object.
func (e *Encoder) Encode(img *gocv.Mat, boxes []image.Rectangle, indices []int) [][]float64 {
	var embeddings [][]float64

	// Iterate over each index in the list of selected boxes
	for i := range indices {
		index := indices[i]
		rect := boxes[index]    // Get the bounding box for the current object
		roi := img.Region(rect) // Extract the region of interest (ROI) from the image

		// Get the dimensions of the ROI
		height, width := roi.Rows(), roi.Cols()
		maxDim := max(height, width)

		// Create a square Mat of the largest dimension to avoid distortion during resizing
		square := gocv.NewMatWithSize(maxDim, maxDim, gocv.MatTypeCV8UC3)

		// Create a region within the square Mat to copy the ROI data into
		roi2 := square.Region(image.Rect(0, 0, width, height))

		// Copy the ROI into the square Mat
		roi.CopyTo(&roi2)

		// Release the original ROI Mat to free up memory
		roi.Close()

		// Normalize the image (mean subtraction and scaling) and preprocess for input to the network
		blob := gocv.BlobFromImage(square, 1.0/255.0, image.Pt(224, 224), gocv.NewScalar(0.485, 0.456, 0.406, 0), false, false)

		// Set the processed image blob as input to the network
		e.Net.SetInput(blob, "")

		// Perform a forward pass through the network to get the output
		results := e.Net.Forward("")

		// Process the network output into an embedding (feature vector)
		embedding := FormatResultsResNet(results)

		// Append the embedding to the list of embeddings
		embeddings = append(embeddings, embedding)

		// Release the square Mat to free up memory after processing
		square.Close()
	}

	return embeddings
}

// FormatResultsResNet converts the output of the ResNet model into a slice of float64 values representing the embedding.
func FormatResultsResNet(r gocv.Mat) []float64 {
	var vec []float64

	// Iterate through the columns of the result matrix to extract the embedding
	for x := 0; x < r.Cols(); x++ {
		vec = append(vec, float64(r.GetFloatAt(0, x))) // Convert each value to float64
	}

	return vec
}
