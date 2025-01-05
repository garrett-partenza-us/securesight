package main

import (
	"gocv.io/x/gocv"
	"image"
)

type Encoder struct {
	Net gocv.Net
}

func NewEncoder(net gocv.Net) Encoder {
	return Encoder{
		Net: net,
	}
}

func (e *Encoder) Encode(img *gocv.Mat, boxes []image.Rectangle, indices []int) [][]float64 {
	var embeddings [][]float64
	for i := range indices {
		index := indices[i]
		rect := boxes[index]
		roi := img.Region(rect)

		height, width := roi.Rows(), roi.Cols()
		maxDim := max(height, width)

		// Create a new square Mat for each image in the loop
		square := gocv.NewMatWithSize(maxDim, maxDim, gocv.MatTypeCV8UC3)

		// Create a region (roi2) within the square Mat to copy the image data into
		roi2 := square.Region(image.Rect(0, 0, width, height))

		// Copy the region of interest (ROI) into the square
		roi.CopyTo(&roi2)

		// Close roi after use to release resources
		roi.Close()

		// Normalize and preprocess the image using BlobFromImage
		blob := gocv.BlobFromImage(square, 1.0/255.0, image.Pt(224, 224), gocv.NewScalar(0.485, 0.456, 0.406, 0), false, false)

		// Pass the blob through the network
		e.Net.SetInput(blob, "")
		results := e.Net.Forward("")

		// Process the results to obtain the embedding
		embedding := FormatResultsResNet(results)

		// Append the embedding to the list of embeddings
		embeddings = append(embeddings, embedding)

		// Free up memory by closing the square Mat after use
		square.Close()
	}
	return embeddings
}

func FormatResultsResNet(r gocv.Mat) []float64 {

	var vec []float64
	for x := 0; x < r.Cols(); x++ {
		vec = append(vec, float64(r.GetFloatAt(0, x)))
	}
	return vec
}
