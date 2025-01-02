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

func (e *Encoder) Encode(img *gocv.Mat, boxes []image.Rectangle, indices []int) [][]float32 {

	var embeddings [][]float32
	for i := range indices {
		index := indices[i]
		rect := boxes[index]
		roi := img.Region(rect)

		height, width := roi.Rows(), roi.Cols()
		maxDim := max(height, width)
		square := gocv.NewMatWithSize(maxDim, maxDim, gocv.MatTypeCV8UC3)
		roi2 := square.Region(image.Rect(0, 0, width, height))
		img.CopyTo(&roi2)
		roi.Close()
		blob := gocv.BlobFromImage(square, 1.0/255.0, image.Pt(224, 224), gocv.NewScalar(0.485, 0.456, 0.406, 0), false, false)
		e.Net.SetInput(blob, "")
		results := e.Net.Forward("")
		embedding := FormatResultsResNet(&results)
		embeddings = append(embeddings, embedding)
	}

	return embeddings

}

func FormatResultsResNet(r *gocv.Mat) []float32 {

	var vec []float32
	for x := 0; x < r.Cols(); x++ {
		vec = append(vec, r.GetFloatAt(0, x))
	}
	return vec
}
