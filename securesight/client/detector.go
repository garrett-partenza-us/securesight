package main

import (
	"gocv.io/x/gocv"
	"image"
)

type Detector struct {
	Net gocv.Net
}

func NewDetector(net gocv.Net) Detector {
	return Detector{
		Net: net,
	}
}

func (d *Detector) Detect(i *gocv.Mat) ([]image.Rectangle, []float32, []int) {
	height, width := i.Rows(), i.Cols()
	maxDim := max(height, width)
	square := gocv.NewMatWithSize(maxDim, maxDim, gocv.MatTypeCV8UC3)
	roi := square.Region(image.Rect(0, 0, width, height))
	i.CopyTo(&roi)
	roi.Close()
	scale := float32(maxDim) / 640.0
	blob := gocv.BlobFromImage(square, 1.0/255.0, image.Pt(640, 640), gocv.NewScalar(0, 0, 0, 0), true, false)
	// Inference
	d.Net.SetInput(blob, "")
	results := d.Net.Forward("")
	boxes, scores, indices := FormatResultsYOLO(&results, scale)
	return boxes, scores, indices
}

func FormatResultsYOLO(m *gocv.Mat, scale float32) ([]image.Rectangle, []float32, []int) {
	var boxes []image.Rectangle // Use []image.Rectangle
	var scores []float32        // Use []float32 for scores

	for boxInfo := 0; boxInfo < 5; boxInfo++ {
		for subSection := 0; subSection < 8400; subSection++ {
			confidence := m.GetFloatAt3(0, 4, subSection)
			if confidence >= 0.25 {
				// Convert gocv.Rect to image.Rectangle and append to boxes

				x := m.GetFloatAt3(0, 0, subSection)
				y := m.GetFloatAt3(0, 1, subSection)
				w := m.GetFloatAt3(0, 2, subSection)
				h := m.GetFloatAt3(0, 3, subSection)

				x1 := int((x - w/2) * scale)
				y1 := int((y - h/2) * scale)
				x2 := int((x + w/2) * scale)
				y2 := int((y + h/2) * scale)

				box := image.Rect(x1, y1, x2, y2)
				boxes = append(boxes, box)
				scores = append(scores, float32(confidence)) // Append the confidence score
			}
		}
	}

	indices := make([]int, len(boxes))
	if len(boxes) > 0 {
		indices = gocv.NMSBoxes(boxes, scores, 0.5, 0.4)
	}

	return boxes, scores, indices

}
