package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
)

func DetectFaces(m *gocv.Net, i *gocv.Mat) ([]image.Rectangle, []float32, []int, float64) {
	height, width := i.Rows(), i.Cols()
	maxDim := max(height, width)
	square := gocv.NewMatWithSize(maxDim, maxDim, gocv.MatTypeCV8UC3)
	roi := square.Region(image.Rect(0, 0, width, height))
	i.CopyTo(&roi)
	roi.Close()
	scale := float64(maxDim) / 640.0
	blob := gocv.BlobFromImage(square, 1.0/255.0, image.Pt(640, 640), gocv.NewScalar(0, 0, 0, 0), true, false)
	// Inference
	m.SetInput(blob, "")
	results := m.Forward("")
	boxes, scores, indices := FormatResults(&results)
	return boxes, scores, indices, scale
}

func FormatResults(m *gocv.Mat) ([]image.Rectangle, []float32, []int) {
	var boxes []image.Rectangle // Use []image.Rectangle
	var scores []float32        // Use []float32 for scores

	for boxInfo := 0; boxInfo < 5; boxInfo++ {
		for subSection := 0; subSection < 8400; subSection++ {
			confidence := m.GetFloatAt3(0, 4, subSection)
			if confidence >= 0.25 {
				// Convert gocv.Rect to image.Rectangle and append to boxes
				box := image.Rect(
					int(m.GetFloatAt3(0, 0, subSection)-(0.5*m.GetFloatAt3(0, 2, subSection))),
					int(m.GetFloatAt3(0, 1, subSection)-(0.5*m.GetFloatAt3(0, 3, subSection))),
					int(m.GetFloatAt3(0, 2, subSection)),
					int(m.GetFloatAt3(0, 3, subSection)),
				)
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

func main() {

	// Initalize video capture device
	webcam, _ := gocv.VideoCaptureDevice(0)
	window := gocv.NewWindow("Webcam")
	img := gocv.NewMat()

	// Load YOLO model
	yolo_path := "../shared/weights/yolov11n-face.onnx"
	yolo_net := gocv.ReadNet(yolo_path, "")
	if yolo_net.Empty() {
		fmt.Println("Failed to load YOLO model.")
	}
	defer yolo_net.Close()

	// Load ResNet Model
	resnet_path := "../shared/weights/inception_resnet_v1.onnx"
	resnet_net := gocv.ReadNet(resnet_path, "")
	if resnet_net.Empty() {
		fmt.Println("Failed to load ResNet model.")
	}
	defer resnet_net.Close()

	for {

		// Capture frame
		webcam.Read(&img)

		// Detect face bounding boxes
		boxes, scores, indices, scale := DetectFaces(&yolo_net, &img)
		fmt.Println(len(boxes), len(scores), len(indices), scale)

		window.IMShow(img)
		window.WaitKey(1)
	}
}
