package main

import (
	"fmt"
	"image"
	"image/color"
	"gocv.io/x/gocv"
)

func main() {

	webcam, _ := gocv.VideoCaptureDevice(0)
	window := gocv.NewWindow("Webcam")
	img := gocv.NewMat()

	yolo_path := "../shared/weights/yolov11n-face.onnx"
	yolo_net := gocv.ReadNet(yolo_path, "")
	if yolo_net.Empty() {
		fmt.Println("Failed to load YOLO model.")
	}
	defer yolo_net.Close()
	detector := NewDetector(yolo_net)

	resnet_path := "../shared/weights/inception_resnet_v1.onnx"
	resnet_net := gocv.ReadNet(resnet_path, "")
	if resnet_net.Empty() {
		fmt.Println("Failed to load ResNet model.")
	}
	defer resnet_net.Close()
	encoder := NewEncoder(resnet_net)

	pca := NewPCA("../shared/weights/pca_components.json")

	for {

		webcam.Read(&img)

		boxes, _, indices := detector.Detect(&img)
		embeddings := encoder.Encode(&img, boxes, indices)
		embeddings = pca.Transform(embeddings)
		predictions, err := CallAPI(embeddings)
		if err != nil {
			panic(err)
			return
		}

		DrawBoxes(&img, predictions, boxes)


		window.IMShow(img)
		window.WaitKey(1)
	}
}

func DrawBoxes(img *gocv.Mat, predictions []string, boxes []image.Rectangle) {

		for i := 0; i < len(predictions); i++ {
			rect := boxes[i]
			gocv.Rectangle(img, rect, color.RGBA{0, 255, 0, 0}, 3)
			rectCenter := image.Pt((rect.Min.X + rect.Max.X) / 2, (rect.Min.Y + rect.Max.Y) / 2)
			text := predictions[i]
			fontFace := gocv.FontHersheySimplex
			fontScale := 1.2
			thickness := 2
			textSize := gocv.GetTextSize(text, fontFace, fontScale, thickness)
			textX := rectCenter.X - textSize.X/2
			textY := rect.Min.Y + textSize.Y + 10 // 10px offset from the top edge
			gocv.PutText(img, text, image.Pt(textX, textY), fontFace, fontScale, color.RGBA{0, 255, 0, 0}, thickness)
		}
}
