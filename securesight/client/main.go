package main

import (
	"fmt"
	"strings"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"sort"
	"time"
)

func main() {

	fmt.Println(strings.Repeat("-", 20)+"\nStarting client...\n"+strings.Repeat("-", 20))
	// Open the video file using VideoCaptureFile
	videoFile := "../model/data/video.mp4"
	webcam, err := gocv.VideoCaptureFile(videoFile)
	if err != nil {
		panic(err)
	}
	defer webcam.Close()

	// Create a window to display the video
	window := gocv.NewWindow("Video Playback")
	defer window.Close()
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

	encryptor := NewEncryptor()

	pca := NewPCA("../shared/weights/pca_components.json")

	for {

		fmt.Println(strings.Repeat("-", 20)+"\nProcessing frame...\n"+strings.Repeat("-", 20))
		webcam.Read(&img)
		
		startTime := time.Now()

		boxes, _, indices := detector.Detect(&img)
		embeddings := encoder.Encode(&img, boxes, indices)
		embeddings = pca.Transform(embeddings)
		var ciphertexts []rlwe.Ciphertext
		for idx := range embeddings {
			ciphertext := encryptor.Encrypt(embeddings[idx])
			ciphertexts = append(ciphertexts, ciphertext)
		}
		publicContext := encryptor.NewPublicContext(ciphertexts)
		serializedPublicContext, err := SerializeObject(publicContext)
		if err != nil {
			panic(err)
		}
		responseData, err := CallAPI(serializedPublicContext)
		if err != nil {
			panic(err)
		}
		distances := encryptor.Decrypt(responseData.Distances)
		predictions, err := DistancesToClasses(distances, responseData.Classes)

		DrawBoxes(&img, predictions, boxes, indices)

		elapsedTime := time.Since(startTime)
		fmt.Println("Total time to process frame: ", elapsedTime.Milliseconds())

		window.IMShow(img)
		window.WaitKey(1)
	}
}

func DrawBoxes(img *gocv.Mat, predictions []string, boxes []image.Rectangle, indices []int) {

	for i := 0; i < len(indices); i++ {
		rect := boxes[indices[i]]
		gocv.Rectangle(img, rect, color.RGBA{0, 255, 0, 0}, 3)
		rectCenter := image.Pt((rect.Min.X+rect.Max.X)/2, (rect.Min.Y+rect.Max.Y)/2)
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

func DistancesToClasses(d [][]float64, c []string) ([]string, error) {
	predictions := []string{}
	for _, distances := range d {

		zipped := make([][2]interface{}, len(distances))
		for i, distance := range distances {
			zipped[i] = [2]interface{}{distance, c[i]}
		}

		sort.Slice(zipped, func(i, j int) bool {
			return zipped[i][0].(float64) < zipped[j][0].(float64)
		})

		predictions = append(predictions, zipped[0][1].(string))
	}

	return predictions, nil
}
