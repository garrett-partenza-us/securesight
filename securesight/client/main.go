package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"bytes"
	"io/ioutil"
	"sort"
	"log"
	"image"
	"image/color"
	"gocv.io/x/gocv"
)

var pca PCA

func DetectFaces(m *gocv.Net, i *gocv.Mat) ([]image.Rectangle, []float32, []int, float32) {
	height, width := i.Rows(), i.Cols()
	maxDim := max(height, width)
	square := gocv.NewMatWithSize(maxDim, maxDim, gocv.MatTypeCV8UC3)
	roi := square.Region(image.Rect(0, 0, width, height))
	i.CopyTo(&roi)
	roi.Close()
	scale := float32(maxDim) / 640.0
	blob := gocv.BlobFromImage(square, 1.0/255.0, image.Pt(640, 640), gocv.NewScalar(0, 0, 0, 0), true, false)
	// Inference
	m.SetInput(blob, "")
	results := m.Forward("")
	boxes, scores, indices := FormatResultsYOLO(&results, scale)
	return boxes, scores, indices, scale
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

				x1 := int((x-w/2) * scale)
				y1 := int((y-h/2) * scale)
				x2 := int((x+w/2) * scale)
				y2 := int((y+h/2) * scale)

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

func EncodeFaces(img *gocv.Mat, m *gocv.Net, boxes []image.Rectangle, indices []int) [][]float32 {



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
		m.SetInput(blob, "")
		results := m.Forward("")
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

type ResponseData struct {
    Distances [][]float32 `json:"Distances"`
    Classes   []string    `json:"Classes"`
}

func getKNNPredictions(embeddings [][]float32) ([]string, error) {

		embeddings = applyPCA(embeddings, pca)

    jsonData, err := json.Marshal(embeddings)
    if err != nil {
        return nil, fmt.Errorf("Error marshaling embeddings: %v", err)
    }

    url := "http://localhost:8080/api/knn"
    resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, fmt.Errorf("Error making POST request: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("Error: received status code %d", resp.StatusCode)
    }

    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("Error reading response body: %v", err)
    }

    var responseData ResponseData
    err = json.Unmarshal(body, &responseData)
    if err != nil {
        return nil, fmt.Errorf("Error unmarshaling response data: %v", err)
    }

    predictions := []string{}
    for _, distances := range responseData.Distances {

        zipped := make([][2]interface{}, len(distances))
        for i, distance := range distances {
            zipped[i] = [2]interface{}{distance, responseData.Classes[i]}
        }

        sort.Slice(zipped, func(i, j int) bool {
            return zipped[i][0].(float32) < zipped[j][0].(float32)
        })

        predictions = append(predictions, zipped[0][1].(string))
    }

    return predictions, nil
}

// Function to apply PCA transformation
func applyPCA(data [][]float32, pca PCA) [][]float32 {
	var transformedData [][]float32

	for _, row := range data {
		// Subtract the mean from each feature in the row
		subtractedRow := make([]float32, len(row))
		for i := range row {
			subtractedRow[i] = row[i] - pca.Mean[i]
		}

		// Apply the transformation (multiply by the principal components)
		transformedRow := make([]float32, len(pca.Components))
		for i := range pca.Components {
			for j := range subtractedRow {
				transformedRow[i] += subtractedRow[j] * pca.Components[i][j]
			}
		}
		transformedData = append(transformedData, transformedRow)
	}

	return transformedData
}

// PCA model representation
type PCA struct {
	Components [][]float32 `json:"components"`
	Mean       []float32   `json:"mean"`
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

	// Load the PCA components from the JSON file
	data, err := ioutil.ReadFile("../shared/weights/pca_components.json")
	if err != nil {
		log.Fatalf("Error reading PCA components file: %v", err)
	}

	err = json.Unmarshal(data, &pca)
	if err != nil {
		log.Fatalf("Error unmarshalling PCA components: %v", err)
	}

	for {

		// Capture frame
		webcam.Read(&img)

		// Detect face bounding boxes
		boxes, scores, indices, scale := DetectFaces(&yolo_net, &img)
		fmt.Println(len(boxes), len(scores), len(indices), scale)
		embeddings := EncodeFaces(&img, &resnet_net, boxes, indices)
		predictions, err := getKNNPredictions(embeddings)
		if err != nil {
			panic(err)
			return
		}

		for i := 0; i < len(predictions); i++ {
			rect := boxes[i]
			gocv.Rectangle(&img, rect, color.RGBA{0, 255, 0, 0}, 3)
			rectCenter := image.Pt((rect.Min.X + rect.Max.X) / 2, (rect.Min.Y + rect.Max.Y) / 2)
			text := predictions[i]
			fontFace := gocv.FontHersheySimplex
			fontScale := 1.2
			thickness := 2
			textSize := gocv.GetTextSize(text, fontFace, fontScale, thickness)
			textX := rectCenter.X - textSize.X/2
			textY := rect.Min.Y + textSize.Y + 10 // 10px offset from the top edge
			gocv.PutText(&img, text, image.Pt(textX, textY), fontFace, fontScale, color.RGBA{0, 255, 0, 0}, thickness)
		}

		window.IMShow(img)
		window.WaitKey(1)
	}
}
