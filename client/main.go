package main

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"sort"
	"strings"
	"time"
)

func main() {

	// Print a start message with a visual separator
	fmt.Println(strings.Repeat("-", 20) + "\nStarting client...\n" + strings.Repeat("-", 20))

	// Open the video file for processing
	videoFile := "../video.mp4"
	webcam, err := gocv.VideoCaptureFile(videoFile)
	if err != nil {
		panic(err) // Panic if video file can't be opened
	}
	defer webcam.Close() // Ensure the webcam is closed after processing

	// Create a window for video playback
	window := gocv.NewWindow("Video Playback")
	defer window.Close()
	img := gocv.NewMat() // Initialize an empty image matrix for each frame

	// Load YOLO model for object detection
	yolo_path := "../weights/yolov11n-face.onnx"
	yolo_net := gocv.ReadNet(yolo_path, "")
	if yolo_net.Empty() {
		fmt.Println("Failed to load YOLO model.") // Handle error if YOLO model fails to load
	}
	defer yolo_net.Close()
	detector := NewDetector(yolo_net) // Create detector using YOLO model

	// Load ResNet model for feature extraction (embeddings)
	resnet_path := "../weights/inception_resnet_v1.onnx"
	resnet_net := gocv.ReadNet(resnet_path, "")
	if resnet_net.Empty() {
		fmt.Println("Failed to load ResNet model.") // Handle error if ResNet model fails to load
	}
	defer resnet_net.Close()
	encoder := NewEncoder(resnet_net) // Create encoder using ResNet model

	// Initialize encryptor for encrypting embeddings
	encryptor := NewEncryptor()

	// Load PCA model for dimensionality reduction (if needed)
	pca := NewPCA("../weights/pca_components.json")
	_ = pca // PCA isn't currently used, but can be enabled if required

	// Start processing video frames
	for {
		// Print message for processing current frame
		fmt.Println(strings.Repeat("-", 20) + "\nProcessing frame...\n" + strings.Repeat("-", 20))

		// Read the next frame from the webcam
		webcam.Read(&img)

		// Track time taken for processing the current frame
		startTime := time.Now()

		// Detect objects in the frame using YOLO (bounding boxes, indices)
		boxes, _, indices := detector.Detect(&img)

		// Extract embeddings (feature vectors) for the detected objects using ResNet
		embeddings := encoder.Encode(&img, boxes, indices)

		// Optional: Apply PCA for dimensionality reduction on embeddings (commented out here)
		// embeddings = pca.Transform(embeddings)

		// Encrypt the embeddings before sending them to the server
		var ciphertexts []rlwe.Ciphertext
		for idx := range embeddings {
			ciphertext := encryptor.Encrypt(embeddings[idx]) // Encrypt each embedding
			ciphertexts = append(ciphertexts, ciphertext)
		}

		// Create public context from encrypted ciphertexts
		publicContext := encryptor.NewPublicContext(ciphertexts)

		// Serialize the public context to send to the server
		serializedPublicContext, err := SerializeObject(publicContext)
		if err != nil {
			panic(err) // Handle error if serialization fails
		}

		// Send the serialized public context to the API and receive the response
		responseData, err := CallAPI(serializedPublicContext)
		if err != nil {
			panic(err) // Handle error if API call fails
		}

		// Decrypt the response data (distances and classes) from the server
		distances, classes := encryptor.Decrypt(responseData.Distances, responseData.Params)

		// Convert the distances into predicted classes based on nearest neighbors
		predictions, err := DistancesToClasses(distances, classes)

		// Draw the bounding boxes and predicted classes on the image
		DrawBoxes(&img, predictions, boxes, indices)

		// Calculate and print the total time taken to process the frame
		elapsedTime := time.Since(startTime)
		fmt.Println("Total time to process frame: ", elapsedTime.Milliseconds())

		// Display the processed frame in the window
		window.IMShow(img)
		window.WaitKey(1) // Wait for a key press (needed for proper window handling)
	}
}

// DrawBoxes overlays bounding boxes and predicted class labels on the image.
func DrawBoxes(img *gocv.Mat, predictions []string, boxes []image.Rectangle, indices []int) {
	for i := 0; i < len(indices); i++ {
		rect := boxes[indices[i]]                              // Get the bounding box for the current detection
		gocv.Rectangle(img, rect, color.RGBA{0, 255, 0, 0}, 3) // Draw the rectangle (green)

		// Calculate the center of the bounding box to position the text
		rectCenter := image.Pt((rect.Min.X+rect.Max.X)/2, (rect.Min.Y+rect.Max.Y)/2)
		text := predictions[i] // The predicted class for the object
		fontFace := gocv.FontHersheySimplex
		fontScale := 1.2
		thickness := 2
		textSize := gocv.GetTextSize(text, fontFace, fontScale, thickness)

		// Position the text above the bounding box
		textX := rectCenter.X - textSize.X/2
		textY := rect.Min.Y + textSize.Y + 10 // 10px offset from the top edge
		gocv.PutText(img, text, image.Pt(textX, textY), fontFace, fontScale, color.RGBA{0, 255, 0, 0}, thickness)
	}
}

// DistancesToClasses converts distances to predicted class labels using nearest neighbors.
func DistancesToClasses(d [][]float64, c [][]string) ([]string, error) {
	predictions := []string{}

	// Iterate over each query and its associated distances
	for q, distances := range d {

		// Zip distances with their corresponding class labels
		zipped := make([][2]interface{}, len(distances))
		for i, distance := range distances {
			zipped[i] = [2]interface{}{distance, c[q][i]} // Pair each distance with its class
		}

		// Sort by distance in ascending order (nearest neighbors first)
		sort.Slice(zipped, func(i, j int) bool {
			return zipped[i][0].(float64) < zipped[j][0].(float64)
		})

		// Select top-k closest neighbors
		k := 5
		var classes []string
		for i := 0; i < k; i++ {
			classes = append(classes, zipped[i][1].(string)) // Add the class label of the neighbor
		}

		// Choose the most common class from the top-k neighbors (majority vote)
		predictions = append(predictions, mostCommonClass(classes, k))
	}

	return predictions, nil
}

// mostCommonClass returns the most common class label from a slice of classes (majority voting).
func mostCommonClass(classes []string, k int) string {
	frequencyMap := make(map[string]int)

	// Count the frequency of each class
	for _, class := range classes {
		frequencyMap[class]++
	}

	// Find the most frequent class
	var mostCommon string
	maxCount := 0
	for class, count := range frequencyMap {
		if count > maxCount {
			mostCommon = class
			maxCount = count
		}
	}
	return mostCommon
}
