package main

import (
	"encoding/csv"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// Global variables
var model KNN             // KNN model containing training data and associated classes
var context PublicContext // PublicContext for managing the encryption context

// Response struct to define the format of the API response
type Response struct {
	Distances [][]Distance    `json:"Distances"` // Distance matrix for KNN predictions
	Classes   []string        `json:"Classes"`   // List of classes for KNN predictions
	Params    ckks.Parameters `json:"Params"`    // Parameters required for decryption
}

// KNN struct to represent the K-Nearest Neighbors model
type KNN struct {
	Data    [][]float64 // Matrix of data points (features) for KNN
	Classes []string    // Corresponding classes for each data point
}

// LoadKNN loads the KNN model from a CSV file.
// The CSV file should contain features as columns and the class as the last column.
func LoadKNN(path string) KNN {
	startTime := time.Now()

	// Open the CSV file
	file, err := os.Open(path)
	if err != nil {
		panic(err) // Panic if file can't be opened
	}
	defer file.Close()

	// Initialize the CSV reader
	reader := csv.NewReader(file)

	var matrix [][]float64
	var classes []string

	// Read the CSV file line by line
	for {
		record, err := reader.Read()
		if err != nil {
			break // Break the loop when the file ends or on error
		}
		var row []float64

		// Convert each feature in the record to a float and append it to the row
		for i := 0; i < len(record)-1; i++ {
			f, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				panic(err) // Panic if a feature can't be parsed as a float
			}
			row = append(row, f)
		}
		matrix = append(matrix, row) // Add the row of features to the data matrix
		class := record[len(record)-1]
		classes = append(classes, class) // Add the class (last column) to the classes list
	}

	// Print out the shape of the KNN model (number of rows and columns)
	rows := len(matrix)
	cols := len(matrix[0])
	fmt.Printf("KNN model shape: %d x %d\n", rows, cols)

	// Calculate and print the time it took to load the model
	elapsedTime := time.Since(startTime)
	fmt.Println("Total time loading model: ", elapsedTime.Milliseconds())

	// Return the KNN model
	return KNN{
		Data:    matrix,
		Classes: classes,
	}
}

func main() {
	// Load the KNN model from the specified CSV file
	model = LoadKNN("../weights/knn.csv")

	// Set up the HTTP server to handle requests
	http.HandleFunc("/api/knn", knnHandler)

	// Start the server and listen for requests on port 8080
	fmt.Println("Server is listening on port 8080...")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err) // Log error and terminate if the server fails to start
	}
}

// knnHandler handles incoming HTTP requests for KNN predictions.
// It expects POST requests containing encrypted data for prediction.
func knnHandler(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	// Print the start of request processing
	fmt.Println(strings.Repeat("-", 20) + "\nProcessing request...\n" + strings.Repeat("-", 20))

	// Check if the request method is POST, return error if not
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read the request body
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusInternalServerError)
		return
	}

	// Deserialize the request body into the PublicContext object
	context, err = DeserializeObject(body)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to deserialize struct: %v", err), http.StatusBadRequest)
		return
	}

	// Perform the encrypted KNN prediction using the model and context
	res, params := PredictEncrypted(&model, &context)

	// Prepare the response with distances, classes, and decryption parameters
	response := Response{
		Distances: res,           // Predicted distances for KNN
		Classes:   model.Classes, // Classes from the KNN model
		Params:    params,        // Parameters needed to decrypt the result
	}

	// Serialize the response object into bytes
	serializedResponse, err := SerializeObject(response)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to serialize response: %v", err), http.StatusInternalServerError)
		return
	}

	// Set the response content type and write the serialized response back to the client
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(serializedResponse)

	// Log the time taken to process the request
	elapsedTime := time.Since(startTime)
	fmt.Println("Total time processing request: ", elapsedTime.Milliseconds())
}
