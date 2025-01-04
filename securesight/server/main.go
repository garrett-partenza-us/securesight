package main

import (
	"time"
	"strings"
	"io/ioutil"
	"encoding/csv"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	//"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

var model KNN
var context PublicContext

type Response struct {
	Distances [][]Distance `json:"Distances"`
	Classes   []string    `json:"Classes"`
}

type KNN struct {
	Data    [][]float64
	Classes []string
}

func LoadKNN(path string) KNN {

	startTime := time.Now()

	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	var matrix [][]float64
	var classes []string

	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		var row []float64

		for i := 0; i < len(record)-1; i++ {
			f, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				panic(err)
			}
			row = append(row, f)
		}
		matrix = append(matrix, row)
		class := record[len(record)-1]
		classes = append(classes, class)
	}
	elapsedTime := time.Since(startTime)
	fmt.Println("Total time loading model: ", elapsedTime.Milliseconds())

	return KNN{
		Data:    matrix,
		Classes: classes,
	}
}

func main() {

	model = LoadKNN("/Users/garrett.partenza/projects/securesight/securesight/shared/weights/knn.csv")

	http.HandleFunc("/api/knn", knnHandler)

	fmt.Println("Server is listening on port 8080...")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}

}

func knnHandler(w http.ResponseWriter, r *http.Request) {

	startTime := time.Now()

	fmt.Println(strings.Repeat("-", 20)+"\nProcessing request...\n"+strings.Repeat("-", 20))

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusInternalServerError)
		return
	}

	// Deserialize the data into the struct
	context, err = DeserializeObject(body)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to deserialize struct: %v", err), http.StatusBadRequest)
		return
	}


	res := PredictEncrypted(&model, &context)


	response := Response{
		Distances: res,
		Classes:   model.Classes,
	}

	// Serialize the response struct
	serializedResponse, err := SerializeObject(response)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to serialize response: %v", err), http.StatusInternalServerError)
		return
	}

	// Set the response content type and send the serialized data back
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(serializedResponse)

	elapsedTime := time.Since(startTime)
	fmt.Println("Total time processing request: ", elapsedTime.Milliseconds())
}
