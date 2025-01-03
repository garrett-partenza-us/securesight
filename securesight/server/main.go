package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
)

var model KNN
var context Context

type Response struct {
	Distances [][]float64 `json:"Distances"`
	Classes   []string    `json:"Classes"`
}

type KNN struct {
	Data    [][]float64
	Classes []string
}

func LoadKNN(path string) KNN {

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

	return KNN{
		Data:    matrix,
		Classes: classes,
	}
}

func main() {

	context = Setup()

	model = LoadKNN("/Users/garrett.partenza/projects/securesight/securesight/shared/weights/knn.csv")

	http.HandleFunc("/api/knn", knnHandler)

	fmt.Println("Server is listening on port 8080...")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}

}

func knnHandler(w http.ResponseWriter, r *http.Request) {

	fmt.Println("Handling request...")

	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var queries [][]float64
	if err := json.NewDecoder(r.Body).Decode(&queries); err != nil {
		http.Error(w, "Internal sever error", http.StatusInternalServerError)
		return
	}

	var distances [][]float64
	for _, query := range queries {
		ssd := PredictEncrypted(&model, &context, query)
		distances = append(distances, ssd)
	}


	response := Response{
		Distances: distances,
		Classes:   model.Classes,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	err := json.NewEncoder(w).Encode(response)

	if err != nil {
		http.Error(w, "Failed to encoder JSON", http.StatusInternalServerError)
	}

}
