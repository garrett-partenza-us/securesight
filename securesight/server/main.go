package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
)

type Response struct {
	Classes [4]int `json:"message"`
}

func main() {

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

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	var data map[string]interface{}
	err = json.Unmarshal(body, &data)
	if err != nil {
		http.Error(w, "Error unmarshalling json", http.StatusInternalServerError)
		return
	}

	exampleClasses := [4]int{10, 20, 30, 40}

	response := Response{
		Classes: exampleClasses,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	err = json.NewEncoder(w).Encode(response)

	if err != nil {
		http.Error(w, "Failed to encoder JSON", http.StatusInternalServerError)
	}

}
