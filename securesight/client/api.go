package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"sort"
)

type ResponseData struct {
	Distances [][]float32 `json:"Distances"`
	Classes   []string    `json:"Classes"`
}

func CallAPI(embeddings [][]float32) ([]string, error) {

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
