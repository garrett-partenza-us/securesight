package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
)

// PCA model representation
type PCA struct {
	Components [][]float64 `json:"components"`
	Mean       []float64   `json:"mean"`
}

func NewPCA(path string) PCA {
	var pca PCA
	// Load the PCA components from the JSON file
	data, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatalf("Error reading PCA components file: %v", err)
	}

	err = json.Unmarshal(data, &pca)
	if err != nil {
		log.Fatalf("Error unmarshalling PCA components: %v", err)
	}
	return pca
}

// Function to apply PCA transformation
func (p *PCA) Transform(data [][]float64) [][]float64 {
	var transformedData [][]float64

	for i:=0; i < len(data); i++ {
		row := data[i]
		// Subtract the mean from each feature in the row
		subtractedRow := make([]float64, len(row))
		for i := range row {
			subtractedRow[i] = row[i] - p.Mean[i]
		}

		// Apply the transformation (multiply by the principal components)
		transformedRow := make([]float64, len(p.Components))
		for i := range p.Components {
			for j := range subtractedRow {
				transformedRow[i] += subtractedRow[j] * p.Components[i][j]
			}
		}
		transformedData = append(transformedData, transformedRow)
	}

	return transformedData
}
