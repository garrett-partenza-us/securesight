package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
)

// PCA model representation, holding components and the mean for transformation.
type PCA struct {
	Components [][]float64 `json:"components"` // Principal components matrix for transformation
	Mean       []float64   `json:"mean"`       // Mean vector for centering the data
}

// NewPCA loads the PCA components and mean from a JSON file and returns the PCA struct.
// The path is the location of the JSON file containing the PCA data.
func NewPCA(path string) PCA {
	var pca PCA

	// Read the contents of the PCA JSON file
	data, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatalf("Error reading PCA components file: %v", err) // Log and stop execution if file read fails
	}

	// Unmarshal the JSON data into the PCA struct
	err = json.Unmarshal(data, &pca)
	if err != nil {
		log.Fatalf("Error unmarshalling PCA components: %v", err) // Log and stop execution if unmarshalling fails
	}
	return pca // Return the loaded PCA object
}

// Transform applies the PCA transformation to the input data.
// The input data is expected to be a 2D slice (rows of data points) where each row represents a data point.
// The transformation applies the principal components to the centered data.
func (p *PCA) Transform(data [][]float64) [][]float64 {
	var transformedData [][]float64

	// Iterate over each data point (row)
	for i := 0; i < len(data); i++ {
		row := data[i]

		// Subtract the mean from each feature in the row (centering the data)
		subtractedRow := make([]float64, len(row))
		for i := range row {
			subtractedRow[i] = row[i] - p.Mean[i] // Center the data by subtracting the mean
		}

		// Apply the transformation by multiplying the centered data with the principal components
		transformedRow := make([]float64, len(p.Components))
		for i := range p.Components {
			// Multiply each feature of the centered data by the corresponding principal component
			for j := range subtractedRow {
				transformedRow[i] += subtractedRow[j] * p.Components[i][j]
			}
		}

		// Append the transformed row to the result
		transformedData = append(transformedData, transformedRow)
	}

	// Return the transformed data after applying PCA
	return transformedData
}
