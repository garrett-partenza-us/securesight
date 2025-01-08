package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"io/ioutil"
	"net/http"
	"time"
)

// ResponseData represents the structure of the response from the KNN API.
type ResponseData struct {
	Distances [][]Distance `json:"Distances"`
	Classes   []string     `json:"Classes"`
	Params    ckks.Parameters `json:"Params"`
}

// Distance of KNN datapoint
type Distance struct{
	Distance rlwe.Ciphertext							// Distance from a given target example
	Classes []string													// Class of the given target example
}

type QueryResult struct {
	Distances []Distance
	QueryNum  int
}

// CallAPI sends a POST request with the serialized data to the KNN API, 
// deserializes the response, and returns it as ResponseData.
func CallAPI(serializedData []byte) (ResponseData, error) {
	// API endpoint for KNN service
	url := "http://localhost:8080/api/knn"
	
	// Send POST request with serialized data as the payload
	resp, err := http.Post(url, "application/octet-stream", bytes.NewReader(serializedData))
	if err != nil {
		panic(err) // Handle request failure
	}
	defer resp.Body.Close() // Ensure response body is closed after reading

	// Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		panic(err) // Handle body read failure
	}

	// Deserialize the response body into a ResponseData struct
	responseData, err := DeserializeObject(body)
	if err != nil {
		panic(err) // Handle deserialization failure
	}

	return responseData, nil
}

// SerializeObject serializes an object into a byte slice using Gob encoding.
// It also logs the time taken to complete the serialization.
func SerializeObject(obj interface{}) ([]byte, error) {
	startTime := time.Now() // Track serialization time

	var buffer bytes.Buffer
	encoder := gob.NewEncoder(&buffer)
	err := encoder.Encode(obj)
	if err != nil {
		return nil, fmt.Errorf("Failed to serialize object: %v", err) // Return error if serialization fails
	}

	// Log time taken for serialization
	elapsedTime := time.Since(startTime)
	fmt.Println("Time to serialize ciphertexts: ", elapsedTime)
	
	return buffer.Bytes(), nil
}

// DeserializeObject deserializes a byte slice into a ResponseData object.
// It also logs the time taken to complete the deserialization.
func DeserializeObject(data []byte) (ResponseData, error) {
	startTime := time.Now() // Track deserialization time

	var obj ResponseData
	decoder := gob.NewDecoder(bytes.NewReader(data))
	err := decoder.Decode(&obj)
	if err != nil {
		panic(err) // Handle deserialization failure
	}

	// Log time taken for deserialization
	elapsedTime := time.Since(startTime)
	fmt.Println("Time to deserialize ciphertexts: ", elapsedTime)

	return obj, nil
}

