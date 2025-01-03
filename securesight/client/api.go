package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"encoding/gob"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type ResponseData struct {
	Distances [][]rlwe.Ciphertext `json:"Distances"`
	Classes   []string    `json:"Classes"`
}

func CallAPI(serializedData []byte) (ResponseData, error) {

	url := "http://localhost:8080/api/knn"
	resp, err := http.Post(url, "application/octet-stream", bytes.NewReader(serializedData))
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		panic(err)
	}

	// Deserialize the data into the struct
	responseData, err := DeserializeObject(body)
	if err != nil {
		panic(err)
	}

	return responseData, nil

}

func SerializeObject(obj interface{}) ([]byte, error) {
	var buffer bytes.Buffer
	encoder := gob.NewEncoder(&buffer)
	err := encoder.Encode(obj)
	if err != nil {
		return nil, fmt.Errorf("Failed to serialize object: %v", err)
	}
	return buffer.Bytes(), nil
}

// Function to deserialize the object
func DeserializeObject(data []byte) (ResponseData, error) {
	var obj ResponseData
	decoder := gob.NewDecoder(bytes.NewReader(data))
	err := decoder.Decode(&obj)
	if err != nil {
		panic(err)
	}
	return obj, nil
}
