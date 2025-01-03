package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Context holds the cryptographic parameters, key management, encryption, decryption,
// and evaluation structures needed to perform FHE operations.
type PublicContext struct {
	Params     ckks.Parameters           // CKKS parameters
	Rlk        rlwe.RelinearizationKey   // Relinearization key for homomorphic multiplication
	Evk        rlwe.MemEvaluationKeySet  // Memory-based evaluation keys for homomorphic operations
	GaloisKeys []rlwe.MemEvaluationKeySet// Decryptor for decrypting ciphertexts
	Query []rlwe.Ciphertext
}

// PredictEncrypted performs a prediction using encrypted data based on the provided
// KNN model and the input query. The query is encrypted, evaluated using homomorphic
// operations, and the distances are decrypted and returned as plaintext.
// 
// Parameters:
//   - k: A pointer to a KNN structure representing the K-Nearest Neighbors model.
//   - c: A pointer to a Context structure that holds the cryptographic parameters and keys.
//   - query: A slice of floats representing the input query data.
// 
// Returns:
//   - distances: A slice of floats representing the decrypted distances to each target in KNN.
func PredictEncrypted(k *KNN, c *PublicContext) [][]rlwe.Ciphertext {

	evaluator := ckks.NewEvaluator(c.Params, &c.Evk)

	// Initialize a slice to store the distances
	var results [][]rlwe.Ciphertext

	for _, ciphertext := range c.Query {

		var result []rlwe.Ciphertext

		// Loop through each target in the KNN model to calculate distances
		for _, target := range k.Data {

			// Compute the difference between the query and the target
			sum, err := evaluator.SubNew(&ciphertext, target)
			if err != nil {
				panic(err)
			}

			// Square the difference (multiply it with itself)
			prod, err := evaluator.MulRelinNew(sum, sum)
			if err != nil {
				panic(err)
			}

			// Create a copy of the product to use for rotations
			res := prod.CopyNew()

			// Perform rotations and additions to compute distances
			for rot := 1; rot <= 4; rot++ {
				evaluator = evaluator.WithKey(&c.GaloisKeys[rot-1])
				// Rotate the ciphertext
				rotatedCiphertext, err := evaluator.RotateNew(res, 1*rot)
				if err != nil {
					panic(err)
				}
				// Add the rotated ciphertext back to the result
				res, err = evaluator.AddNew(res, rotatedCiphertext)
				if err != nil {
					panic(err)
				}
			}
			result = append(result, *res)
		}
		results = append(results, result)
	}

	// Return the distances to each target
	return results
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
func DeserializeObject(data []byte) (PublicContext, error) {
	var obj PublicContext
	decoder := gob.NewDecoder(bytes.NewReader(data))
	err := decoder.Decode(&obj)
	if err != nil {
		panic(err)
	}
	return obj, nil
}
