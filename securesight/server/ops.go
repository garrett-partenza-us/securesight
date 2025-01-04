package main

import (
	"fmt"
	"time"
	"bytes"
	"encoding/gob"
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

	startTimeTotal := time.Now()
	startTime := time.Now()
	evaluator := ckks.NewEvaluator(c.Params, &c.Evk)
	fmt.Println("Total time loading evaluator: ", time.Since(startTime).Milliseconds())

	var subTime int64
	var mulTime int64
	var rotTime int64
	var addTime int64

	var totalSubs int
	var totalMuls int
	var totalRots int
	var totalAdds int

	// Initialize a slice to store the distances
	var results [][]rlwe.Ciphertext

	for _, ciphertext := range c.Query {

		var result []rlwe.Ciphertext

		// Loop through each target in the KNN model to calculate distances
		for _, target := range k.Data {

			// Compute the difference between the query and the target
			startTime = time.Now()
			sum, err := evaluator.SubNew(&ciphertext, target)
			if err != nil {
				panic(err)
			}
			subTime+=time.Since(startTime).Milliseconds()
			totalSubs++

			// Square the difference (multiply it with itself)
			startTime = time.Now()
			prod, err := evaluator.MulRelinNew(sum, sum)
			if err != nil {
				panic(err)
			}
			mulTime+=time.Since(startTime).Milliseconds()
			totalMuls++

			// Create a copy of the product to use for rotations
			res := prod.CopyNew()
			// Perform rotations and additions to compute distances
			// Starting Rotation Average: 16ms
			for rot := 1; rot <= 4; rot++ {
				startTime = time.Now()
				evaluator = evaluator.WithKey(&c.GaloisKeys[rot-1])
				// Rotate the ciphertext
				rotatedCiphertext, err := evaluator.RotateNew(res, 1*rot)
				if err != nil {
					panic(err)
				}
				rotTime+=time.Since(startTime).Milliseconds()
				totalRots++
				// Add the rotated ciphertext back to the result
				startTime = time.Now()
				res, err = evaluator.AddNew(res, rotatedCiphertext)
				if err != nil {
					panic(err)
				}
				addTime+=time.Since(startTime).Milliseconds()
				totalAdds++
			}
			result = append(result, *res)
		}
		results = append(results, result)
	}
	fmt.Printf("Total subtractions: %d, Total time: %d, Average time: %.2f\n", totalSubs, subTime, float64(subTime)/float64(totalSubs))
	fmt.Printf("Total multiplications: %d, Total time: %d, Average time: %.2f\n", totalMuls, mulTime, float64(mulTime)/float64(totalMuls))
	fmt.Printf("Total rotations: %d, Total time: %d, Average time: %.2f\n", totalRots, rotTime, float64(rotTime)/float64(totalRots))
	fmt.Printf("Total additions: %d, Total time: %d, Average time: %.2f\n", totalAdds, addTime, float64(addTime)/float64(totalAdds))

	elapsedTime := time.Since(startTime)
	fmt.Println("Time to predict: ", elapsedTime.Milliseconds())
	elapsedTime = time.Since(startTimeTotal)
	fmt.Println("Total time processing request: ", elapsedTime)


	// Return the distances to each target
	return results
}

func SerializeObject(obj interface{}) ([]byte, error) {
	startTime := time.Now()
	var buffer bytes.Buffer
	encoder := gob.NewEncoder(&buffer)
	err := encoder.Encode(obj)
	if err != nil {
		return nil, fmt.Errorf("Failed to serialize object: %v", err)
	}
	elapsedTime := time.Since(startTime)
	fmt.Println("Time to serialize ciphertexts: ", elapsedTime.Milliseconds())	
	return buffer.Bytes(), nil
}


// Function to deserialize the object
func DeserializeObject(data []byte) (PublicContext, error) {
	startTime := time.Now()
	var obj PublicContext
	decoder := gob.NewDecoder(bytes.NewReader(data))
	err := decoder.Decode(&obj)
	if err != nil {
		panic(err)
	}
	elapsedTime := time.Since(startTime)
	fmt.Println("Time to deserialize ciphertexts: ", elapsedTime.Milliseconds())
	return obj, nil
}
