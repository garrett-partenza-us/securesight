package main

import (
	"fmt"
	"time"
	"sync"
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

type Distance struct{
	Distance rlwe.Ciphertext
	Class string
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
// Starting func time: 7.4s
// Outer routine: 3.4s
// Inner routine: 2.2
func PredictEncrypted(k *KNN, c *PublicContext) [][]Distance {
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
    var results [][]Distance

    // Create a channel to collect results from goroutines
    resultChannel := make(chan []Distance, len(c.Query))

    // Iterate through each ciphertext (query) concurrently
    var wg sync.WaitGroup
    for _, ciphertext := range c.Query {
        wg.Add(1)
        go func(ciphertext rlwe.Ciphertext) {
            defer wg.Done()

            var result []Distance
            resultChannelInner := make(chan Distance, len(k.Data)) // Channel for each target
            var wgInner sync.WaitGroup

            // Loop through each target in the KNN model to calculate distances
            for j, target := range k.Data {
                wgInner.Add(1)
                go func(target []float64, idx int) {
                    defer wgInner.Done()

                    // Compute the difference between the query and the target
                    startTime = time.Now()
                    sum, err := evaluator.SubNew(&ciphertext, target)
                    if err != nil {
                        panic(err)
                    }
                    subTime += time.Since(startTime).Milliseconds()
                    totalSubs++

                    // Square the difference (multiply it with itself)
                    startTime = time.Now()
                    prod, err := evaluator.MulRelinNew(sum, sum)
                    if err != nil {
                        panic(err)
                    }
                    mulTime += time.Since(startTime).Milliseconds()
                    totalMuls++

                    // Create a copy of the product to use 
                    res := prod.CopyNew()
										/*
										
										var wgInnerInner sync.WaitGroup
										resultChannelInnerInner := make(chan *rlwe.Ciphertext, 4) // Channel for each target
										for rot:= 1; rot <=4; rot++{
											wgInnerInner.Add(1)
											go func(rot int){
												defer wgInnerInner.Done()
                        startTime = time.Now()
												subevaluator := evaluator.WithKey(&c.GaloisKeys[rot-1])
                        // Rotate the ciphertext
                        rotatedCiphertext, err := subevaluator.RotateNew(res, 1*rot)
                        if err != nil {
                            panic(err)
                        }
                        rotTime += time.Since(startTime).Milliseconds()
                        totalRots++
												resultChannelInnerInner <- rotatedCiphertext
											}(rot)

										}

										wgInnerInner.Wait()

										// Close the inner channel after all target goroutines are done
										close(resultChannelInnerInner)

										// Collect the results from the inner channel for the current query
										for rotatedCiphertext := range resultChannelInnerInner {
                        // Add the rotated ciphertext back to the result
                        startTime = time.Now()
                        res, err = evaluator.AddNew(res, rotatedCiphertext)
                        if err != nil {
                            panic(err)
                        }
                        addTime += time.Since(startTime).Milliseconds()
                        totalAdds++
										}
										*/
										
										


                    // Send the result for the current target to the inner channel
										tuple := Distance{
											Distance: *res,
											Class: k.Classes[idx],
										}
                    resultChannelInner <- tuple
                }(target, j)
            }

            // Wait for all target goroutines to complete
            wgInner.Wait()

            // Close the inner channel after all target goroutines are done
            close(resultChannelInner)

            // Collect the results from the inner channel for the current query
            for distance := range resultChannelInner {
                result = append(result, distance)
            }

            // Send the result for the current ciphertext (query) into the result channel
            resultChannel <- result
        }(ciphertext)
    }

    // Wait for all query goroutines to complete
    wg.Wait()

    // Close the result channel after all queries are processed
    close(resultChannel)

    // Collect the results from the result channel
    for result := range resultChannel {
        results = append(results, result)
    }

    // Print timing stats
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
