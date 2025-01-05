package main

import (
	"fmt"
	"sync"
	"bytes"
	"encoding/gob"
	"sort"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Server side CKKS context
type PublicContext struct {
	Params     ckks.Parameters            // CKKS parameters
	Rlk        rlwe.RelinearizationKey    // Relinearization key for homomorphic multiplication
	Evk        rlwe.MemEvaluationKeySet   // Memory-based evaluation keys for homomorphic operations
	GaloisKeys []rlwe.MemEvaluationKeySet // Decryptor for decrypting ciphertexts
	Query []rlwe.Ciphertext							  // List of encrypted query vectors
}

// Distance of KNN datapoint
type Distance struct{
	Distance rlwe.Ciphertext							// Distance from a given target example
	Class string													// Class of the given target example
}

type QueryResult struct {
	Distances []Distance
	QueryNum  int
}

func PredictEncrypted(k *KNN, c *PublicContext) [][]Distance {

		// Initalize evaluator from server side context
    evaluator := ckks.NewEvaluator(c.Params, &c.Evk)

    // Results for client of shape len(faces) x len(knn datapoints)
    var results [][]Distance

    // Create a channel to collect results from goroutines
		multipleQueryDistances := make(chan struct {
			Distances []Distance
			QueryNum int
		}, len(c.Query))

    // Iterate through each ciphertext (query) concurrently
    var wg sync.WaitGroup
    for qidx, ciphertext := range c.Query {
        wg.Add(1)
        go func(ciphertext rlwe.Ciphertext, qidx int) {
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
                    sum, err := evaluator.SubNew(&ciphertext, target)
                    if err != nil {
                        panic(err)
                    }

                    // Square the difference (multiply it with itself)
                    prod, err := evaluator.MulRelinNew(sum, sum)
                    if err != nil {
                        panic(err)
                    }

                    // Create a copy of the product to use 
                    res := prod.CopyNew()
										
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
						multipleQueryDistances <- struct {
								Distances []Distance
								QueryNum  int
							}{
								Distances: result,  // Assign result to the Distances field
								QueryNum:  qidx,     // Assign qidx to the QueryNum field
						}
        }(ciphertext, qidx)
    }

    // Wait for all query goroutines to complete
    wg.Wait()

    // Close the result channel after all queries are processed
    close(multipleQueryDistances)

		// Step 1: Receive all the struct data from the channel into a slice
		var unsortedResults []QueryResult
		for result := range multipleQueryDistances {
			unsortedResults = append(unsortedResults, result)
		}

		// Step 2: Sort the slice by QueryNum using sort.Slice
		sort.Slice(unsortedResults, func(i, j int) bool {
			return unsortedResults[i].QueryNum < unsortedResults[j].QueryNum
		})

		// Step 3: Print the sorted results
		for _, result := range unsortedResults {
			results = append(results, result.Distances)
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
