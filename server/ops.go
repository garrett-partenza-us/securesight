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
	Query			 []rlwe.Ciphertext							  // List of encrypted query vectors
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

// PredictEncrypted calculates the Euclidean distance of encrypted queries in CKKS FHE for a KNN model
// It performs the calculation concurrently for multiple queries and multiple KNN data points.
func PredictEncrypted(knnModel *KNN, context *PublicContext) ([][]Distance, ckks.Parameters) {
	// Initialize evaluator from server-side context for FHE operations
	evaluator := ckks.NewEvaluator(context.Params, &context.Evk)

	// Channel for collecting results from goroutines
	resultChannel := make(chan QueryResult, len(context.Query))

	// WaitGroup to manage concurrent execution of query processing
	var wg sync.WaitGroup

	// Process each encrypted query concurrently
	for queryIdx, ciphertext := range context.Query {
		wg.Add(1)
		go processQuery(ciphertext, queryIdx, knnModel, *evaluator.ShallowCopy(), resultChannel, &wg)
	}

	// Wait for all query processing goroutines to finish
	wg.Wait()

	// Close the result channel after all queries are processed
	close(resultChannel)

	// Collect the results and sort by query index
	return collectAndSortResults(resultChannel), context.Params
}

// processQuery calculates the Euclidean distance for a single query against all KNN data points.
// It runs in a separate goroutine for each query.
func processQuery(ciphertext rlwe.Ciphertext, queryIdx int, knnModel *KNN, evaluator ckks.Evaluator, resultChannel chan<- QueryResult, wg *sync.WaitGroup) {
	defer wg.Done()

	// Channel for collecting distances of the current query from each target in the KNN model
	innerResultChannel := make(chan Distance, len(knnModel.Data))
	var innerWg sync.WaitGroup

	// Process each target in the KNN model concurrently
	for targetIdx, target := range knnModel.Data {
		innerWg.Add(1)
		go processTarget(ciphertext, target, targetIdx, knnModel.Classes, *evaluator.ShallowCopy(), innerResultChannel, &innerWg)
	}

	// Wait for all target distance calculations to finish
	innerWg.Wait()

	// Close the inner result channel after all targets have been processed
	close(innerResultChannel)

	// Collect the distances for the current query
	var distances []Distance
	for dist := range innerResultChannel {
		distances = append(distances, dist)
	}

	// Send the result for the current query into the main result channel
	resultChannel <- QueryResult{
		QueryNum:  queryIdx,
		Distances: distances,
	}
}

// processTarget computes the squared Euclidean distance for a single target and a query.
// It is executed concurrently for each target in the KNN model.
func processTarget(ciphertext rlwe.Ciphertext, target []float64, targetIdx int, classes []string, evaluator ckks.Evaluator, resultChannel chan<- Distance, wg *sync.WaitGroup) {
	defer wg.Done()

	// Compute the difference between the query and the target
	diff, err := evaluator.SubNew(&ciphertext, target)
	if err != nil {
		panic(err)
	}

	// Square the difference (compute the squared Euclidean distance)
	squaredDiff, err := evaluator.MulRelinNew(diff, diff)
	if err != nil {
		panic(err)
	}


	// Rescale again after squaring to keep the ciphertext manageable
	if err := evaluator.Rescale(squaredDiff, squaredDiff); err != nil {
		panic(err)
	}

	// Send the result to the result channel (using squaredDiff as the final distance)
	resultChannel <- Distance{
		Distance: *squaredDiff, // You might want to send squaredDiff as the distance, not diff
		Class:    classes[targetIdx],
	}
}

// collectAndSortResults collects results from the result channel and sorts them by query index.
func collectAndSortResults(resultChannel <-chan QueryResult) [][]Distance {
	// Collect all results into a slice
	var unsortedResults []QueryResult
	for result := range resultChannel {
		unsortedResults = append(unsortedResults, result)
	}

	// Sort the results by query index (QueryNum)
	sort.Slice(unsortedResults, func(i, j int) bool {
		return unsortedResults[i].QueryNum < unsortedResults[j].QueryNum
	})

	// Extract and return the sorted distances for each query
	var sortedResults [][]Distance
	for _, result := range unsortedResults {
		sortedResults = append(sortedResults, result.Distances)
	}
	return sortedResults
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
