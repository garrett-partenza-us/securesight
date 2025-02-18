package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"sort"
	"sync"
)

// Server side CKKS context
type PublicContext struct {
	Params ckks.Parameters          // CKKS parameters
	Rlk    rlwe.RelinearizationKey  // Relinearization key for homomorphic multiplication
	Evk    rlwe.MemEvaluationKeySet // Memory-based evaluation keys for homomorphic operations
	Query  []rlwe.Ciphertext        // List of encrypted query vectors
}

// Distance of KNN datapoint
type Distance struct {
	Distance rlwe.Ciphertext // Distance from a given target example
	Classes  []string        // Class of the given target example
}

type QueryResult struct {
	Distances []Distance
	QueryNum  int
}

type PackedTarget struct {
	Vec     []float64
	Classes []string
}

// PredictEncrypted calculates the Euclidean distance of encrypted queries in CKKS FHE for a KNN model
// It performs the calculation concurrently for multiple queries and multiple KNN data points.
func PredictEncrypted(knnModel *KNN, context *PublicContext) ([][]Distance, ckks.Parameters) {
	// Initialize evaluator from server-side context for FHE operations
	evaluator := ckks.NewEvaluator(context.Params, &context.Evk)

	// Channel for collecting results from goroutines
	resultChannel := make(chan QueryResult, len(context.Query))

	maxRepeat := int(context.Params.MaxSlots()) / 512
	batches := batchTargets(knnModel.Data, maxRepeat)
	packs := packTargets(batches, knnModel.Classes)

	// WaitGroup to manage concurrent execution of query processing
	var wg sync.WaitGroup

	// Process each encrypted query concurrently
	for queryIdx, ciphertext := range context.Query {
		wg.Add(1)
		go processQuery(ciphertext, queryIdx, packs, *evaluator.ShallowCopy(), resultChannel, &wg)
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
func processQuery(ciphertext rlwe.Ciphertext, queryIdx int, packs []PackedTarget, evaluator ckks.Evaluator, resultChannel chan<- QueryResult, wg *sync.WaitGroup) {
	defer wg.Done()

	// Channel for collecting distances of the current query from each target in the KNN model
	innerResultChannel := make(chan Distance, len(packs))
	var innerWg sync.WaitGroup

	// Process each target in the KNN model concurrently
	for _, pack := range packs {
		innerWg.Add(1)
		go processTarget(ciphertext, pack, *evaluator.ShallowCopy(), innerResultChannel, &innerWg)
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
func processTarget(ciphertext rlwe.Ciphertext, pack PackedTarget, evaluator ckks.Evaluator, resultChannel chan<- Distance, wg *sync.WaitGroup) {

	defer wg.Done()

	// Compute the difference between the query and the target
	diff, err := evaluator.SubNew(&ciphertext, pack.Vec)
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
		Classes:  pack.Classes,
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

func batchTargets(targets [][]float64, n int) [][][]float64 {
	var batches [][][]float64
	for i := 0; i < len(targets); i += n {
		end := i + n
		if end > len(targets) {
			end = len(targets)
		}

		batch := targets[i:end]
		batches = append(batches, batch)
	}
	return batches
}

func packTargets(batches [][][]float64, classes []string) []PackedTarget {
	var packs []PackedTarget
	targetIdx := 0
	for _, batch := range batches {
		var concatenated []float64
		var packClasses []string
		for _, vec := range batch {
			concatenated = append(concatenated, vec...)
			packClasses = append(packClasses, classes[targetIdx])
			targetIdx++
		}
		packedTarget := PackedTarget{
			Vec:     concatenated,
			Classes: packClasses,
		}
		packs = append(packs, packedTarget)
	}
	return packs
}
