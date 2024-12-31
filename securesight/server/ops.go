package main

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Function to compute and print the sum of squared differences (SSD) between the query vector and each target vector
func computeSSDs(query []float64, targets [][]float64) {
	// Iterate over each target vector
	for i, target := range targets {
		// Compute the sum of squared differences for this target vector
		ssd := 0.0
		for j := 0; j < len(query); j++ {
			// Calculate the squared difference and add to the sum
			diff := query[j] - target[j]
			ssd += diff * diff
		}
		// Print the result for the current target
		fmt.Printf("SSD for target vector %d: %f\n", i, ssd)
	}
}

func setup() {

	var params ckks.Parameters

	params, err := ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            14,                                    // log2(ring degree)
			LogQ: []int{60, 50, 50, 50, 50, 50, 50, 50}, // increase these values
			LogP:            []int{61},                             // log2(primes P) (auxiliary modulus)
			LogDefaultScale: 45,                                    // log2(scale)
		});
	if err != nil {
		panic(err)
	}

	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	encryptor := rlwe.NewEncryptor(params, sk)

	query := []float64{1.0, 2.0, 3.0, 4.0}

	plaintextQuery := ckks.NewPlaintext(params, params.MaxLevel())
	if err = encoder.Encode(query, plaintextQuery); err != nil{
		panic(err)
	}

	var ciphertextQuery *rlwe.Ciphertext
	if ciphertextQuery, err = encryptor.EncryptNew(plaintextQuery); err != nil {
		panic(err)
	}

	targets := [][]float64{
		{1.1, 2.2, 3.3, 4.3},
		{2.0, 3.0, 4.0, 5.0},
		{4.0, 1.0, 2.0, 1.0},
	}

	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evaluator := ckks.NewEvaluator(params, evk)
	decryptor := rlwe.NewDecryptor(params, sk)

	for _, target := range targets {

		sum, err := evaluator.SubNew(ciphertextQuery, target)
		if err != nil {
			panic(err)
		}
		prod, err := evaluator.MulRelinNew(sum, sum)
		if err != nil {
			panic(err)
		}

		//////////////
		// ROTATION //
		//////////////
		res := prod.CopyNew()
		for rot := 1; rot < len(query)/2 + 1; rot++ {
			galEls := []uint64{
				params.GaloisElement(1*rot),
				params.GaloisElementForComplexConjugation(),
			}
			evaluator = evaluator.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))
			rotatedCiphertext, err := evaluator.RotateNew(res, 1*rot)
			if err != nil{
				panic(err)
			}
			res, err = evaluator.AddNew(res, rotatedCiphertext)
			if err != nil{
				panic(err)
			}
		}

		decryptedPlaintext := decryptor.DecryptNew(res)

		// Decodes the plaintext
		have := make([]float64, params.MaxSlots())
		if err = encoder.Decode(decryptedPlaintext, have); err != nil {
			panic(err)
		}

		fmt.Println(have[:4])

	}

	computeSSDs(query, targets)

}
