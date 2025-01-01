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

type Context struct {
	Params ckks.Parameters
	Encoder ckks.Encoder
	Kgen rlwe.KeyGenerator
	Sk rlwe.SecretKey
	Encryptor rlwe.Encryptor
	Rlk rlwe.RelinearizationKey
	Evk rlwe.MemEvaluationKeySet
	Evaluator *ckks.Evaluator
	Decryptor rlwe.Decryptor
}

func Setup() Context {

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
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evaluator := ckks.NewEvaluator(params, evk)
	decryptor := rlwe.NewDecryptor(params, sk)

	return Context{
		Params: params,
		Encoder: *encoder,
		Kgen: *kgen,
		Sk: *sk,
		Encryptor: *encryptor,
		Rlk: *rlk,
		Evk: *evk,
		Evaluator: evaluator,
		Decryptor: *decryptor,
	}

}

func PredictEncrypted(k *KNN, c *Context, query []float64) []float64 {

	var distances []float64

	plaintextQuery := ckks.NewPlaintext(c.Params, c.Params.MaxLevel())
	if err := c.Encoder.Encode(query, plaintextQuery); err != nil{
		panic(err)
	}
	
	ciphertextQuery, err := c.Encryptor.EncryptNew(plaintextQuery)
	if err != nil {
		panic(err)
	}

	for _, target := range k.Data {

		sum, err := c.Evaluator.SubNew(ciphertextQuery, target)
		if err != nil {
			panic(err)
		}
		prod, err := c.Evaluator.MulRelinNew(sum, sum)
		if err != nil {
			panic(err)
		}

		//////////////
		// ROTATION //
		//////////////
		res := prod.CopyNew()
		for rot := 1; rot < len(query)/2 + 1; rot++ {
			galEls := []uint64{
				c.Params.GaloisElement(1*rot),
				c.Params.GaloisElementForComplexConjugation(),
			}
			c.Evaluator = c.Evaluator.WithKey(rlwe.NewMemEvaluationKeySet(&c.Rlk, c.Kgen.GenGaloisKeysNew(galEls, &c.Sk)...))
			rotatedCiphertext, err := c.Evaluator.RotateNew(res, 1*rot)
			if err != nil{
				panic(err)
			}
			res, err = c.Evaluator.AddNew(res, rotatedCiphertext)
			if err != nil{
				panic(err)
			}
		}

		decryptedPlaintext := c.Decryptor.DecryptNew(res)

		// Decodes the plaintext
		have := make([]float64, c.Params.MaxSlots())
		if err = c.Encoder.Decode(decryptedPlaintext, have); err != nil {
			panic(err)
		}

		distances = append(distances, have[0])

	}

	return distances

}
