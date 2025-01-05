package main

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Context holds the cryptographic parameters, key management, encryption, decryption,
// and evaluation structures needed to perform FHE operations.
type Context struct {
	Params     ckks.Parameters           // CKKS parameters
	Encoder    ckks.Encoder              // Encoder for encoding and decoding plaintexts
	Kgen       rlwe.KeyGenerator         // Key generator for secret and public key generation
	Sk         rlwe.SecretKey            // Secret key used for encryption and decryption
	Encryptor  rlwe.Encryptor            // Encryptor used for encryption operations
	Rlk        rlwe.RelinearizationKey   // Relinearization key for homomorphic multiplication
	Evk        rlwe.MemEvaluationKeySet  // Memory-based evaluation keys for homomorphic operations
	Evaluator  *ckks.Evaluator           // Evaluator used for homomorphic operations on ciphertexts
	Decryptor  rlwe.Decryptor            // Decryptor for decrypting ciphertexts
}

// Setup initializes the cryptographic parameters and returns a Context object with
// the necessary keys, encoder, encryptor, and evaluator to perform FHE operations.
// 
// Returns:
//   - Context: A structure containing all necessary cryptographic objects and settings.
func Setup() Context {
	// Initialize CKKS parameters
	var params ckks.Parameters
	params, err := ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            14,                                // log2(ring degree)
			LogQ: []int{60, 50, 50, 50, 50, 50, 50, 50},        // Moduli sizes for CKKS
			LogP:            []int{61},                         // Log2 of auxiliary modulus P
			LogDefaultScale: 45,                                // Default scale factor
		})
	if err != nil {
		panic(err)
	}

	// Initialize cryptographic components
	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	encryptor := rlwe.NewEncryptor(params, sk)
	// Secret key: key.Value.P.Coeffs and key.Value.Q.Coeffs
	// Public key: key.Value[:2].P.Coeffs and key.Value[:2].Q.Coeffs
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evaluator := ckks.NewEvaluator(params, evk)
	decryptor := rlwe.NewDecryptor(params, sk)

	// Return the fully populated Context
	return Context{
		Params:     params,
		Encoder:    *encoder,
		Kgen:       *kgen,
		Sk:         *sk,
		Encryptor:  *encryptor,
		Rlk:        *rlk,
		Evk:        *evk,
		Evaluator:  evaluator,
		Decryptor:  *decryptor,
	}
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
func PredictEncrypted(k *KNN, c *Context, query []float64) []float64 {
	// Initialize a slice to store the distances
	var distances []float64

	// Create a plaintext for the query
	plaintextQuery := ckks.NewPlaintext(c.Params, c.Params.MaxLevel())
	if err := c.Encoder.Encode(query, plaintextQuery); err != nil {
		panic(err)
	}

	// Encrypt the query
	ciphertextQuery, err := c.Encryptor.EncryptNew(plaintextQuery)
	if err != nil {
		panic(err)
	}

	// Loop through each target in the KNN model to calculate distances
	for _, target := range k.Data {

		// Compute the difference between the query and the target
		sum, err := c.Evaluator.SubNew(ciphertextQuery, target)
		if err != nil {
			panic(err)
		}

		// Square the difference (multiply it with itself)
		prod, err := c.Evaluator.MulRelinNew(sum, sum)
		if err != nil {
			panic(err)
		}

		// Create a copy of the product to use for rotations
		res := prod.CopyNew()

		// Perform rotations and additions to compute distances
		for rot := 1; rot < len(query)/2+1; rot++ {
			galEls := []uint64{
				c.Params.GaloisElement(1 * rot),
				c.Params.GaloisElementForComplexConjugation(),
			}
			// Update the evaluator with the new Galois keys
			c.Evaluator = c.Evaluator.WithKey(rlwe.NewMemEvaluationKeySet(&c.Rlk, c.Kgen.GenGaloisKeysNew(galEls, &c.Sk)...))
			// Rotate the ciphertext
			rotatedCiphertext, err := c.Evaluator.RotateNew(res, 1*rot)
			if err != nil {
				panic(err)
			}
			// Add the rotated ciphertext back to the result
			res, err = c.Evaluator.AddNew(res, rotatedCiphertext)
			if err != nil {
				panic(err)
			}
		}

		// Decrypt the result back into plaintext
		decryptedPlaintext := c.Decryptor.DecryptNew(res)

		// Decode the decrypted result into a float64 slice
		have := make([]float64, c.Params.MaxSlots())
		if err = c.Encoder.Decode(decryptedPlaintext, have); err != nil {
			panic(err)
		}

		// Append the computed distance to the result list
		distances = append(distances, have[0])
	}

	// Return the distances to each target
	return distances
}
