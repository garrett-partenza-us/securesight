package main

import (
	"fmt"
	"time"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Context holds the cryptographic parameters, key management, encryption, decryption,
// and evaluation structures needed to perform FHE operations.
type Context struct {
	Params    ckks.Parameters          // CKKS parameters
	Encoder   ckks.Encoder             // Encoder for encoding and decoding plaintexts
	Kgen      rlwe.KeyGenerator        // Key generator for secret and public key generation
	Sk        rlwe.SecretKey           // Secret key used for encryption and decryption
	Encryptor rlwe.Encryptor           // Encryptor used for encryption operations
	Rlk       rlwe.RelinearizationKey  // Relinearization key for homomorphic multiplication
	Evk       rlwe.MemEvaluationKeySet // Memory-based evaluation keys for homomorphic operations
	Evaluator *ckks.Evaluator          // Evaluator used for homomorphic operations on ciphertexts
	Decryptor rlwe.Decryptor           // Decryptor for decrypting ciphertexts
}

// Context holds the cryptographic parameters, key management, encryption, decryption,
// and evaluation structures needed to perform FHE operations.
type PublicContext struct {
	Params     ckks.Parameters            // CKKS parameters
	Rlk        rlwe.RelinearizationKey    // Relinearization key for homomorphic multiplication
	Evk        rlwe.MemEvaluationKeySet   // Memory-based evaluation keys for homomorphic operations
	GaloisKeys []rlwe.MemEvaluationKeySet // Decryptor for decrypting ciphertexts
	Query      []rlwe.Ciphertext
}

func NewEncryptor() Context {
	startTime := time.Now()
	// Initialize CKKS parameters
	var params ckks.Parameters
	params, err := ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            14,                                    // log2(ring degree)
			LogQ:            []int{60, 50, 50, 50, 50, 50, 50, 50}, // Moduli sizes for CKKS
			LogP:            []int{61},                             // Log2 of auxiliary modulus P
			LogDefaultScale: 45,                                    // Default scale factor
		})
	if err != nil {
		panic(err)
	}

	// Initialize cryptographic components
	encoder := ckks.NewEncoder(params)
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	encryptor := rlwe.NewEncryptor(params, sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	evaluator := ckks.NewEvaluator(params, evk)
	decryptor := rlwe.NewDecryptor(params, sk)

	elapsedTime := time.Since(startTime)
	fmt.Println("Time to create local CKKS context: ", elapsedTime.Milliseconds())

	// Return the fully populated Context
	return Context{
		Params:    params,
		Encoder:   *encoder,
		Kgen:      *kgen,
		Sk:        *sk,
		Encryptor: *encryptor,
		Rlk:       *rlk,
		Evk:       *evk,
		Evaluator: evaluator,
		Decryptor: *decryptor,
	}
}

func (c *Context) Encrypt(vec []float64) rlwe.Ciphertext {

	startTime := time.Now()

	plaintext := ckks.NewPlaintext(c.Params, c.Params.MaxLevel())
	if err := c.Encoder.Encode(vec, plaintext); err != nil {
		panic(err)
	}

	ciphertext, err := c.Encryptor.EncryptNew(plaintext)
	if err != nil {
		panic(err)
	}

	elapsedTime := time.Since(startTime)
	fmt.Println("Time to encrypt: ", elapsedTime.Milliseconds())

	return *ciphertext
}

func (c *Context) NewPublicContext(query []rlwe.Ciphertext) PublicContext {

	startTime := time.Now()

	var keys []rlwe.MemEvaluationKeySet
	for rot := 1; rot <= 4; rot++ {
		galEls := []uint64{
			c.Params.GaloisElement(1 * rot),
			c.Params.GaloisElementForComplexConjugation(),
		}
		galoisKey := rlwe.NewMemEvaluationKeySet(&c.Rlk, c.Kgen.GenGaloisKeysNew(galEls, &c.Sk)...)
		keys = append(keys, *galoisKey)
	}
	elapsedTime := time.Since(startTime)
	fmt.Println("Time to create public CKKS context: ", elapsedTime.Milliseconds())
	return PublicContext{
		Params:     c.Params,
		Rlk:        c.Rlk,
		Evk:        c.Evk,
		GaloisKeys: keys,
		Query:      query,
	}
}

func (c *Context) Decrypt(res [][]Distance) ([][]float64, [][]string) {

	startTime := time.Now()

	var results [][]float64
	var resultsClasses [][]string

	for _, face := range res {
		var distances []float64
		var classes []string
		for _, target := range face {

			// Decrypt the result back into plaintext
			decryptedPlaintext := c.Decryptor.DecryptNew(&target.Distance)

			// Decode the decrypted result into a float64 slice
			have := make([]float64, c.Params.MaxSlots())
			if err := c.Encoder.Decode(decryptedPlaintext, have); err != nil {
				panic(err)
			}

			distances = append(distances, sum(have[:4]))
			classes = append(classes, target.Class)

		}
		results = append(results, distances)
		resultsClasses = append(resultsClasses, classes)
	}

	elapsedTime := time.Since(startTime)
	fmt.Println("Time to decrypt: ", elapsedTime.Milliseconds())
	return results, resultsClasses

}

func sum(arr []float64) float64 {
	var total float64
	for _, num := range arr {
		total+=num
	}
	return total
}
