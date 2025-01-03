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

// Context holds the cryptographic parameters, key management, encryption, decryption,
// and evaluation structures needed to perform FHE operations.
type PublicContext struct {
	Params     ckks.Parameters           // CKKS parameters
	Rlk        rlwe.RelinearizationKey   // Relinearization key for homomorphic multiplication
	Evk        rlwe.MemEvaluationKeySet  // Memory-based evaluation keys for homomorphic operations
	GaloisKeys []rlwe.MemEvaluationKeySet// Decryptor for decrypting ciphertexts
	Query []rlwe.Ciphertext
}


func NewEncryptor() Context {
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

func (c *Context) Encrypt(vec []float64) rlwe.Ciphertext {

	plaintext := ckks.NewPlaintext(c.Params, c.Params.MaxLevel())
	if err := c.Encoder.Encode(vec, plaintext); err != nil {
		panic(err)
	}

	ciphertext, err := c.Encryptor.EncryptNew(plaintext)
	if err != nil {
		panic(err)
	}

	return *ciphertext
}

func (c *Context) NewPublicContext(query []rlwe.Ciphertext) PublicContext {

	var keys []rlwe.MemEvaluationKeySet
	for rot := 1; rot <= 4; rot++ {
		galEls := []uint64{
			c.Params.GaloisElement(1 * rot),
			c.Params.GaloisElementForComplexConjugation(),
		}
		galoisKey := rlwe.NewMemEvaluationKeySet(&c.Rlk, c.Kgen.GenGaloisKeysNew(galEls, &c.Sk)...)
		keys = append(keys, *galoisKey)
	}
	return PublicContext{
		Params:     c.Params,
		Rlk:        c.Rlk,
		Evk:        c.Evk,
		GaloisKeys: keys,
		Query: query,
	}
}

func (c *Context) Decrypt(res [][]rlwe.Ciphertext) [][]float64 {

		var results [][]float64

		for _, face := range res {
			var distances []float64
			for _, target := range face{

				// Decrypt the result back into plaintext
				decryptedPlaintext := c.Decryptor.DecryptNew(&target)

				// Decode the decrypted result into a float64 slice
				have := make([]float64, c.Params.MaxSlots())
				if err := c.Encoder.Decode(decryptedPlaintext, have); err != nil {
					panic(err)
				}

				distances = append(distances, have[0])

			}
			results = append(results, distances)
		}
		return results

}
