package main

import (
	"fmt"
	"reflect"
)

func PrintTypeAndAttributes(v interface{}) {
	// Get the type and value of the passed object using reflect
	val := reflect.ValueOf(v)
	typ := reflect.TypeOf(v)

	// Print the type
	fmt.Printf("Type: %s\n", typ)

	// If the passed value is a struct, we can inspect its fields
	if val.Kind() == reflect.Struct {
		// Iterate through all fields of the struct
		for i := 0; i < val.NumField(); i++ {
			// Get the field name and value
			field := val.Field(i)
			fieldType := typ.Field(i)
			// Print field name, type, and value
			fmt.Printf("Field: %s\n", fieldType.Name)
			fmt.Printf(" - Type: %s\n", fieldType.Type)
			fmt.Printf(" - Value: %v\n", field)
		}
	}
}
