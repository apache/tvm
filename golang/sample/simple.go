/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application deployment over tvm.
 * \file simple.go
 */

package main

import (
    "fmt"
    "runtime"
    "./gotvm"
    "math/rand"
)

// NNVM compiled model paths.
const (
    modLib    = "./deploy.so"
)

// main
func main() {
    // Welcome
    defer runtime.GC()
    fmt.Printf("TVM Version   : v%v\n", gotvm.TVMVersion)
    fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPackVersion)

    // Import tvm module (so)
    modp, _ := gotvm.LoadModuleFromFile(modLib)
    fmt.Printf("Module Imported\n")


    // Allocate Array for inputs and outputs.
    // Allocation by explicit type and context.
    tshapeIn  := []int64{4}
    inX, _ := gotvm.Empty(tshapeIn, "float32", gotvm.CPU(0))

    // Default allocation on CPU
    inY, _ := gotvm.Empty(tshapeIn, "float32")

    // Default allocation to type "float32" and on CPU
    out, _ := gotvm.Empty(tshapeIn)
    fmt.Printf("Input and Output Arrays allocated\n")

    // Fill Input Data : inX , inY
    inXSlice := make([]float32, 4)
    inYSlice := make([]float32, 4)
    for i := range inXSlice {
        inXSlice[i] = rand.Float32()
        inYSlice[i] = rand.Float32()
    }


    // Copy the data on target memory through runtime CopyFrom api.
    inX.CopyFrom(inXSlice)
    inY.CopyFrom(inYSlice)
    fmt.Printf("X: %v\n", inXSlice)
    fmt.Printf("Y: %v\n", inYSlice)

    // Get function "myadd"
    funp, _ := modp.GetFunction("myadd")

    // Call function
    funp.Invoke(inX, inY, out)
    fmt.Printf("Module function myadd executed\n")

    // Get the output tensor as an interface holding a slice through runtime CopyTo api.
    outSlice, _ := out.AsSlice()

    // Print results
    fmt.Printf("Result:%v\n", outSlice.([]float32))
}
