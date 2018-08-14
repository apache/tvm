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
)

// NNVM compiled model paths.
const (
    modLib    = "./deploy.so"
)

// main
func main() {
    // Welcome
    defer runtime.GC()
    fmt.Printf("TVM Go Interface : v%v\n", gotvm.GoTVMVersion)
    fmt.Printf("TVM Version   : v%v\n", gotvm.TVMVersion)
    fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPackVersion)

    // Import tvm module (so)
    modp, _ := gotvm.LoadModuleFromFile(modLib)
    fmt.Printf("Module Imported\n")


    // Allocate TVMArray for inputs and outputs.

    // Allocation by explicit type and context.
    tshapeIn  := []int64{4}
    inX, _ := gotvm.Empty(tshapeIn, "float32", gotvm.Context{gotvm.KDLCPU, 0})

    // Default allocation on CPU
    inY, _ := gotvm.Empty(tshapeIn, "float32")

    // Default allocation to type "float32" and on CPU
    out, _ := gotvm.Empty(tshapeIn)

    fmt.Printf("Input and Output TVMArrays allocated\n")

    // Fill Input Data : inX , inY
    inXSlice := []float32 {1, 2, 3, 4}
    inYSlice := []float32 {5, 6, 7, 8}

    // Copy the data on target memory through runtime CopyFrom api.
    inX.CopyFrom(inXSlice)
    inY.CopyFrom(inYSlice)

    fmt.Printf("X: %v\n", inXSlice)
    fmt.Printf("Y: %v\n", inYSlice)

    // Get function "myadd"
    funp, _ := modp.GetFunction("myadd")

    // Call function
    funp(inX, inY, out)

    fmt.Printf("Module function myadd executed\n")

    // Get the output tensor as an interface holding a slice through runtime CopyTo api.
    outSlice, _ := out.AsSlice()

    // Print results
    fmt.Printf("Result:%v\n", outSlice.([]float32))
}
