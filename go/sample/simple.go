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

    // Import tvm module (dso)
    modp, _ := gotvm.ModLoadFromFile(modLib)
    fmt.Printf("Module Imported\n")


    // Allocate input TVMArray : inX
    tshapeIn  := []int64{4}
    inX, _ := gotvm.EmptyArray(tshapeIn, gotvm.NewTVMType("float32"), gotvm.KDLCPU)

    // Allocate input TVMArray : inY
    inY, _ := gotvm.EmptyArray(tshapeIn, gotvm.NewTVMType("float32"), gotvm.KDLCPU)

    // Allocate output TVMArray : out
    out, _ := gotvm.EmptyArray(tshapeIn, gotvm.NewTVMType("float32"), gotvm.KDLCPU)

    fmt.Printf("Input and Output TVMArrays allocated\n")

    // Fill Input Data : inX , inY
    // We use unsafe package to access underlying array to any type.
    inXSlice := inX.GetData().([]float32)
    inYSlice := inY.GetData().([]float32)

    for ii := 0; ii < 4 ; ii++ {
        inXSlice[ii] = float32(ii)
        inYSlice[ii] = float32(ii+5)
    }

    fmt.Printf("X: %v\n", inXSlice)
    fmt.Printf("Y: %v\n", inYSlice)

    // Get function "myadd"
    funp, _ := modp.GetFunction("myadd")

    // Call function
    funp(inX, inY, out)

    fmt.Printf("Module function myadd executed\n")

    // We use unsafe package to access underlying array to any type.
    outSlice := out.GetData().([]float32)
    fmt.Printf("Result:%v\n", outSlice)
}
