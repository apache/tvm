/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application deployment over tvm.
 * \file simple.go
 */

package main

import (
    "fmt"
    "unsafe"
    "./gotvm"
)

// NNVM compiled model paths.
const (
    modLib    = "./deploy.so"
)

// main
func main() {
    // Welcome
    fmt.Printf("TVM Go Interface : v%v\n", gotvm.GoTVMVersion)
    fmt.Printf("TVM Version   : v%v\n", gotvm.TVMVersion)
    fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPackVersion)

    // Query global functions available
    funcNames := []string{}
    if gotvm.TVMFuncListGlobalNames(&funcNames) != 0 {
        fmt.Printf("%v", gotvm.TVMGetLastError())
        return
    }

    fmt.Printf("Global Functions:%v\n", funcNames)

    // Import tvm module (dso)
    var modp gotvm.TVMModule

    if gotvm.TVMModLoadFromFile(modLib, "so", &modp) != 0 {
        fmt.Printf("%v", gotvm.TVMGetLastError())
        return
    }
    defer gotvm.TVMModFree(modp)

    fmt.Printf("Module Imported\n")

    // TVMArray allocation attributes
    var ndim int32 = 1
    dtypeCode := gotvm.KDLFloat
    var dtypeBits int32 = 32
    var dtypeLanes int32 = 1
    deviceType := gotvm.KDLCPU
    var deviceID int32
    tshapeIn  := []int64{4}

    // Allocate input TVMArray : inX
    var inX gotvm.TVMArray

    if gotvm.TVMArrayAlloc(tshapeIn, ndim, dtypeCode, dtypeBits, dtypeLanes,
                           deviceType, deviceID, &inX) != 0 {
        fmt.Printf("%v", gotvm.TVMGetLastError())
        return
    }
    defer gotvm.TVMArrayFree(inX)

    // Allocate input TVMArray : inY
    var inY gotvm.TVMArray

    if gotvm.TVMArrayAlloc(tshapeIn, ndim, dtypeCode, dtypeBits, dtypeLanes,
                           deviceType, deviceID, &inY) != 0 {
        fmt.Printf("%v", gotvm.TVMGetLastError())
        return
    }
    defer gotvm.TVMArrayFree(inY)

    // Allocate output TVMArray
    var out gotvm.TVMArray

    tshapeOut := []int64{4}

    if gotvm.TVMArrayAlloc(tshapeOut, ndim, dtypeCode, dtypeBits, dtypeLanes,
                           deviceType, deviceID, &out) != 0 {
        fmt.Printf("%v", gotvm.TVMGetLastError())
        return
    }
    defer gotvm.TVMArrayFree(out)

    fmt.Printf("Input and Output TVMArrays allocated\n")

    // Fill Input Data : inX , inY
    // We use unsafe package to access underlying array to any type.
    inXSlice := (*[1<<15] float32)(unsafe.Pointer(inX.GetData()))[:4:4]
    inYSlice := (*[1<<15] float32)(unsafe.Pointer(inY.GetData()))[:4:4]

    for ii := 0; ii < 4 ; ii++ {
        inXSlice[ii] = float32(ii)
        inYSlice[ii] = float32(ii+5)
    }

    fmt.Printf("X: %v\n", inXSlice)
    fmt.Printf("Y: %v\n", inYSlice)

    // Call module function myadd
    _, _, tvmerr := gotvm.TVMFunctionExec(modp, "myadd", inX, inY, out)
    if tvmerr != nil {
        fmt.Print(tvmerr)
        return
    }

    fmt.Printf("Module function myadd executed\n")

    // We use unsafe package to access underlying array to any type.
    outSlice := (*[1<<15] float32)(unsafe.Pointer(out.GetData()))[:4:4]
    fmt.Printf("Result:%v\n", outSlice)
}
