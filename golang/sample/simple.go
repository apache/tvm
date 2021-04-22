/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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
    // Allocation by explicit type and device.
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
