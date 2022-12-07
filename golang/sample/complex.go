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
 * \file complex.go
 */

package main

import (
    "fmt"
    "io/ioutil"
    "math/rand"
    "./gotvm"
    "runtime"
)

// NNVM compiled model paths.
const (
    modLib    = "./mobilenet.so"
    modJSON   = "./mobilenet.json"
    modParams = "./mobilenet.params"
)

// main
func main() {
    defer runtime.GC()
    // Welcome
    fmt.Printf("TVM Version   : v%v\n", gotvm.TVMVersion)
    fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPackVersion)

    // Query global functions available
    funcNames, err := gotvm.FuncListGlobalNames()
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Global Functions:%v\n", funcNames)

    // Import tvm module (so)
    modp, err := gotvm.LoadModuleFromFile(modLib)
    if err != nil {
        fmt.Print(err)
        fmt.Printf("Please copy tvm compiled modules here and update the sample.go accordingly.\n")
        fmt.Printf("You may need to update modLib, modJSON, modParams, tshapeIn, tshapeOut\n")
        return
    }
    fmt.Printf("Module Imported:%p\n", modp)
    bytes, err := ioutil.ReadFile(modJSON)
    if err != nil {
        fmt.Print(err)
        return
    }
    jsonStr := string(bytes)

    // Load module on tvm runtime - call tvm.graph_executor.create
    funp, err := gotvm.GetGlobalFunction("tvm.graph_executor.create")
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Calling tvm.graph_executor.create\n")
    // Call function
    graphrt, err := funp.Invoke(jsonStr, modp, (int64)(gotvm.KDLCPU), (int64)(0))
    if err != nil {
        fmt.Print(err)
        return
    }
    graphmod := graphrt.AsModule()
    fmt.Printf("Graph executor Created\n")

    // Array allocation attributes
    tshapeIn  := []int64{1, 224, 224, 3}
    tshapeOut := []int64{1, 1001}

    // Allocate input Array
    inX, err := gotvm.Empty(tshapeIn, "float32", gotvm.CPU(0))
    if err != nil {
        fmt.Print(err)
        return
    }

    // Allocate output Array
    out, err := gotvm.Empty(tshapeOut)
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Input and Output Arrays allocated\n")

    // Get module function from graph executor : load_params
    // Read params
    bytes, err = ioutil.ReadFile(modParams)
    if err != nil {
        fmt.Print(err)
    }

    // Load Params
    funp, err = graphmod.GetFunction("load_params")
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Func load_params:%p\n", funp)

    // Call function
    _, err = funp.Invoke(bytes)
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Module params loaded\n")

    // Set some data in input Array
    inSlice := make([]float32, (224 * 224 * 3))
    rand.Seed(10)
    rand.Shuffle(len(inSlice), func(i, j int) {inSlice[i],
                                               inSlice[j] = rand.Float32(),
                                               rand.Float32() })
    inX.CopyFrom(inSlice)

    // Set Input
    funp, err = graphmod.GetFunction("set_input")
    if err != nil {
        fmt.Print(err)
        return
    }

    // Call function
    _, err = funp.Invoke("input", inX)
    if err != nil {
        fmt.Print(err)
        return
    }

    fmt.Printf("Module input is set\n")

    // Run
    funp, err = graphmod.GetFunction("run")
    if err != nil {
        fmt.Print(err)
        return
    }

    // Call function
    _, err = funp.Invoke()
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Module Executed \n")

    // Call runtime function get_output
    funp, err = graphmod.GetFunction("get_output")
    if err != nil {
        fmt.Print(err)
        return
    }

    // Call function
    _, err = funp.Invoke(int64(0), out)
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Got Module Output \n")

    // Print results
    outIntf, _ := out.AsSlice()
    outSlice := outIntf.([]float32)
    fmt.Printf("Result:%v\n", outSlice[:10])
}
