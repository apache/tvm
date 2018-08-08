/*!
 *  Copyright (c) 2018 by Contributors
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
    fmt.Printf("TVM Go Interface : v%v\n", gotvm.GoTVMVersion)
    fmt.Printf("TVM Version   : v%v\n", gotvm.TVMVersion)
    fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPackVersion)

    // Query global functions available
    funcNames, err := gotvm.TVMFuncListGlobalNames()
    if err != nil {
        fmt.Print(err)
        return
    }

    fmt.Printf("Global Functions:%v\n", funcNames)

    // Import tvm module (dso)
    modp, err := gotvm.ModLoadFromFile(modLib)
    if err != nil {
        fmt.Print(err)
        fmt.Printf("Please copy tvm compiled modules here and update the sample.go accordingly.")
        fmt.Printf("You may need to update modLib, modJSON, modParams, tshapeIn, tshapeOut")
        return
    }
    fmt.Printf("Module Imported:%p\n", modp)

    bytes, err := ioutil.ReadFile(modJSON)
    if err != nil {
        fmt.Print(err)
        return
    }
    jsonStr := string(bytes)

    // Load module on tvm runtime - call tvm.graph_runtime.create
    funp, err := gotvm.GetGlobalFunction("tvm.graph_runtime.create")
    if err != nil {
        fmt.Print(err)
        return
    }

    // Call function
    graphrt, err := funp(jsonStr, modp, (int64)(gotvm.KDLCPU), (int64)(0))
    if err != nil {
        fmt.Print(err)
        return
    }

    graphmod := graphrt.(*gotvm.TVMModule)

    fmt.Printf("Graph runtime Created\n")

    // TVMArray allocation attributes
    tshapeIn  := []int64{1, 224, 224, 3}
    tshapeOut := []int64{1, 1001}

    // Allocate input TVMArray
    inX, err := gotvm.EmptyArray(tshapeIn, gotvm.NewTVMType("float32"), gotvm.KDLCPU)
    if err != nil {
        fmt.Print(err)
        return
    }

    // Allocate output TVMArray
    out, err := gotvm.EmptyArray(tshapeOut, gotvm.NewTVMType("float32"), gotvm.KDLCPU)
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Input and Output TVMArrays allocated\n")

    // Get module function from graph runtime : load_params
    // Read params
    bytes, err = ioutil.ReadFile(modParams)
    if err != nil {
        fmt.Print(err)
    }
    paramsByteArray := gotvm.NewTVMByteArray(bytes)

    // Load Params
    funp, err = graphmod.GetFunction("load_params")
    if err != nil {
        fmt.Print(err)
        return
    }

    fmt.Printf("Func load_params:%p\n", funp)

    // Call function
    _, err = funp(paramsByteArray)
    if err != nil {
        fmt.Print(err)
        return
    }

    fmt.Printf("Module params loaded\n")

    // Set some data in input TVMArray
    // We use unsafe package to access underlying array to any type.
    inSlice := inX.GetData().([]float32)
    rand.Seed(10)
    rand.Shuffle(len(inSlice), func(i, j int) {inSlice[i],
                                               inSlice[j] = rand.Float32(),
                                               rand.Float32() })

    // Set Input
    funp, err = graphmod.GetFunction("set_input")
    if err != nil {
        fmt.Print(err)
        return
    }

    // Call function
    _, err = funp("input", inX)
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
    _, err = funp()
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
    _, err = funp(int64(0), out)
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Got Module Output \n")

    // We use unsafe package to access underlying array to any type.
    outSlice := out.GetData().([]float32)
    fmt.Printf("Result:%v\n", outSlice[:10])
}
