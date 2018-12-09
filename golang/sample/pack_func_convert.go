/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application to demonstrate function conversion to packed function.
 * \file pack_func_convert.go
 */

package main

import (
    "fmt"
    "./gotvm"
)

// sampleCb is a simple golang callback function like C = A + B.
func sampleCb(args ...*gotvm.Value) (retVal interface{}, err error) {
    for _, v := range args {
        fmt.Printf("ARGS:%T : %v\n", v.AsInt64(), v.AsInt64())
    }
    val1 := args[0].AsInt64()
    val2 := args[1].AsInt64()
    retVal = int64(val1+val2)
    return
}

// main
func main() {
    // Welcome

    // Simple convert to a packed function
    fhandle, err := gotvm.ConvertFunction(sampleCb)
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Converted function\n")

    retVal, err := fhandle.Invoke(10, 20)
    fmt.Printf("Invoke Completed\n")
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Result:%v\n", retVal.AsInt64())
}
