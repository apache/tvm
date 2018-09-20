/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application to demonstrate function register into TVM global functions.
 * \file pack_func_register.go
 */

package main

import (
    "fmt"
    "./gotvm"
    "strings"
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
    // Register sampleCb with TVM packed function system and call and check Global Function List.
    gotvm.RegisterFunction(sampleCb, "sampleCb");
    // Query global functions available
    funcNames, err := gotvm.FuncListGlobalNames()
    if err != nil {
        fmt.Print(err)
        return
    }

    found := 0
    for ii := range (funcNames) {
        if strings.Compare(funcNames[ii], "sampleCb") == 0 {
            found = 1
        }
    }
    if found == 0 {
        fmt.Printf("Function registerd but, not listed\n")
        return
    }


    // Get "sampleCb" and verify the call.
    funp, err := gotvm.GetGlobalFunction("sampleCb")
    if err != nil {
        fmt.Print(err)
        return
    }

    // Call function
    result, err := funp.Invoke((int64)(10), (int64)(20))
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("sampleCb result: %v\n", result.AsInt64())
}
