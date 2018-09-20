/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application to demonstrate go-closure given to a packed function argument.
 * \file pack_func_closure_arg.go
 */

package main

import (
    "fmt"
    "./gotvm"
)


// sampleFunctionArg receives a Packed Function handle and calls it.
func sampleFunctionArg(args ...*gotvm.Value) (retVal interface{}, err error) {
    // Reveive Packed Function Handle
    pfunc := args[0].AsFunction()
    // Call Packed Function
    retVal, err = pfunc.Invoke(args[1].AsInt64(), args[2].AsInt64())
    return
}

// main
func main() {
    // Not passing a function name implicitely
    // picks the name from reflection as "main.sampleDunctionArg"
    gotvm.RegisterFunction(sampleFunctionArg);
    fmt.Printf("Registered: sampleFunctionArg\n")

    // Get registered global function.
    funp, err := gotvm.GetGlobalFunction("main.sampleFunctionArg")
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("GetGlobalFunction: main.sampleFunctionArg - Success\n")

    // funccall is a simple golang callback function like C = A + B.
    funccall := func (args ...*gotvm.Value) (retVal interface{}, err error) {
        for _, v := range args {
            fmt.Printf("ARGS:%T : %v\n", v.AsInt64(), v.AsInt64())
        }
        val1 := args[0].AsInt64()
        val2 := args[1].AsInt64()
        retVal = int64(val1+val2)
        return
    }

    // Call function
    result, err := funp.Invoke(funccall, 30, 50)
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Invoked sampleFunctionArg with function closure arg : Result:%v\n", result.AsInt64())
}
