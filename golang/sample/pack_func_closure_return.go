/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application to demonstrate go-closure returned from a callback function.
 * \file pack_func_closure_return.go
 */

package main

import (
    "fmt"
    "./gotvm"
)

// sampleFunctionCb returns a function closure which is embed as packed function in TVMValue.
func sampleFunctionCb(args ...*gotvm.Value) (retVal interface{}, err error) {
    funccall := func (cargs ...*gotvm.Value) (fret interface{}, ferr error) {
        for _, v := range cargs {
            fmt.Printf("ARGS:%T : %v\n", v.AsInt64(), v.AsInt64())
        }
        val1 := cargs[0].AsInt64()
        val2 := cargs[1].AsInt64()
        fret = int64(val1+val2)
        return
    }
    retVal = funccall
    return
}

// main
func main() {
    // Not passing a function name implicitely
    // picks the name from reflection as "main.sampleDunctionCb"
    gotvm.RegisterFunction(sampleFunctionCb);
    fmt.Printf("Registered: sampleFunctionCb\n")

    // Get registered global function
    funp, err := gotvm.GetGlobalFunction("main.sampleFunctionCb")
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("GetGlobalFunction: main.sampleFunctionCb - Success\n")

    // Call function
    result, err := funp.Invoke()
    if err != nil {
        fmt.Print(err)
        return
    }
    fmt.Printf("Invoked main.sampleFunctionCb via Function handle\n")

    pfunc := result.AsFunction()
    fmt.Printf("Function Handle received via Packed Function call:%T - %v \n", pfunc, pfunc)

    pfuncRet, err := pfunc.Invoke(30, 40)
    fmt.Printf("Invoked closure inside sampleFunctionCb result:%v\n", pfuncRet.AsInt64())
}
