/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application to demonstrate function callbacks in go
 * \file funccb.go
 */

package main

import (
    "fmt"
    "runtime"
    "./gotvm"
    "strings"
)

// sampleCb is a simple golang callback function like C = A + B.
func sampleCb(args ...gotvm.Value) (retVal interface{}, err error) {
    for i, v := range args {
        fmt.Printf("ARGS:%T: %v --- %T : %v\n",i, i, v, v)
    }

    val1 := args[0].AsInt64()
    val2 := args[1].AsInt64()

    retVal = int64(val1+val2)

    return
}

// sampleErrorCb is a callback function which returns a golang error.
func sampleErrorCb(args ...gotvm.Value) (retVal interface{}, err error) {

    err = fmt.Errorf("Callback function returns an error\n")

    return
}


// sampleFunctionCb returns a function closure which is embed as packed function in TVMValue.
func sampleFunctionCb(args ...gotvm.Value) (retVal interface{}, err error) {
    funccall := func (cargs ...gotvm.Value) (interface{}, error) {
        return sampleCb(cargs...)
    }

    retVal = funccall

    return
}

// main
func main() {
    // Welcome
    defer runtime.GC()
    fmt.Printf("TVM Go Interface : v%v\n", gotvm.GoTVMVersion)
    fmt.Printf("TVM Version   : v%v\n", gotvm.TVMVersion)
    fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPackVersion)


    fmt.Printf("\n\n ------ Register Function With TVM ------ \n")
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

    //TODO: funp here is a function closure.
    //      Do we convert this to Function handle to keep it common across.
    //      New method Invoke can be used to call.

    // Call function
    result, err := funp.Invoke((int64)(10), (int64)(20))
    if err != nil {
        fmt.Print(err)
        return
    }

    fmt.Printf("sampleCb result: %v\n", result.AsInt64())
    //TODO: Do we still need AsXXX wrappers?
    //      As packed function is defined to return interface{} instead.

    fmt.Printf("\n\n ------ Convert Function With TVM ------ \n")
    // Simple convert to a packed function
    fhandle, err := gotvm.ConvertFunction(sampleErrorCb)
    _, err = fhandle.Invoke()

    if err == nil {
        fmt.Printf("Expected err but not received via packed function\n")
    }

    fmt.Printf("Error received as expected as :###%v###\n ", err.Error())


    fmt.Printf("\n\n ------ Function Closure Return Type With TVM ------ \n")
    // Check function closure through packed function system.

    // Not passing a function name implicitely
    // picks the name from reflection as "main.sampleDunctionCb"
    gotvm.RegisterFunction(sampleFunctionCb);

    funp, err = gotvm.GetGlobalFunction("main.sampleFunctionCb")
    if err != nil {
        fmt.Print(err)
        return
    }

    // Call function
    result, err = funp.Invoke()
    if err != nil {
        fmt.Print(err)
        return
    }

    pfunc := result.AsFunction()
    pfuncRet, err := pfunc.Invoke((int64)(30), (int64)(40))
    fmt.Printf("sampleFunctionCb result:%v\n", pfuncRet.AsInt64())
}
