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
