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
 * \brief gotvm package
 * \file function_test.go
 */

package gotvm

import (
    "testing"
    "reflect"
    "math/rand"
    "strings"
    "fmt"
)

// Check global function list API
func TestFunctionGlobals(t *testing.T) {
    funcNames, err := FuncListGlobalNames()
    if err != nil {
        t.Error(err.Error())
        return
    }
    if len(funcNames) < 1 {
        t.Errorf("Global Function names received:%v\n", funcNames)
    }
}

// Check GetFunction API
func TestFunctionGlobalGet(t *testing.T) {
    funp, err := GetGlobalFunction("tvm.graph_executor.create")
    if err != nil {
        t.Error(err.Error())
        return
    }
    if reflect.TypeOf(funp).Kind() != reflect.Ptr {
        t.Error("Function type mis matched\n")
        return
    }
}

func TestFunctionModuleGet(t *testing.T) {
    modp, err := LoadModuleFromFile("./deploy.so")
    if err != nil {
        t.Error(err.Error())
        return
    }
    funp, err := modp.GetFunction("myadd")
    if err != nil {
        t.Error(err.Error())
        return
    }
    if reflect.TypeOf(funp).Kind() != reflect.Ptr {
        t.Error("Function type mis matched\n")
        return
    }

    dlen := int64(1024)
    shape := []int64{dlen}
    inX, _ := Empty(shape)
    inY, _ := Empty(shape)
    out, _ := Empty(shape)
    dataX := make([]float32, (dlen))
    dataY := make([]float32, (dlen))
    outExpected :=  make([]float32, (dlen))

    for i := range dataX {
        dataX[i] = rand.Float32()
        dataY[i] = rand.Float32()
        outExpected[i] = dataX[i] + dataY[i]
    }

    inX.CopyFrom(dataX)
    inY.CopyFrom(dataY)

    funp.Invoke(inX, inY, out)
    outi, _ := out.AsSlice()
    outSlice := outi.([]float32)
    if len(outSlice) != len(outExpected) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(outExpected), len(outSlice))
            return
    }
    for i := range outSlice {
        if outExpected[i] != outSlice[i] {
            t.Errorf("Data expected: %v Got :%v at index %v\n", outExpected[i], outSlice[i], i)
            return
        }
    }
}

// Check FunctionConvert API
func TestFunctionConvert(t *testing.T) {
    sampleCb := func (args ...*Value) (retVal interface{}, err error) {
        val1 := args[0].AsInt64()
        val2 := args[1].AsInt64()
        retVal = int64(val1+val2)
        return
    }

    fhandle, err := ConvertFunction(sampleCb)
    if err != nil {
        t.Error(err.Error())
        return
    }

    retVal, err := fhandle.Invoke(10, 20)
    if err != nil {
        t.Error(err.Error())
        return
    }

    if retVal.AsInt64() != int64(30) {
        t.Errorf("Expected result :30 got:%v\n", retVal.AsInt64())
        return
    }
}

func TestFunctionError(t *testing.T) {
    sampleCb := func (args ...*Value) (retVal interface{}, err error) {
        err = fmt.Errorf("Sample Error XYZABC");
        return
    }

    fhandle, err := ConvertFunction(sampleCb)
    if err != nil {
        t.Error(err.Error())
        return
    }

    _, err = fhandle.Invoke()
    if err == nil {
        t.Error("Expected error but didn't received\n")
        return
    }

    if  !strings.Contains(err.Error(), string("Sample Error XYZABC")) {
        t.Errorf("Expected Error should contain :\"Sample Error XYZABC\" got :%v\n", err.Error())
    }
}

// Check FunctionRegister
func TestFunctionRegister(t *testing.T) {
    sampleCb := func (args ...*Value) (retVal interface{}, err error) {
        val1 := args[0].AsInt64()
        val2 := args[1].AsInt64()
        retVal = int64(val1+val2)
        return
    }

    RegisterFunction(sampleCb, "TestFunctionRegister.sampleCb");
    // Query global functions available
    funcNames, err := FuncListGlobalNames()
    if err != nil {
        t.Error(err.Error())
        return
    }

    found := 0
    for ii := range (funcNames) {
        if strings.Compare(funcNames[ii], "TestFunctionRegister.sampleCb") == 0 {
            found = 1
        }
    }
    if found == 0 {
        t.Error("Registered function not found in global function list.")
        return
    }

    // Get "sampleCb" and verify the call.
    funp, err := GetGlobalFunction("TestFunctionRegister.sampleCb")
    if err != nil {
        t.Error(err.Error())
        return
    }

    // Call function
    result, err := funp.Invoke((int64)(10), (int64)(20))
    if err != nil {
        t.Error(err.Error())
        return
    }
    if result.AsInt64() != int64(30) {
        t.Errorf("Expected result :30 got:%v\n", result.AsInt64())
        return
    }
}

// Check packed function receiving go-closure as argument.
func TestFunctionClosureArg(t *testing.T) {
    // sampleFunctionArg receives a Packed Function handle and calls it.
    sampleFunctionArg := func (args ...*Value) (retVal interface{}, err error) {
        // Reveive Packed Function Handle
        pfunc := args[0].AsFunction()

        // Call Packed Function by Value
        ret, err := pfunc.Invoke(args[1], args[2])
        if err != nil {
            return
        }

        // Call Packed Function with extracted values
        ret1, err := pfunc.Invoke(args[1].AsInt64(), args[2].AsInt64())
        if err != nil {
            return
        }
        if ret1.AsInt64() != ret.AsInt64() {
            err = fmt.Errorf("Invoke with int64 didn't match with Value")
            return
        }
        retVal = ret
        return
    }

    RegisterFunction(sampleFunctionArg, "TestFunctionClosureArg.sampleFunctionArg");
    funp, err := GetGlobalFunction("TestFunctionClosureArg.sampleFunctionArg")
    if err != nil {
        t.Error(err.Error())
        return
    }

    // funccall is a simple golang callback function like C = A + B.
    funccall := func (args ...*Value) (retVal interface{}, err error) {
        val1 := args[0].AsInt64()
        val2 := args[1].AsInt64()
        retVal = int64(val1+val2)
        return
    }

    // Call function
    result, err := funp.Invoke(funccall, 30, 50)
    if err != nil {
        t.Error(err.Error())
        return
    }

    if result.AsInt64() != int64(80) {
        t.Errorf("Expected result :80 got:%v\n", result.AsInt64())
        return
    }
}

// Check packed function returning a go-closure.
func TestFunctionClosureReturn(t *testing.T) {
    // sampleFunctionCb returns a function closure which is embed as packed function in TVMValue.
    sampleFunctionCb := func (args ...*Value) (retVal interface{}, err error) {
        funccall := func (cargs ...*Value) (fret interface{}, ferr error) {
            val1 := cargs[0].AsInt64()
            val2 := cargs[1].AsInt64()
            fret = int64(val1+val2)
            return
        }
        retVal = funccall
        return
    }

    RegisterFunction(sampleFunctionCb, "TestFunctionClosureReturn.sampleFunctionCb");
    funp, err := GetGlobalFunction("TestFunctionClosureReturn.sampleFunctionCb")
    if err != nil {
        t.Error(err.Error())
        return
    }

    // Call function
    result, err := funp.Invoke()
    if err != nil {
        t.Error(err.Error())
        return
    }

    pfunc := result.AsFunction()
    pfuncRet, err := pfunc.Invoke(30, 40)
    if err != nil {
        t.Error(err.Error())
        return
    }
    if pfuncRet.AsInt64() != int64(70) {
        t.Errorf("Expected result :70 got:%v\n", pfuncRet.AsInt64())
        return
    }
}

// Check packed function with no arguments and no return values.
func TestFunctionNoArgsReturns(t *testing.T) {
    sampleFunction := func (args ...*Value) (retVal interface{}, err error) {
        return
    }

    fhandle, err := ConvertFunction(sampleFunction)
    if err != nil {
        t.Error(err.Error())
        return
    }

    _, err = fhandle.Invoke()
    if err != nil {
        t.Error(err.Error())
        return
    }
}

// Check packed function returning a go-closure with no arg and returns.
func TestFunctionNoArgsReturns2(t *testing.T) {
    // sampleFunctionCb returns a function closure which is embed as packed function in TVMValue.
    sampleFunctionCb := func (args ...*Value) (retVal interface{}, err error) {
        funccall := func (cargs ...*Value) (fret interface{}, ferr error) {
            return
        }
        retVal = funccall
        return
    }

    funp, err := ConvertFunction(sampleFunctionCb)
    if err != nil {
        t.Error(err.Error())
        return
    }

    // Call function
    result, err := funp.Invoke()
    if err != nil {
        t.Error(err.Error())
        return
    }

    pfunc := result.AsFunction()
    _, err = pfunc.Invoke()
    if err != nil {
        t.Error(err.Error())
        return
    }
}
