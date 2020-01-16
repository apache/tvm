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
 * \file value_test.go
 */

package gotvm

import (
    "testing"
    "math/rand"
    "strings"
)

// Check Int64 Value looping via packed function calling another packed function.
func TestValueLoopInt64(t *testing.T) {
    // Receive a function Handle and argument and echo the Value on the handle.
    sampleFunctionLoop := func (args ...*Value) (retVal interface{}, err error) {
        // Reveive Packed Function Handle
        pfunc := args[0].AsFunction()
        newArgs := args[1:]

        // Call Packed Function by Value
        return pfunc.Invoke(newArgs)
    }

    fhandle, err := ConvertFunction(sampleFunctionLoop)
    if err != nil {
        t.Error(err.Error())
        return
    }

    // funccall is a simple golang callback function like C = A + B.
    funccall := func (args ...*Value) (retVal interface{}, err error) {
        retVal = args[0]
        return
    }

    result := rand.Int63()
    retVal, err := fhandle.Invoke(funccall, result)
    if err != nil {
        t.Error(err.Error())
        return
    }
    if retVal.AsInt64() != result {
        t.Errorf("Expected : %v got:%v\n", result, retVal.AsInt64())
        return
    }
}

// Check Int32 Value looping via packed function calling another packed function.
func TestValueLoopInt32(t *testing.T) {
    // Receive a function Handle and argument and echo the Value on the handle.
    sampleFunctionLoop := func (args ...*Value) (retVal interface{}, err error) {
        // Reveive Packed Function Handle
        pfunc := args[0].AsFunction()
        newArgs := args[1:]

        // Call Packed Function by Value
        return pfunc.Invoke(newArgs)
    }

    fhandle, err := ConvertFunction(sampleFunctionLoop)
    if err != nil {
        t.Error(err.Error())
        return
    }

    // funccall is a simple golang callback function like C = A + B.
    funccall := func (args ...*Value) (retVal interface{}, err error) {
        retVal = args[0]
        return
    }

    result := rand.Int31()
    retVal, err := fhandle.Invoke(funccall, result)
    if err != nil {
        t.Error(err.Error())
        return
    }

    if retVal.AsInt64() != int64(result) {
        t.Errorf("Expected : %v got:%v\n", result, retVal.AsInt64())
        return
    }
}

// Check Float32 Value looping via packed function calling another packed function.
func TestValueLoopFloat32(t *testing.T) {
    // Receive a function Handle and argument and echo the Value on the handle.
    sampleFunctionLoop := func (args ...*Value) (retVal interface{}, err error) {
        // Reveive Packed Function Handle
        pfunc := args[0].AsFunction()
        newArgs := args[1:]
        // Call Packed Function by Value
        return pfunc.Invoke(newArgs)
    }

    fhandle, err := ConvertFunction(sampleFunctionLoop)
    if err != nil {
        t.Error(err.Error())
        return
    }

    // funccall is a simple golang callback function like C = A + B.
    funccall := func (args ...*Value) (retVal interface{}, err error) {
        retVal = args[0]
        return
    }

    result := rand.Float32()
    retVal, err := fhandle.Invoke(funccall, result)
    if err != nil {
        t.Error(err.Error())
        return
    }

    if retVal.AsFloat64() != float64(result) {
        t.Errorf("Expected : %v got:%v\n", result, retVal.AsInt64())
        return
    }
}

// Check Float64 Value looping via packed function calling another packed function.
func TestValueLoopFloat64(t *testing.T) {
    // Receive a function Handle and argument and echo the Value on the handle.
    sampleFunctionLoop := func (args ...*Value) (retVal interface{}, err error) {
        // Reveive Packed Function Handle
        pfunc := args[0].AsFunction()
        newArgs := args[1:]
        // Call Packed Function by Value
        return pfunc.Invoke(newArgs)
    }

    fhandle, err := ConvertFunction(sampleFunctionLoop)
    if err != nil {
        t.Error(err.Error())
        return
    }

    // funccall is a simple golang callback function like C = A + B.
    funccall := func (args ...*Value) (retVal interface{}, err error) {
        retVal = args[0]
        return
    }

    result := rand.Float64()
    retVal, err := fhandle.Invoke(funccall, result)
    if err != nil {
        t.Error(err.Error())
        return
    }

    if retVal.AsFloat64() != result {
        t.Errorf("Expected : %v got:%v\n", result, retVal.AsInt64())
        return
    }
}

func TestValueLoopString(t *testing.T) {
    // Receive a function Handle and argument and echo the Value on the handle.
    sampleFunctionLoop := func (args ...*Value) (retVal interface{}, err error) {
        // Reveive Packed Function Handle
        pfunc := args[0].AsFunction()
        argStr := args[1].AsStr()
        // Call Packed Function by Value
        return pfunc.Invoke(argStr)
    }

    fhandle, err := ConvertFunction(sampleFunctionLoop)
    if err != nil {
        t.Error(err.Error())
        return
    }

    // funccall is a simple golang callback function like C = A + B.
    funccall := func (args ...*Value) (retVal interface{}, err error) {
        retVal =  args[0].AsStr()
        return
    }

    retVal, err := fhandle.Invoke(funccall, "TestString")
    if err != nil {
        t.Error(err.Error())
        return
    }

    vStr := retVal.AsStr()
    if strings.Compare(vStr, string("TestString")) != 0  {
        t.Errorf("Expected : %v got:%v\n", string("TestString"), vStr)
        return
    }
}

// Check []byte Value looping via packed function calling another packed function.
func TestValueLoopByteSlice(t *testing.T) {
    // Receive a function Handle and argument and echo the Value on the handle.
    sampleFunctionLoop := func (args ...*Value) (retVal interface{}, err error) {
        // Reveive Packed Function Handle
        pfunc := args[0].AsFunction()
        argBytes := args[1].AsBytes()
        // Call Packed Function by Value
        return pfunc.Invoke(argBytes)
    }

    fhandle, err := ConvertFunction(sampleFunctionLoop)
    if err != nil {
        t.Error(err.Error())
        return
    }

    // funccall is a simple golang callback function like C = A + B.
    funccall := func (args ...*Value) (retVal interface{}, err error) {
        retVal = args[0].AsBytes()
        return
    }

    result := make([]byte, 1024)
    rand.Read(result)
    retVal, err := fhandle.Invoke(funccall, result)
    if err != nil {
        t.Error(err.Error())
        return
    }

    received := retVal.AsBytes()
    if len(result) != len(received) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(result), len(received))
            return
    }
    for i := range result {
        if result[i] != received[i] {
            t.Errorf("Data expected: %v Got :%v at index %v\n", result[i], received[i], i)
            return
        }
    }
}
