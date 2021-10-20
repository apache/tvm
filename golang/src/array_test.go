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
 * \file array_test.go
 */


package gotvm

import (
    "testing"
    "unsafe"
    "math/rand"
)

// Create an array and check size.
func TestArrayCreateSize(t *testing.T) {
    _, err := Empty([]int64{4})
    if err != nil {
        t.Error(err.Error())
        return
    }

    _, err = Empty([]int64{4, 5, 6})
    if err != nil {
        t.Error(err.Error())
        return
    }

    _, err = Empty([]int64{})
    if err == nil {
        t.Error("Expected err for empty Array created, but didn't got !!")
        return
    }
}

// Check array creation via various different arguments.
func TestArrayCreateArgs(t *testing.T) {
    _, err := Empty([]int64{4, 2}, "float32", CPU(0))
    if err != nil {
        t.Error(err.Error())
        return
    }

    _, err = Empty([]int64{4, 2}, "float32")
    if err != nil {
        t.Error(err.Error())
        return
    }

    _, err = Empty([]int64{4, 2}, CPU(0))
    if err != nil {
        t.Error(err.Error())
        return
    }

    _, err = Empty([]int64{4, 2}, CPU(0), "float32")
    if err != nil {
        t.Error(err.Error())
        return
    }
}

// Create an array and check the NDim.
func TestArrayNDim(t *testing.T) {
    arr, err := Empty([]int64{4, 5, 6})
    if err != nil {
        t.Error(err.Error())
        return
    }

    if 3 != arr.GetNdim() {
        t.Errorf("GetNdim failed Expected: 3 Got :%v\n", arr.GetNdim())
        return
    }
}

// Create an array and check Shape.
func TestArrayShape(t *testing.T) {
    arr, err := Empty([]int64{4, 5, 6})
    if err != nil {
        t.Error(err.Error())
        return
    }

    shape := arr.GetShape()
    if len(shape) != 3 {
        t.Errorf("Shape slice expected: 3 Got :%v\n", len(shape))
        return
    }

    if shape[0] != 4 || shape[1] != 5 || shape[2] != 6 {
        t.Errorf("Shape values expected {4, 5, 6} Got : %v\n", shape);
        return
    }
}

// Create an array and check created Device.
func TestArrayDevice(t *testing.T) {
    // TODO: Could some test cases for other targets
    arr, err := Empty([]int64{4}, CPU(0))
    if err != nil {
        t.Error(err.Error())
        return
    }

    dev := arr.GetDevice()
    if dev.DeviceType != KDLCPU {
        t.Errorf("Dev DeviceType expected: %v Got :%v\n", KDLCPU, dev.DeviceType)
        return
    }
    if dev.DeviceID != 0 {
        t.Errorf("Dev DeviceID expected: %v Got :%v\n", KDLCPU, dev.DeviceID)
        return
    }

    arr, err = Empty([]int64{4}, CPU(2))
    if err != nil {
        t.Error(err.Error())
        return
    }

    dev = arr.GetDevice()
    if dev.DeviceType != KDLCPU {
        t.Errorf("Dev DeviceType expected: %v Got :%v\n", KDLCPU, dev.DeviceType)
        return
    }
    if dev.DeviceID != 2 {
        t.Errorf("Dev DeviceID expected: %v Got :%v\n", KDLCPU, dev.DeviceID)
        return
    }
}

// Create array of different dtypes and check dtypes.
func TestArrayDType(t *testing.T) {
    for _, dtype := range  []string{"int8", "int16", "int32", "int64",
                                    "uint8", "uint16", "uint32", "uint64",
                                    "float32", "float64"} {
        arr, err := Empty([]int64{4}, dtype)
        if err != nil {
            t.Error(err.Error())
            return
        }

        if dtype != arr.GetDType() {
            t.Errorf("Dtype expected: %v Got :%v\n", dtype, arr.GetDType())
            return
        }
    }
}

// Copy Int8 data to created Array and verify.
func TestArrayCopySliceInt8(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "int8")

    if err != nil {
        t.Error(err.Error())
        return
    }

    bdata := make([]byte, dlen)
    rand.Read(bdata)
    data := (*[1<<31]int8)(unsafe.Pointer(&bdata[0]))[:dlen:dlen]

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []int8:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }

    dataRet := ret.([]int8)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}

// Copy Int16 data to created Array and verify.
func TestArrayCopySliceInt16(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "int16")
    if err != nil {
        t.Error(err.Error())
        return
    }

    bdata := make([]byte, dlen*2)
    rand.Read(bdata)
    data := (*[1<<31]int16)(unsafe.Pointer(&bdata[0]))[:dlen:dlen]

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }
    switch ret.(type) {
        case []int16:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }

    dataRet := ret.([]int16)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}

// Copy Int32 data to created Array and verify.
func TestArrayCopySliceInt32(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "int32")
    if err != nil {
        t.Error(err.Error())
        return
    }

    bdata := make([]byte, dlen*4)
    rand.Read(bdata)
    data := (*[1<<31]int32)(unsafe.Pointer(&bdata[0]))[:dlen:dlen]

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []int32:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }
    dataRet := ret.([]int32)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}

// Copy Int64 data to created Array and verify.
func TestArrayCopySliceInt64(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "int64")
    if err != nil {
        t.Error(err.Error())
        return
    }

    bdata := make([]byte, dlen*8)
    rand.Read(bdata)
    data := (*[1<<31]int64)(unsafe.Pointer(&bdata[0]))[:dlen:dlen]

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []int64:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }
    dataRet := ret.([]int64)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}

// Copy UInt8 data to created Array and verify.
func TestArrayCopySliceUInt8(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "uint8")
    if err != nil {
        t.Error(err.Error())
        return
    }

    bdata := make([]byte, dlen)
    rand.Read(bdata)
    data := (*[1<<31]uint8)(unsafe.Pointer(&bdata[0]))[:dlen:dlen]

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []uint8:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }
    dataRet := ret.([]uint8)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}

// Copy UInt16 data to created Array and verify.
func TestArrayCopySliceUInt16(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "uint16")
    if err != nil {
        t.Error(err.Error())
        return
    }

    bdata := make([]byte, dlen*2)
    rand.Read(bdata)
    data := (*[1<<31]uint16)(unsafe.Pointer(&bdata[0]))[:dlen:dlen]

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []uint16:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }
    dataRet := ret.([]uint16)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}

// Copy UInt32 data to created Array and verify.
func TestArrayCopySliceUInt32(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "uint32")
    if err != nil {
        t.Error(err.Error())
        return
    }

    bdata := make([]byte, dlen*4)
    rand.Read(bdata)
    data := (*[1<<31]uint32)(unsafe.Pointer(&bdata[0]))[:dlen:dlen]

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []uint32:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }
    dataRet := ret.([]uint32)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}

// Copy UInt64 data to created Array and verify.
func TestArrayCopySliceUInt64(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "uint64")
    if err != nil {
        t.Error(err.Error())
        return
    }

    bdata := make([]byte, dlen*8)
    rand.Read(bdata)
    data := (*[1<<31]uint64)(unsafe.Pointer(&bdata[0]))[:dlen:dlen]

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []uint64:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }
    dataRet := ret.([]uint64)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}

// Copy Float32 data to created Array and verify.
func TestArrayCopySliceFloat32(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "float32")
    if err != nil {
        t.Error(err.Error())
        return
    }

    data := make([]float32, dlen)

    for i := range data {
        data[i] = rand.Float32()
    }

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []float32:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }
    dataRet := ret.([]float32)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v \nGot :%v \n", data, dataRet)
            return
        }
    }
}

// Copy Float64 data to created Array and verify.
func TestArrayCopySliceFloat64(t *testing.T) {
    dlen := int64(32)
    arr, err := Empty([]int64{4, dlen/4}, "float64")
    if err != nil {
        t.Error(err.Error())
        return
    }

    data := make([]float64, dlen)

    for i := range data {
        data[i] = rand.Float64()
    }

    err = arr.CopyFrom(data)
    if err != nil {
        t.Error(err.Error())
        return
    }

    ret, err := arr.AsSlice()
    if err != nil {
        t.Error(err.Error())
        return
    }

    switch ret.(type) {
        case []float64:
        default:
            t.Errorf("Expected : %T but got :%T\n", data, ret)
            return
    }
    dataRet := ret.([]float64)
    if len(data) != len(dataRet) {
            t.Errorf("Data expected Len: %v Got :%v\n", len(data), len(dataRet))
            return
    }
    for i := range data {
        if data[i] != dataRet[i] {
            t.Errorf("Data expected: %v Got :%v\n", data, dataRet)
            return
        }
    }
}
