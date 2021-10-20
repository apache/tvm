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
 * \brief gotvm package source for TVMArray aka DLTensor
 * \file ndarray.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
    "fmt"
    "errors"
    "runtime"
    "reflect"
)

// Array type in golang hold pointer for the TVMArray object from dlpack.
//
// Array initialization happen through Empty api
type Array uintptr

// nativeCPtr returns type freed uintptr for the Array.
func (parray Array) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(parray)
    return
}

func (parray Array) nativeCopyFrom(data unsafe.Pointer, datalen int) (err error) {
    ret := C.TVMArrayCopyFromBytes((*C.DLTensor)(unsafe.Pointer(parray.nativeCPtr())),
                                   data,
                                   C.ulong(datalen))
    if ret != 0 {
        err = errors.New(getTVMLastError())
    }
    return
}

// CopyFrom copies given golang data slice into Array.
//
// `val` is interface homding a slice of Array data type.
//
// returns err is any.
// TOD: Use reflections for better handling
func (parray Array) CopyFrom(val interface{}) (err error) {
    var data unsafe.Pointer
    var datalen int
    dtype := ((*C.DLTensor)(unsafe.Pointer(parray))).dtype

    switch val.(type) {
        case []int8:
            sliceVal := val.([]int8)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []int16:
            sliceVal := val.([]int16)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []int32:
            sliceVal := val.([]int32)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []int64:
            sliceVal := val.([]int64)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []uint8:
            sliceVal := val.([]uint8)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
         case []uint16:
            sliceVal := val.([]uint16)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []uint32:
            sliceVal := val.([]uint32)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []uint64:
            sliceVal := val.([]uint64)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []float32:
            sliceVal := val.([]float32)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []float64:
            sliceVal := val.([]float64)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        default:
            err = fmt.Errorf("Given type not supported : %v", reflect.TypeOf(val))
            return
    }
    return
}

func (parray Array) nativeCopyTo (data unsafe.Pointer, datalen int) (err error){
    ret := C.TVMArrayCopyToBytes((*C.DLTensor)(unsafe.Pointer(parray.nativeCPtr())),
                                  unsafe.Pointer(data),
                                  C.ulong(datalen))

    if ret != 0 {
        err = errors.New(getTVMLastError())
    }
   return
}

// AsSlice returns the unitptr of for the data inside Array.
//
// returns the slice of array inside Array and err of any.
// TOD: Use reflections for better handling
func (parray Array) AsSlice() (retVal interface{}, err error) {
    shape := parray.GetShape()
    size := int64(1)
    var data unsafe.Pointer
    var datalen int

    for ii := range shape {
        size *= shape[ii]
    }
    dtype := ((*C.DLTensor)(unsafe.Pointer(parray))).dtype

    switch parray.GetDType() {
        case "int8":
            sliceVal := make([]int8, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "int16":
            sliceVal := make([]int16, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "int32":
            sliceVal := make([]int32, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "int64":
            sliceVal := make([]int64, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint8":
            sliceVal := make([]uint8, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint16":
            sliceVal := make([]uint16, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint32":
            sliceVal := make([]uint32, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint64":
            sliceVal := make([]uint64, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "float32":
            sliceVal := make([]float32, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "float64":
            sliceVal := make([]float64, size)
            data = unsafe.Pointer(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        default:
            err = fmt.Errorf("Given type not supported : %v", parray.GetDType())
            return
    }
    return
}

// GetNdim returns the number of dimentions in Array
func (parray Array) GetNdim() (retVal int32) {
    retVal = int32(((*C.DLTensor)(unsafe.Pointer(parray))).ndim)
    return
}

// GetShape returns the number of dimentions in Array
func (parray Array) GetShape() (retVal []int64) {
    shapePtr := (*C.int64_t)(((*C.DLTensor)(unsafe.Pointer(parray))).shape)
    ndim := parray.GetNdim()

    shapeSlice := (*[1<<31] int64)(unsafe.Pointer(shapePtr))[:ndim:ndim]
    retVal = make([]int64, ndim)
    copy(retVal, shapeSlice)
    return
}

// GetDType returns the number of dimentions in Array
func (parray Array) GetDType() (retVal string) {
    ret := ((*C.DLTensor)(unsafe.Pointer(parray))).dtype
    retVal, _ = dtypeFromTVMType(*(*pTVMType)(unsafe.Pointer(&ret)))
    return
}

// GetDevice returns the number of dimentions in Array
func (parray Array) GetDevice() (retVal Device) {
    ret := ((*C.DLTensor)(unsafe.Pointer(parray))).device
    retVal = *(*Device)(unsafe.Pointer(&ret))
    return
}

// nativeTVMArrayAlloc is used to allocate TVMArray from given attributes.
//
// `shape` is int64 slice holding shape of the Array to be created.
//
// `ndim` is the rank of the Array to be created.
//
// `dtypeCode`, `dtypeBits` and `dtypeLanes` describe the data type in Array.
//
// `deviceType` indicates the device on whose memory the Array to allocated.
//
// `deviceID` indicates device index if multiple devices of same type present.
//
// return argument holding native pointer to newly created Array and error is any.
func nativeTVMArrayAlloc(shape []int64, ndim int32,
                   dtypeCode int32, dtypeBits int32, dtypeLanes int32,
                   deviceType int32, deviceID int32) (retVal uintptr, err error) {
    ret := (int32)(C.TVMArrayAlloc((*C.long)(&(shape[0])),
                                   C.int(ndim),
                                   C.int(dtypeCode),
                                   C.int(dtypeBits),
                                   C.int(dtypeLanes),
                                   C.int(deviceType),
                                   C.int(deviceID),
                                   (*C.TVMArrayHandle)(unsafe.Pointer(&retVal))))
    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }
    return
}

// Empty is used to allocate TVM empty array of given epecification.
//
// `shape` is int64 slice holding shape of the Array
//
// `args` is variadic args for
//
//        `args[0]` is string for data type. Default value is 'float32'
//
//        `args[1]` is Device. Default value is '{KDLCPU, 0}'
//
// returns pointer to Array on successful execution and error if any.
func Empty(shape []int64, args ...interface{}) (parray *Array, err error) {
    typeName := "float32"
    dev := Device{KDLCPU, 0}

    if len(shape) < 1 {
        err = fmt.Errorf("Invalid shape for Array creation: %v", len(shape))
        return
    }

    for i, val := range args {
        switch val.(type) {
            case string:
                typeName = args[i].(string)
            case Device:
                dev = args[i].(Device)
            default:
                err = fmt.Errorf("Invalid Optional Argument Type: %T", val)
                return
        }
    }

    tvmType, err := dtypeToTVMType(typeName)
    if err != nil {
        return
    }
    ndim := int32(len(shape))
    newArray, err := nativeTVMArrayAlloc(shape, ndim, int32(tvmType.code),
                                    int32(tvmType.bits), int32(tvmType.lanes),
                                    dev.DeviceType, dev.DeviceID)
    if err != nil {
        return
    }
    handle := new(Array)
    *handle = Array(newArray)

    finalizer := func (ahandle *Array) {
        nativeTVMArrayFree(*ahandle)
        ahandle = nil
    }
    runtime.SetFinalizer(handle, finalizer)
    parray = handle
    return
}

// nativeTVMArrayFree is used to release the Array.
//
// `parray` is the Array handle.
//
// `ret` indicates the status of this api execution.
func nativeTVMArrayFree(parray Array) (retVal int32) {
    retVal = (int32)(C.TVMArrayFree((*C.DLTensor)(unsafe.Pointer(parray.nativeCPtr()))))
    return
}
