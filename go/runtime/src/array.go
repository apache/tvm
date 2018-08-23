/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for TVMArray aka DLTensor
 * \file array.go
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


func (parray Array) nativeCopyFrom(data C.native_voidp, datalen int) (err error) {
    ret := C._TVMArrayCopyFromBytes(C.native_voidp(parray.nativeCPtr()), data, C.int(datalen))

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
    var data C.native_voidp
    var datalen int

    dtype := C._DLTensorGetDType(C.uintptr_t(parray))

    switch val.(type) {
        case []int8:
            sliceVal := val.([]int8)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []int16:
            sliceVal := val.([]int16)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []int32:
            sliceVal := val.([]int32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []int64:
            sliceVal := val.([]int64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []uint8:
            sliceVal := val.([]uint8)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
         case []uint16:
            sliceVal := val.([]uint16)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []uint32:
            sliceVal := val.([]uint32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []uint64:
            sliceVal := val.([]uint64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []float32:
            sliceVal := val.([]float32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        case []float64:
            sliceVal := val.([]float64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return parray.nativeCopyFrom(data, datalen)
        default:
            err = fmt.Errorf("Given type not supported : %v\n", reflect.TypeOf(val))
            return
    }

    return
}

func (parray Array) nativeCopyTo (data C.native_voidp, datalen int) (err error){
    ret := C._TVMArrayCopyToBytes(C.native_voidp(parray.nativeCPtr()), data, C.int(datalen))

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

    for ii := range shape {
        size *= shape[ii]
    }

    var data C.native_voidp
    var datalen int

    dtype := C._DLTensorGetDType(C.uintptr_t(parray))

    switch parray.GetDType() {
        case "int8":
            sliceVal := make([]int8, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "int16":
            sliceVal := make([]int16, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "int32":
            sliceVal := make([]int32, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "int64":
            sliceVal := make([]int64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint8":
            sliceVal := make([]uint8, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint16":
            sliceVal := make([]uint16, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint32":
            sliceVal := make([]uint32, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint64":
            sliceVal := make([]uint64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "float32":
            sliceVal := make([]float32, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "float64":
            sliceVal := make([]float64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = parray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        default:
            err = fmt.Errorf("Given type not supported : %v\n", parray.GetDType())
            return
    }

    return
}

// GetNdim returns the number of dimentions in Array
func (parray Array) GetNdim() (retVal int32) {
    retVal = (int32)(C._DLTensorGetNdim(C.uintptr_t(parray)))
    return
}

// GetShape returns the number of dimentions in Array
func (parray Array) GetShape() (retVal []int64) {
    shapeArr :=  C._DLTensorGetShape(C.uintptr_t(parray))
    ndim := parray.GetNdim()

    shapeSlice := (*[1<<31] int64)(unsafe.Pointer(shapeArr))[:ndim:ndim]
    retVal = make([]int64, ndim)
    copy(retVal, shapeSlice)

    return
}

// GetDType returns the number of dimentions in Array
func (parray Array) GetDType() (retVal string) {
    ret := C._DLTensorGetDType(C.uintptr_t(parray))
    retVal, _ = dtypeFromTVMType(*(*pTVMType)(unsafe.Pointer(&ret)))
    return
}

// GetCtx returns the number of dimentions in Array
func (parray Array) GetCtx() (retVal Context) {
    ret := C._DLTensorGetCtx(C.uintptr_t(parray))
    retVal = *(*Context)(unsafe.Pointer(&ret))
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
    ret := (int32)(C._TVMArrayAlloc(C.native_voidp(&(shape[0])), C.int(ndim),
                                   C.int(dtypeCode), C.int(dtypeBits), C.int(dtypeLanes),
                                   C.int(deviceType), C.int(deviceID), C.native_voidp(&retVal)))

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
//        `args[1]` is Context. Default value is '{KDLCPU, 0}'
//
// returns pointer to Array on successful execution and error if any.
func Empty(shape []int64, args ...interface{}) (parray *Array, err error) {
    typeName := "float32"
    ctx := Context{KDLCPU, 0}

    if len(shape) < 1 {
        err = fmt.Errorf("Invalid shape for Array creation: %v\n", len(shape))
        return
    }

    for i, val := range args {
        switch val.(type) {
            case string:
                typeName = args[i].(string)
            case Context:
                ctx = args[i].(Context)
            default:
                err = fmt.Errorf("Invalid Optional Argument Type: %T\n", val)
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
                                    ctx.DeviceType, ctx.DeviceID)

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
    retVal = (int32)(C._TVMArrayFree(C.native_voidp(parray.nativeCPtr())))
    return
}
