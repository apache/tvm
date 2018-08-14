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

// TVMArray type in golang hold pointer for the TVMArray object from dlpack.
//
// TVMArray initialization happen through TVMArrayAlloc
type TVMArray uintptr

// nativeCPtr returns type freed uintptr for the TVMArray.
func (ptvmarray TVMArray) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(ptvmarray)
    return
}


func (ptvmarray TVMArray) nativeCopyFrom(data C.native_voidp, datalen int) (err error) {
    ret := C._TVMArrayCopyFromBytes(C.native_voidp(ptvmarray.nativeCPtr()), data, C.int(datalen))

    if ret != 0 {
        err = errors.New(getTVMLastError())
    }

    return
}

// CopyFrom copies given golang data slice into TVMArray.
//
// `val` is interface homding a slice of TVMArray data type.
//
// returns err is any.
// TOD: Use reflections for better handling
func (ptvmarray TVMArray) CopyFrom(val interface{}) (err error) {
    var data C.native_voidp
    var datalen int

    dtype := C._DLTensorGetDType(C.uintptr_t(ptvmarray))

    switch val.(type) {
        case []int32:
            sliceVal := val.([]int32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeCopyFrom(data, datalen)
        case []int64:
            sliceVal := val.([]int64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeCopyFrom(data, datalen)
        case []uint32:
            sliceVal := val.([]uint32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeCopyFrom(data, datalen)
        case []uint64:
            sliceVal := val.([]uint64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeCopyFrom(data, datalen)
        case []float32:
            sliceVal := val.([]float32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeCopyFrom(data, datalen)
        case []float64:
            sliceVal := val.([]float64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeCopyFrom(data, datalen)
        default:
            err = fmt.Errorf("Given type not supported : %v\n", reflect.TypeOf(val))
            return
    }

    return
}

func (ptvmarray TVMArray) nativeCopyTo (data C.native_voidp, datalen int) (err error){
    ret := C._TVMArrayCopyToBytes(C.native_voidp(ptvmarray.nativeCPtr()), data, C.int(datalen))

    if ret != 0 {
        err = errors.New(getTVMLastError())
    }

   return
}

// AsSlice returns the unitptr of for the data inside TVMArray.
//
// returns the slice of array inside TVMArray and err of any.
// TOD: Use reflections for better handling
func (ptvmarray TVMArray) AsSlice() (retVal interface{}, err error) {
    shape := ptvmarray.GetShape()
    size := int64(1)

    for ii := range shape {
        size *= shape[ii]
    }

    var data C.native_voidp
    var datalen int

    dtype := C._DLTensorGetDType(C.uintptr_t(ptvmarray))

    switch ptvmarray.GetDType() {
        case "int32":
            sliceVal := make([]int32, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "int64":
            sliceVal := make([]int64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint32":
            sliceVal := make([]uint32, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "uint64":
            sliceVal := make([]uint64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "float32":
            sliceVal := make([]float32, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        case "float64":
            sliceVal := make([]float64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeCopyTo(data, datalen)
            retVal = sliceVal
        default:
            err = fmt.Errorf("Given type not supported : %v\n", ptvmarray.GetDType())
            return
    }

    return
}

// GetNdim returns the number of dimentions in TVMArray
func (ptvmarray TVMArray) GetNdim() (retVal int32) {
    retVal = (int32)(C._DLTensorGetNdim(C.uintptr_t(ptvmarray)))
    return
}

// GetShape returns the number of dimentions in TVMArray
func (ptvmarray TVMArray) GetShape() (retVal []int64) {
    shapeArr :=  C._DLTensorGetShape(C.uintptr_t(ptvmarray))
    ndim := ptvmarray.GetNdim()

    shapeSlice := (*[1<<31] int64)(unsafe.Pointer(shapeArr))[:ndim:ndim]
    retVal = make([]int64, ndim)
    copy(retVal, shapeSlice)

    return
}

// GetDType returns the number of dimentions in TVMArray
func (ptvmarray TVMArray) GetDType() (retVal string) {
    ret := C._DLTensorGetDType(C.uintptr_t(ptvmarray))
    retVal, _ = dtypeFromTVMType(*(*pTVMType)(unsafe.Pointer(&ret)))
    return
}

// GetCtx returns the number of dimentions in TVMArray
func (ptvmarray TVMArray) GetCtx() (retVal Context) {
    ret := C._DLTensorGetCtx(C.uintptr_t(ptvmarray))
    retVal = *(*Context)(unsafe.Pointer(&ret))
    return
}

// nativeTVMArrayAlloc is used to allocate TVMArray from given attributes.
//
// `shape` is int64 slice holding shape of the TVMArray to be created.
//
// `ndim` is the rank of the TVMArray to be created.
//
// `dtypeCode`, `dtypeBits` and `dtypeLanes` describe the data type in TVMArray.
//
// `deviceType` indicates the device on whose memory the TVMArray to allocated.
//
// `deviceID` indicates device index if multiple devices of same type present.
//
// return argument holding native pointer to newly created TVMArray and error is any.
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
// `shape` is int64 slice holding shape of the TVMArray
//
// `args` is variadic args for
//
//        `args[0]` is string for data type. Default value is 'float32'
//
//        `args[1]` is Context. Default value is '{KDLCPU, 0}'
//
// returns pointer to TVMArray on successful execution and error if any.
func Empty(shape []int64, args ...interface{}) (tvmArray *TVMArray, err error) {
    typeName := "float32"
    ctx := Context{KDLCPU, 0}

    if len(args) > 0 {
        typeName = args[0].(string)
    }

    tvmType, err := dtypeToTVMType(typeName)

    if err != nil {
        return
    }

    if len(args) > 1 {
        ctx = args[1].(Context)
    }

    ndim := int32(len(shape))

    newArray, err := nativeTVMArrayAlloc(shape, ndim, int32(tvmType.code),
                                    int32(tvmType.bits), int32(tvmType.lanes),
                                    ctx.DeviceType, ctx.DeviceID)

    if err != nil {
        return
    }

    handle := new(TVMArray)
    *handle = TVMArray(newArray)

    finalizer := func (ahandle *TVMArray) {
        nativeTVMArrayFree(*ahandle)
        ahandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    tvmArray = handle
    return
}

// nativeTVMArrayFree is used to release the TVMArray.
//
// `ptvmarray` is the TVMArray handle.
//
// `ret` indicates the status of this api execution.
func nativeTVMArrayFree(ptvmarray TVMArray) (retVal int32) {
    retVal = (int32)(C._TVMArrayFree(C.native_voidp(ptvmarray.nativeCPtr())))
    return
}
