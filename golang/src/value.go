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
 * \brief gotvm package source for TVMValue interface
 * \file value.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "fmt"
    "runtime"
    "unsafe"
)

// KHandle is golang type code for TVM enum kTVMOpaqueHandle.
var KHandle                 = int32(C.kTVMOpaqueHandle)
// KNull is golang type code for TVM kTVMNullptr.
var KNull                   = int32(C.kTVMNullptr)
// KTVMType is golang type code for TVM kTVMDataType.
var KTVMType                = int32(C.kTVMDataType)
// KDLDevice is golang type code for TVM kDLDevice.
var KDLDevice               = int32(C.kDLDevice)
// KArrayHandle is golang type code for TVM kTVMDLTensorHandle.
var KArrayHandle            = int32(C.kTVMDLTensorHandle)
// KObjectHandle is golang type code for TVM kTVMObjectHandle.
var KObjectHandle             = int32(C.kTVMObjectHandle)
// KModuleHandle is gonag type code for TVM kTVMModuleHandle.
var KModuleHandle           = int32(C.kTVMModuleHandle)
// KFuncHandle is gonalg type code for TVM kTVMPackedFuncHandle.
var KFuncHandle             = int32(C.kTVMPackedFuncHandle)
// KStr is golang type code for TVM kTVMStr.
var KStr                    = int32(C.kTVMStr)
// KBytes is golang type code for TVM kTVMBytes.
var KBytes                  = int32(C.kTVMBytes)
// KNDArrayContainer is golang typecode for kTVMNDArrayHandle.
var KNDArrayContainer       = int32(C.kTVMNDArrayHandle)
// KExtBegin is golang enum corresponding to TVM kTVMExtBegin.
var KExtBegin               = int32(C.kTVMExtBegin)
// KNNVMFirst is golang enum corresponding to TVM kNNVMFirst.
var KNNVMFirst              = int32(C.kTVMNNVMFirst)
// KNNVMLast is golang enum corresponding to TVM kNNVMLast.
var KNNVMLast               = int32(C.kTVMNNVMLast)
// KExtReserveEnd is golang enum corresponding to TVM kExtReserveEnd.
var KExtReserveEnd          = int32(C.kTVMExtReserveEnd)
// KExtEnd is golang enum corresponding to TVM kExtEnd.
var KExtEnd                 = int32(C.kTVMExtEnd)
// KDLInt is golang type code for TVM kDLInt.
var KDLInt                  = int32(C.kDLInt)
// KDLUInt is golang type code for TVM kDLUInt.
var KDLUInt                 = int32(C.kDLUInt)
// KDLFloat is golang type code for TVM kDLFloat.
var KDLFloat                = int32(C.kDLFloat)

// Value Typemap for union exposed by TVM runtime API.
//
// gotvm maps it to a uintptr and then dynamically allocates memory by newTVMValue method.
type Value struct {
    nptr  uintptr
    dtype int32
    isLocal bool
}

// AsInt64 returns the int64 value inside the Value.
func (tvmval *Value)  AsInt64() (retVal int64) {
    retVal = tvmval.getVInt64()
    return
}

// AsFloat64 returns the Float64 value inside the Value.
func (tvmval *Value)  AsFloat64() (retVal float64) {
    retVal = tvmval.getVFloat64()
    return
}

// AsModule returns the Module inside the Value.
func (tvmval *Value)  AsModule() (retVal *Module) {
    mhandle := tvmval.getVMHandle()
    retVal = &mhandle
    return
}

// AsFunction returns the Function inside the Value.
func (tvmval *Value)  AsFunction() (retVal *Function) {
    fhandle := tvmval.getVFHandle()
    retVal = &fhandle

    return
}

// AsBytes returns the byte slice value inside the Value.
func (tvmval *Value)  AsBytes() (retVal []byte) {
    retVal = tvmval.getVBHandle().getData()
    return
}

// AsStr returns the golang string in the Value.
func (tvmval *Value) AsStr() (retVal string) {
    str := tvmval.getVStr()
    retVal = str
    return
}

// nativeCPtr return the unitptr corresponding to Value type.
func (tvmval *Value) nativeCPtr() (ret uintptr) {
    ret = (uintptr)(tvmval.nptr)
    return
}

// moveFrom copies the tvmval from other Value object.
func (tvmval *Value) moveFrom(fromval *Value) () {
    C.memcpy(unsafe.Pointer(tvmval.nativeCPtr()),
             unsafe.Pointer(fromval.nativeCPtr()),
             C.sizeof_TVMValue)

    // Move the dtype too.
    tvmval.dtype = fromval.dtype
    fromval.dtype = KNull
    return
}

// setVInt64 initializes the Value object with given int64 value.
//
// `val` is the int64 value to initialize the Value
func (tvmval *Value) setVInt64(val int64) {
    valp := (*C.int64_t)(unsafe.Pointer(tvmval.nativeCPtr()))
    *valp = C.int64_t(val)
    tvmval.dtype = KDLInt
    return
}


// getVInt64 returns the int64 value inside the Value.
func (tvmval *Value) getVInt64() (retVal int64) {
    valp := (*C.int64_t)(unsafe.Pointer(tvmval.nativeCPtr()))
    retVal = int64(*valp)
    return
}

// setVFloat64 initializes the Value object with given float64 value.
//
// `val` is the float64 value to initialize the Value.
func (tvmval *Value) setVFloat64(val float64) {
    valp := (*C.double)(unsafe.Pointer(tvmval.nativeCPtr()))
    *valp = C.double(val)
    tvmval.dtype = KDLFloat
    return
}

// getVFloat64 returns the float64 value inside Value.
func (tvmval *Value) getVFloat64() (retVal float64) {
    valp := (*C.double)(unsafe.Pointer(tvmval.nativeCPtr()))
    retVal = float64(*valp)
    return
}

// setVHandle initializes the handle inside the Value.
//
// Can be used to store any uintptr type object like
// module handle, function handle and any object's nativeCPtr.
//
// `val` is the uintptr type of given handle.
func (tvmval *Value) setVHandle(val uintptr) {
    valp := (**C.void)(unsafe.Pointer(tvmval.nativeCPtr()))
    *valp = (*C.void)(unsafe.Pointer(val))
}

// getVHandle returns the uintptr handle
func (tvmval *Value) getVHandle() (retVal uintptr) {
    valp := (**C.void)(unsafe.Pointer(tvmval.nativeCPtr()))
    retVal = uintptr(unsafe.Pointer(*valp))
    return
}

// setVStr intializes the Value with given golang string object.
//
// `val` is the golang string object used to initialize the Value.
func (tvmval *Value) setVStr(val string) {
    valp := (**C.char)(unsafe.Pointer(tvmval.nativeCPtr()))
    *valp = C.CString(val)
    tvmval.dtype = KStr
    return
}


// getVStr returns the golang string for the native string inside Value.
func (tvmval *Value) getVStr() (retVal string) {
    valp := (**C.char)(unsafe.Pointer(tvmval.nativeCPtr()))
    retVal = C.GoString(*valp)
    return
}

// unSetVStr release the memory allocated in setVStr
func (tvmval *Value) unSetVStr() {
    valp := (**C.char)(unsafe.Pointer(tvmval.nativeCPtr()))
	C.free(unsafe.Pointer(*valp))
    tvmval.dtype = KNull
}

// setVAHandle is used to set Array handle in Value.
//
// Application can call the setVHandle with nativeCPtr instead too.
// This is a wrapper to accept Array directly.
func (tvmval *Value) setVAHandle(ptvmarray Array) {
    tvmval.setVHandle(ptvmarray.nativeCPtr())
    tvmval.dtype = KArrayHandle
    return
}

// getVAHandle is used to get Array handle in Value.
func (tvmval *Value) getVAHandle() (retVal Array) {
	retVal = (Array)(tvmval.getVHandle())
    return
}

// setVMHandle is used to set Module handle in Value.
//
// Application can call the setVHandle with nativeCPtr instead too.
// This is a wrapper to accept Module directly.
func (tvmval *Value) setVMHandle(tvmmodule Module) {
    tvmval.setVHandle(tvmmodule.nativeCPtr())
    tvmval.dtype = KModuleHandle
    return
}

// getVMHandle is used to get Module handle in Value.
func (tvmval *Value) getVMHandle() (retVal Module) {
	retVal = (Module)(tvmval.getVHandle())
    return
}

// setVFHandle is used to set Function handle in Value.
//
// Application can call the setVHandle with nativeCPtr instead.
// This is a wrapper to accept Function directly.
func (tvmval *Value) setVFHandle(tvmfunction Function) {
    tvmval.setVHandle(tvmfunction.nativeCPtr())
    tvmval.dtype = KFuncHandle
    return
}

// getVFHandle is used to get Function handle in Value.
func (tvmval *Value) getVFHandle() (retVal Function) {
	retVal = (Function)(tvmval.getVHandle())
    return
}

// setVBHandle is used to set ByteArray handle in Value.
//
// Application can call the setVHandle with nativeCPtr instead.
// This is a wrapper to accept ByteArray directly.
func (tvmval *Value) setVBHandle(tbytearray ByteArray) {
    tvmval.setVHandle(tbytearray.nativeCPtr())
    tvmval.dtype = KBytes
    return
}

// getVBHandle is used to get ByteArray handle in Value.
func (tvmval *Value) getVBHandle() (retVal ByteArray) {
	retVal = (ByteArray)(tvmval.getVHandle())
    return
}

// setValue is used to set the given value in Value.
//
// `val` is value of types accepted by Value container or native union.
func (tvmval *Value) setValue(val interface{}) (retVal int32, err error) {
    retVal = KNull
    switch val.(type) {
        case string:
            tvmval.setVStr(val.(string))
        case uint8:
            tvmval.setVInt64(int64(val.(uint8)))
        case uint16:
            tvmval.setVInt64(int64(val.(uint16)))
        case uint32:
            tvmval.setVInt64(int64(val.(uint32)))
        case uint64:
            tvmval.setVInt64(int64(val.(uint64)))
        case int:
            tvmval.setVInt64(int64(val.(int)))
        case int8:
            tvmval.setVInt64(int64(val.(int8)))
        case int16:
            tvmval.setVInt64(int64(val.(int16)))
        case int32:
            tvmval.setVInt64(int64(val.(int32)))
        case int64:
            tvmval.setVInt64(val.(int64))
        case float32:
            tvmval.setVFloat64(float64(val.(float32)))
        case float64:
            tvmval.setVFloat64(val.(float64))
        case *Module:
            tvmval.setVMHandle(*(val.(*Module)))
        case *Function:
            tvmval.setVFHandle(*(val.(*Function)))
        case *ByteArray:
            tvmval.setVBHandle(*(val.(*ByteArray)))
        case []byte:
            barray := newByteArray(val.([]byte))
            tvmval.setVBHandle(barray)
        case *Array:
            tvmval.setVAHandle(*(val.(*Array)))
        case func (args ...*Value) (interface{}, error):
            fhandle, apierr := ConvertFunction(val)
            if apierr != nil {
                err = fmt.Errorf("Given value Type not defined for Value: %v : %T", val, val);
                return
            }
            tvmval.setVFHandle(*fhandle)

            // Clear the finalizer as we don't need to control it anymore.
            runtime.SetFinalizer(fhandle, nil)
        case *Value:
            tvmval.moveFrom(val.(*Value))
        case Value:
            fromval := val.(Value)
            tvmval.moveFrom(&fromval)
        default:
            err = fmt.Errorf("Given value Type not defined for Value: %v : %T", val, val);
    }
    retVal = tvmval.dtype
    return
}

// newTVMValue initialize the TVMValue native object.
//
// This is intended to use as intermediate type between native and golang types.
// Allocated from FuncCall or Callback to handle conversions.
func newTVMValue() (retVal *Value) {
    handle := new(Value)

    handle.nptr = (uintptr(C.malloc(C.sizeof_TVMValue)))
    handle.dtype = KNull
    handle.isLocal = true
    finalizer := func(vhandle *Value) {
        vhandle.deleteTVMValue()
        vhandle = nil
    }
    runtime.SetFinalizer(handle, finalizer)
    retVal = handle
    return
}

// deleteTVMValue free the native Value object which is allocated in newTVMValue.
func (tvmval Value) deleteTVMValue() {
    if tvmval.isLocal == true {
        if tvmval.dtype == KStr {
            tvmval.unSetVStr()
        }
        if tvmval.dtype == KBytes {
            tvmval.getVBHandle().deleteTVMByteArray()
        }
    }

	C.free(unsafe.Pointer(tvmval.nativeCPtr()))
}
