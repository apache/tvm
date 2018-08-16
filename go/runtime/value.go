/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for TVMValue interface
 * \file value.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
    "fmt"
    "runtime"
)

// KHandle is golang type code for TVM enum kHandle.
var KHandle int32           = int32(C.kHandle)
// KNull is golang type code for TVM kNull.
var KNull int32             = int32(C.kNull)
// KTVMType is golang type code for TVM kTVMType.
var KTVMType int32          = int32(C.kTVMType)
// KTVMContext is golang type code for TVM kTVMContext.
var KTVMContext int32       = int32(C.kTVMContext)
// KArrayHandle is golang type code for TVM kArrayHandle.
var KArrayHandle int32      = int32(C.kArrayHandle)
// KNodeHandle is golang type code for TVM kNodeHandle.
var KNodeHandle int32       = int32(C.kNodeHandle)
// KModuleHandle is gonag type code for TVM kModuleHandle.
var KModuleHandle int32     = int32(C.kModuleHandle)
// KFuncHandle is gonalg type code for TVM kFuncHandle.
var KFuncHandle int32       = int32(C.kFuncHandle)
// KStr is golang type code for TVM kStr.
var KStr int32              = int32(C.kStr)
// KBytes is golang type code for TVM kBytes.
var KBytes int32            = int32(C.kBytes)
// KNDArrayContainer is golang typecode for kNDArrayContainer.
var KNDArrayContainer int32 = int32(C.kNDArrayContainer)
// KExtBegin is golang enum corresponding to TVM kExtBegin.
var KExtBegin int32         = int32(C.kExtBegin)
// KNNVMFirst is golang enum corresponding to TVM kNNVMFirst.
var KNNVMFirst int32        = int32(C.kNNVMFirst)
// KNNVMLast is golang enum corresponding to TVM kNNVMLast.
var KNNVMLast int32         = int32(C.kNNVMLast)
// KExtReserveEnd is golang enum corresponding to TVM kExtReserveEnd.
var KExtReserveEnd int32    = int32(C.kExtReserveEnd)
// KExtEnd is golang enum corresponding to TVM kExtEnd.
var KExtEnd int32           = int32(C.kExtEnd)
// KDLInt is golang type code for TVM kDLInt.
var KDLInt int32            = int32(C.kDLInt)
// KDLUInt is golang type code for TVM kDLUInt.
var KDLUInt int32           = int32(C.kDLUInt)
// KDLFloat is golang type code for TVM kDLFloat.
var KDLFloat int32          = int32(C.kDLFloat)

// TVMValue Typemap for union exposed by TVM runtime API.
//
// gotvm maps it to a uintptr and then dynamically allocates memory by newTVMValue method.
type TVMValue uintptr

// AsInt64 returns the int64 value inside the TVMValue.
func (tvmval TVMValue)  AsInt64() (retVal int64) {
    retVal = tvmval.getVInt64()

    return
}

// AsFloat64 returns the Float64 value inside the TVMValue.
func (tvmval TVMValue)  AsFloat64() (retVal float64) {
    retVal = tvmval.getVFloat64()

    return
}

// AsModule returns the TVMModule inside the TVMValue.
func (tvmval TVMValue)  AsModule() (retVal *TVMModule) {
    finalizerModule := func(mhandle *TVMModule) {
        fmt.Printf("finalizerModule called \n")
        nativeTVMModFree(*mhandle)
        mhandle = nil
    }

    mhandle := new(TVMModule)
    *mhandle = tvmval.getVMHandle()
    runtime.SetFinalizer(mhandle, finalizerModule)
    retVal = mhandle

    return
}

// AsFunction returns the TVMFunction inside the TVMValue.
func (tvmval TVMValue)  AsFunction() (retVal *TVMFunction) {
    finalizerFunction := func(fhandle *TVMFunction) {
        fmt.Printf("finalizerFunction called \n")
        nativeTVMFuncFree(*fhandle)
        fhandle = nil
    }

    fhandle := new(TVMFunction)
    *fhandle = tvmval.getVFHandle()
    runtime.SetFinalizer(fhandle, finalizerFunction)
    retVal = fhandle

    return
}

// AsStr returns the golang string slice in the TVMValue.
//
// Note: Calling this function automativally release the underlaying native memory.
// Hence repeated calls to this may lead to segmentation faults.
func (tvmval TVMValue) AsStr() (retVal string) {
    str := tvmval.getVStr()
    tvmval.unSetVStr()
    retVal = str

    return
}

// nativeCPtr return the unitptr corresponding to TVMValue type.
func (tvmval TVMValue) nativeCPtr() (ret uintptr) {
    ret = (uintptr)(tvmval)
    return
}

// setVInt64 initializes the TVMValue object with given int64 value.
//
// `val` is the int64 value to initialize the TVMValue
func (tvmval TVMValue) setVInt64(val int64) {
	C._TVMValueSetInt64(C.uintptr_t(tvmval), C.native_long_long(val))
}


// getVInt64 returns the int64 value inside the TVMValue.
func (tvmval TVMValue) getVInt64() (retVal int64) {
	retVal = (int64)(C._TVMValueGetInt64(C.uintptr_t(tvmval)))
    return
}

// setVFloat64 initializes the TVMValue object with given float64 value.
//
// `val` is the float64 value to initialize the TVMValue.
func (tvmval TVMValue) setVFloat64(val float64) {
	C._TVMValueSetFloat64(C.uintptr_t(tvmval), C.double(val))
}

// getVFloat64 returns the float64 value inside TVMValue.
func (tvmval TVMValue) getVFloat64() (retVal float64) {
	retVal = (float64)(C._TVMValueGetFloat64(C.uintptr_t(tvmval)))
    return
}

// setVHandle initializes the handle inside the TVMValue.
//
// Can be used to store any uintptr type object like
// module handle, function handle and any object's nativeCPtr.
//
// `val` is the uintptr type of given handle.
func (tvmval TVMValue) setVHandle(val uintptr) {
	C._TVMValueSetHandle(C.uintptr_t(tvmval), C.uintptr_t(val))
}

// getVHandle returns the uintptr handle
func (tvmval TVMValue) getVHandle() (retVal uintptr) {
	retVal = (uintptr)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setVStr intializes the TVMValue with given golang string object.
//
// Native wrapper allocate memory to store the golang string which need to be cleaned
// by callint unSetVStr.
//
// `val` is the golang string object used to initialize the TVMValue.
func (tvmval TVMValue) setVStr(val string) {
	C._TVMValueSetStr(C.uintptr_t(tvmval), *(*C._gostring_)(unsafe.Pointer(&val)))
}


// getVStr returns the golang string for the native string inside TVMValue.
func (tvmval TVMValue) getVStr() (retVal string) {
	str := C._TVMValueGetStr(C.uintptr_t(tvmval))
    retVal = goStringFromNative(*(*string)(unsafe.Pointer(&str)))
    return
}

// unSetVStr release the memory allocated in setVStr
func (tvmval TVMValue) unSetVStr() {
	C._TVMValueUnSetStr(C.uintptr_t(tvmval))
}

// clearStr clars native allocated memory for string
func (tvmval TVMValue)clearVStr(val interface{}) {
    switch val.(type) {
        case string:
            tvmval.unSetVStr()
    }
}

// setVAHandle is used to set TVMArray handle in TVMValue.
//
// Application can call the setVHandle with nativeCPtr instead too.
// This is a wrapper to accept TVMArray directly.
func (tvmval TVMValue) setVAHandle(ptvmarray TVMArray) {
    tvmval.setVHandle(ptvmarray.nativeCPtr())
}

// getVAHandle is used to get TVMArray handle in TVMValue.
func (tvmval TVMValue) getVAHandle() (retVal TVMArray) {
	retVal = (TVMArray)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setVMHandle is used to set TVMModule handle in TVMValue.
//
// Application can call the setVHandle with nativeCPtr instead too.
// This is a wrapper to accept TVMModule directly.
func (tvmval TVMValue) setVMHandle(tvmmodule TVMModule) {
    tvmval.setVHandle(tvmmodule.nativeCPtr())
}

// getVMHandle is used to get TVMModule handle in TVMValue.
func (tvmval TVMValue) getVMHandle() (retVal TVMModule) {
	retVal = (TVMModule)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setVFHandle is used to set TVMFunction handle in TVMValue.
//
// Application can call the setVHandle with nativeCPtr instead.
// This is a wrapper to accept TVMFunction directly.
func (tvmval TVMValue) setVFHandle(tvmfunction TVMFunction) {
    tvmval.setVHandle(tvmfunction.nativeCPtr())
}

// getVFHandle is used to get TVMFunction handle in TVMValue.
func (tvmval TVMValue) getVFHandle() (retVal TVMFunction) {
	retVal = (TVMFunction)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setVBHandle is used to set TVMByteArray handle in TVMValue.
//
// Application can call the setVHandle with nativeCPtr instead.
// This is a wrapper to accept TVMByteArray directly.
func (tvmval TVMValue) setVBHandle(tbytearray TVMByteArray) {
    tvmval.setVHandle(tbytearray.nativeCPtr())
}

// getVBHandle is used to get TVMByteArray handle in TVMValue.
func (tvmval TVMValue) getVBHandle() (retVal TVMByteArray) {
	retVal = (TVMByteArray)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setValue is used to set the given value in TVMValue.
//
// `val` is value of types accepted by TVMValue container or native union.
func (tvmval TVMValue) setValue(val interface{}) (retVal int32, err error) {
    retVal = KNull
    switch val.(type) {
        case string:
            tvmval.setVStr(val.(string))
            retVal = KStr
        case int64:
            tvmval.setVInt64(val.(int64))
            retVal = KDLInt
        case float64:
            tvmval.setVFloat64(val.(float64))
            retVal = KDLFloat
        case TVMModule:
            tvmval.setVMHandle(val.(TVMModule))
            retVal = KModuleHandle
        case *TVMModule:
            tvmval.setVMHandle(*(val.(*TVMModule)))
            retVal = KModuleHandle
        case TVMFunction:
            tvmval.setVFHandle(val.(TVMFunction))
            retVal = KFuncHandle
        case TVMByteArray:
            tvmval.setVBHandle(val.(TVMByteArray))
            retVal = KBytes
        case *TVMByteArray:
            tvmval.setVBHandle(*(val.(*TVMByteArray)))
            retVal = KBytes
        case TVMArray:
            tvmval.setVAHandle(val.(TVMArray))
            retVal = KArrayHandle
        case *TVMArray:
            tvmval.setVAHandle(*(val.(*TVMArray)))
            retVal = KArrayHandle
        default:
            err = fmt.Errorf("Given value Type not defined for TVMValue: %v : %T\n", val, val);
    }
    return
}

/*
// getFinalizedValue is used to get the given from TVMValue container or union.
//
// `tvmtype` is types accepted by TVMValue container or native union.
func (tvmval TVMValue) getFinalizedValue(tvmtype int32) (retVal interface{}, err error) {
    switch tvmtype {
        case KDLInt:
            retVal = tvmval.getVInt64()
        case KDLFloat:
            retVal = tvmval.getVFloat64()
        case KStr:
            str := tvmval.getVStr()
            tvmval.unSetVStr()
            retVal = str
        case KModuleHandle:
            retVal = tvmval.getVMHandle()
        case KFuncHandle:
            retVal = tvmval.getVFHandle()
        default:
            err = fmt.Errorf("Cannot get requested value type from given TVMValue: %v\n", tvmtype);
    }

    return
}
*/

// newTVMValue initialize the TVMValue native object.
//
// Before calling any setter or getter on any uderlaying objects of TVMValue
// it should be initialized by thi API.
func newTVMValue() (retVal TVMValue) {
	retVal = (TVMValue(C._NewTVMValue()))
    return
}

// deleteTVMValue free the native TVMValue object which is allocated in newTVMValue.
func (tvmval TVMValue) deleteTVMValue() {
	C._DeleteTVMValue(C.uintptr_t(tvmval.nativeCPtr()))
}
