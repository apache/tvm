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

// Value Typemap for union exposed by TVM runtime API.
//
// gotvm maps it to a uintptr and then dynamically allocates memory by newTVMValue method.
type Value uintptr

// AsInt64 returns the int64 value inside the Value.
func (tvmval Value)  AsInt64() (retVal int64) {
    retVal = tvmval.getVInt64()

    return
}

// AsFloat64 returns the Float64 value inside the Value.
func (tvmval Value)  AsFloat64() (retVal float64) {
    retVal = tvmval.getVFloat64()

    return
}

// AsModule returns the Module inside the Value.
func (tvmval Value)  AsModule() (retVal *Module) {
    finalizerModule := func(mhandle *Module) {
        nativeTVMModFree(*mhandle)
        mhandle = nil
    }

    mhandle := new(Module)
    *mhandle = tvmval.getVMHandle()
    runtime.SetFinalizer(mhandle, finalizerModule)
    retVal = mhandle

    return
}

// AsFunction returns the Function inside the Value.
func (tvmval Value)  AsFunction() (retVal *Function) {
    finalizerFunction := func(fhandle *Function) {
        nativeTVMFuncFree(*fhandle)
        fhandle = nil
    }

    fhandle := new(Function)
    *fhandle = tvmval.getVFHandle()
    runtime.SetFinalizer(fhandle, finalizerFunction)
    retVal = fhandle

    return
}

// AsStr returns the golang string slice in the Value.
//
// Note: Calling this function automativally release the underlaying native memory.
// Hence repeated calls to this may lead to segmentation faults.
func (tvmval Value) AsStr() (retVal string) {
    str := tvmval.getVStr()
    tvmval.unSetVStr()
    retVal = str

    return
}

// nativeCPtr return the unitptr corresponding to Value type.
func (tvmval Value) nativeCPtr() (ret uintptr) {
    ret = (uintptr)(tvmval)
    return
}

// copyFrom copies the tvmval from other Value object.
func (tvmval Value) copyFrom(fromval *Value) () {
    C._TVMValueCopyFrom(C.uintptr_t(tvmval), C.uintptr_t(*fromval))
    return
}

// setVInt64 initializes the Value object with given int64 value.
//
// `val` is the int64 value to initialize the Value
func (tvmval Value) setVInt64(val int64) {
	C._TVMValueSetInt64(C.uintptr_t(tvmval), C.native_long_long(val))
}


// getVInt64 returns the int64 value inside the Value.
func (tvmval Value) getVInt64() (retVal int64) {
	retVal = (int64)(C._TVMValueGetInt64(C.uintptr_t(tvmval)))
    return
}

// setVFloat64 initializes the Value object with given float64 value.
//
// `val` is the float64 value to initialize the Value.
func (tvmval Value) setVFloat64(val float64) {
	C._TVMValueSetFloat64(C.uintptr_t(tvmval), C.double(val))
}

// getVFloat64 returns the float64 value inside Value.
func (tvmval Value) getVFloat64() (retVal float64) {
	retVal = (float64)(C._TVMValueGetFloat64(C.uintptr_t(tvmval)))
    return
}

// setVHandle initializes the handle inside the Value.
//
// Can be used to store any uintptr type object like
// module handle, function handle and any object's nativeCPtr.
//
// `val` is the uintptr type of given handle.
func (tvmval Value) setVHandle(val uintptr) {
	C._TVMValueSetHandle(C.uintptr_t(tvmval), C.uintptr_t(val))
}

// getVHandle returns the uintptr handle
func (tvmval Value) getVHandle() (retVal uintptr) {
	retVal = (uintptr)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setVStr intializes the Value with given golang string object.
//
// Native wrapper allocate memory to store the golang string which need to be cleaned
// by callint unSetVStr.
//
// `val` is the golang string object used to initialize the Value.
func (tvmval Value) setVStr(val string) {
	C._TVMValueSetStr(C.uintptr_t(tvmval), *(*C._gostring_)(unsafe.Pointer(&val)))
}


// getVStr returns the golang string for the native string inside Value.
func (tvmval Value) getVStr() (retVal string) {
	str := C._TVMValueGetStr(C.uintptr_t(tvmval))
    retVal = goStringFromNative(*(*string)(unsafe.Pointer(&str)))
    return
}

// unSetVStr release the memory allocated in setVStr
func (tvmval Value) unSetVStr() {
	C._TVMValueUnSetStr(C.uintptr_t(tvmval))
}

// clearStr clars native allocated memory for string
func (tvmval Value)clearVStr(val interface{}) {
    switch val.(type) {
        case string:
            tvmval.unSetVStr()
    }
}

// setVAHandle is used to set Array handle in Value.
//
// Application can call the setVHandle with nativeCPtr instead too.
// This is a wrapper to accept Array directly.
func (tvmval Value) setVAHandle(ptvmarray Array) {
    tvmval.setVHandle(ptvmarray.nativeCPtr())
}

// getVAHandle is used to get Array handle in Value.
func (tvmval Value) getVAHandle() (retVal Array) {
	retVal = (Array)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setVMHandle is used to set Module handle in Value.
//
// Application can call the setVHandle with nativeCPtr instead too.
// This is a wrapper to accept Module directly.
func (tvmval Value) setVMHandle(tvmmodule Module) {
    tvmval.setVHandle(tvmmodule.nativeCPtr())
}

// getVMHandle is used to get Module handle in Value.
func (tvmval Value) getVMHandle() (retVal Module) {
	retVal = (Module)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setVFHandle is used to set Function handle in Value.
//
// Application can call the setVHandle with nativeCPtr instead.
// This is a wrapper to accept Function directly.
func (tvmval Value) setVFHandle(tvmfunction Function) {
    tvmval.setVHandle(tvmfunction.nativeCPtr())
}

// getVFHandle is used to get Function handle in Value.
func (tvmval Value) getVFHandle() (retVal Function) {
	retVal = (Function)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setVBHandle is used to set ByteArray handle in Value.
//
// Application can call the setVHandle with nativeCPtr instead.
// This is a wrapper to accept ByteArray directly.
func (tvmval Value) setVBHandle(tbytearray ByteArray) {
    tvmval.setVHandle(tbytearray.nativeCPtr())
}

// getVBHandle is used to get ByteArray handle in Value.
func (tvmval Value) getVBHandle() (retVal ByteArray) {
	retVal = (ByteArray)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
    return
}

// setValue is used to set the given value in Value.
//
// `val` is value of types accepted by Value container or native union.
func (tvmval Value) setValue(val interface{}) (retVal int32, err error) {
    retVal = KNull
    switch val.(type) {
        case string:
            tvmval.setVStr(val.(string))
            retVal = KStr
        case int64:
            tvmval.setVInt64(val.(int64))
            retVal = KDLInt
        case int:
            tvmval.setVInt64(int64(val.(int)))
            retVal = KDLInt
        case float64:
            tvmval.setVFloat64(val.(float64))
            retVal = KDLFloat
        case Module:
            tvmval.setVMHandle(val.(Module))
            retVal = KModuleHandle
        case *Module:
            tvmval.setVMHandle(*(val.(*Module)))
            retVal = KModuleHandle
        case Function:
            tvmval.setVFHandle(val.(Function))
            retVal = KFuncHandle
        case ByteArray:
            tvmval.setVBHandle(val.(ByteArray))
            retVal = KBytes
        case *ByteArray:
            tvmval.setVBHandle(*(val.(*ByteArray)))
            retVal = KBytes
        case Array:
            tvmval.setVAHandle(val.(Array))
            retVal = KArrayHandle
        case *Array:
            tvmval.setVAHandle(*(val.(*Array)))
            retVal = KArrayHandle
        case func (args ...Value) (interface{}, error):
            fhandle, apierr := ConvertFunction(val)
            if apierr != nil {
                err = fmt.Errorf("Given value Type not defined for Value: %v : %T\n", val, val);
                return
            }
            tvmval.setVFHandle(fhandle)
            retVal = KFuncHandle
        case *Value:
            tvmval.copyFrom(val.(*Value))
            // TODO: Hope to see dtype emneding into TVMValue for proper casting.
            retVal = KDLInt
        case Value:
            fromval := val.(Value)
            tvmval.copyFrom(&fromval)
            // TODO: Hope to see dtype emneding into TVMValue for proper casting.
            retVal = KDLInt
        default:
            err = fmt.Errorf("Given value Type not defined for Value: %v : %T\n", val, val);
    }
    return
}

// getValue is used to get the given from Value container or union.
//
// `tvmtype` is types accepted by Value container or native union.
func (tvmval Value) getValue(tvmtype int32) (retVal interface{}, err error) {
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
            err = fmt.Errorf("Cannot get requested value type from given Value: %v\n", tvmtype);
    }

    return
}

// newTVMValue initialize the TVMValue native object.
//
// Before calling any setter or getter on any uderlaying objects of Value
// it should be initialized by thi API.
func newTVMValue() (retVal Value) {
	retVal = (Value(C._NewTVMValue()))
    return
}

// deleteTVMValue free the native Value object which is allocated in newTVMValue.
func (tvmval Value) deleteTVMValue() {
	C._DeleteTVMValue(C.uintptr_t(tvmval.nativeCPtr()))
}
