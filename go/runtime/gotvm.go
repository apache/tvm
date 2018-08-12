/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package
 * \file gotvm.go
 */

// Package gotvm is TVM runtime interface definition for golang.
//
// Application need to import this package to access the c_runtime_api exposed by TVM.
package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
    "encoding/binary"
    "fmt"
    "errors"
    "runtime"
    "reflect"
)

// Variables from this package are the enums exported from TVM.
// All enums are wrapped here as golang require package exports to be started with a upper case.

// GoTVMVersion is gotvm package version information.
var GoTVMVersion            = "0.1"
// DLPackVersion is the dlpack version of tvm runtime.
var DLPackVersion int       = int(C.DLPACK_VERSION)
// TVMVersion is the TVM runtime version.
var TVMVersion              = getTVMVersion()
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
// KDLCPU is golang enum correspond to TVM device type kDLCPU.
var KDLCPU int32            = int32(C.kDLCPU)
// KDLGPU is golang enum correspond to TVM device type kDLGPU.
var KDLGPU int32            = int32(C.kDLGPU)
// KDLCPUPinned is golang enum correspond to TVM device type kDLCPUPinned.
var KDLCPUPinned int32      = int32(C.kDLCPUPinned)
// KDLOpenCL is golang enum correspond to TVM device type kDLOpenCL.
var KDLOpenCL int32         = int32(C.kDLOpenCL)
// KDLMetal is golang enum correspond to TVM device type kDLMetal.
var KDLMetal int32          = int32(C.kDLMetal)
// KDLVPI is golang enum correspond to TVM device type kDLVPI.
var KDLVPI int32            = int32(C.kDLVPI)
// KDLROCM is golang enum correspond to TVM device type kDLROCM.
var KDLROCM int32           = int32(C.kDLROCM)
// KDLSDAccel is golang enum correspond to TVM device type kDLSDAccel.
var KDLSDAccel int32        = int32(C.kDLSDAccel)
// KDLVulkan is golang enum correspond to TVM device type kDLVulkan.
var KDLVulkan int32         = int32(C.kDLVulkan)
// KOpenGL is golang enum correspond to TVM device type kOpenGL.
var KOpenGL int32           = int32(C.kOpenGL)
// KExtDev is golang enum correspond to TVM device type kExtDev.
var KExtDev int32           = int32(C.kExtDev)
// KDLInt is golang type code for TVM kDLInt.
var KDLInt int32            = int32(C.kDLInt)
// KDLUInt is golang type code for TVM kDLUInt.
var KDLUInt int32           = int32(C.kDLUInt)
// KDLFloat is golang type code for TVM kDLFloat.
var KDLFloat int32          = int32(C.kDLFloat)

func getTVMVersion() (retStr string) {
    version := C._TVM_VERSION()
    fmt.Printf("Welcome to gotvm\n")
    retStr = goStringFromNative(*(*string)(unsafe.Pointer(&version)))
    return
}

// Native string map for go string
type nativeGoString struct { p uintptr; n int32 }

func goStringFromNative (s string) (retStr string) {
    p := *(*nativeGoString)(unsafe.Pointer(&s))
    retStr = string((*[0x7fffffff]byte)(unsafe.Pointer(p.p))[:p.n])
    C._native_free(unsafe.Pointer(p.p))
    return
}

// TVMContext dtype corresponding to DLContext
type TVMContext struct {
    DeviceType int32
    DeviceID    int32
}

// pTVMType corresponding to data types.
type pTVMType struct {
    code uint8
    bits uint8
    lanes uint16
}

// TVMValue Typemap for union exposed by TVM runtime API.
//
// gotvm maps it to a uintptr and then dynamically allocates memory by newTVMValue method.
type TVMValue uintptr

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

// getFinalizedValue is used to get the given from TVMValue container or union.
//
// `tvmtype` is types accepted by TVMValue container or native union.
func (tvmval TVMValue) getFinalizedValue(tvmtype int32) (retVal interface{}, err error) {
    finalizerModule := func(mhandle *TVMModule) {
        nativeTVMModFree(*mhandle)
        mhandle = nil
    }

    finalizerFunction := func(fhandle *TVMFunction) {
        nativeTVMFuncFree(*fhandle)
        fhandle = nil
    }

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
            handle := new(TVMModule)
            *handle = tvmval.getVMHandle()
            runtime.SetFinalizer(handle, finalizerModule)
            retVal = handle
        case KFuncHandle:
            handle := new(TVMFunction)
            *handle = tvmval.getVFHandle()
            runtime.SetFinalizer(handle, finalizerFunction)
            retVal = handle
        default:
            err = fmt.Errorf("Cannot get requested value type from given TVMValue: %v\n", tvmtype);
    }

    return
}

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

// TVMArray type in golang hold pointer for the TVMArray object from dlpack.
//
// TVMArray initialization happen through TVMArrayAlloc
type TVMArray uintptr

// nativeCPtr returns type freed uintptr for the TVMArray.
func (ptvmarray TVMArray) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(ptvmarray)
    return
}


func (ptvmarray TVMArray) nativeSetData(data C.native_voidp, datalen int) (err error) {
    ret := C._TVMArrayCopyFromBytes(C.native_voidp(ptvmarray.nativeCPtr()), data, C.int(datalen))

    if ret != 0 {
        err = errors.New(getTVMLastError())
    }

    return
}

// SetData copies given data into TVMArray.
//
// `val` is interface homding a slice of TVMArray data type.
//
// returns err is any.
// TOD: Use reflections for better handling
func (ptvmarray TVMArray) SetData(val interface{}) (err error) {
    var data C.native_voidp
    var datalen int

    dtype := C._DLTensorGetDType(C.uintptr_t(ptvmarray))

    switch val.(type) {
        case []int32:
            sliceVal := val.([]int32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeSetData(data, datalen)
        case []int64:
            sliceVal := val.([]int64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeSetData(data, datalen)
        case []uint32:
            sliceVal := val.([]uint32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeSetData(data, datalen)
        case []uint64:
            sliceVal := val.([]uint64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeSetData(data, datalen)
        case []float32:
            sliceVal := val.([]float32)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeSetData(data, datalen)
        case []float64:
            sliceVal := val.([]float64)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            return ptvmarray.nativeSetData(data, datalen)
        default:
            err = fmt.Errorf("Given type not supported : %v\n", reflect.TypeOf(val))
            return
    }

    return
}

func (ptvmarray TVMArray) nativeGetData (data C.native_voidp, datalen int) (err error){
    ret := C._TVMArrayCopyToBytes(C.native_voidp(ptvmarray.nativeCPtr()), data, C.int(datalen))

    if ret != 0 {
        err = errors.New(getTVMLastError())
    }

   return
}

// GetData returns the unitptr of for the data inside TVMArray.
//
// returns the slice of array inside TVMArray and err of any.
// TOD: Use reflections for better handling
func (ptvmarray TVMArray) GetData() (retVal interface{}, err error) {
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
            err = ptvmarray.nativeGetData(data, datalen)
            retVal = sliceVal
        case "int64":
            sliceVal := make([]int64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeGetData(data, datalen)
            retVal = sliceVal
        case "uint32":
            sliceVal := make([]uint32, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeGetData(data, datalen)
            retVal = sliceVal
        case "uint64":
            sliceVal := make([]uint64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeGetData(data, datalen)
            retVal = sliceVal
        case "float32":
            sliceVal := make([]float32, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeGetData(data, datalen)
            retVal = sliceVal
        case "float64":
            sliceVal := make([]float64, size)
            data = C.native_voidp(&sliceVal[0])
            datalen = len(sliceVal) * int(dtype.bits / 8)
            err = ptvmarray.nativeGetData(data, datalen)
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
func (ptvmarray TVMArray) GetCtx() (retVal TVMContext) {
    ret := C._DLTensorGetCtx(C.uintptr_t(ptvmarray))
    retVal = *(*TVMContext)(unsafe.Pointer(&ret))
    return
}

// TVMByteArray type wraps the TVMByteArray of C runtime API.
// 
// This can be used to hold raw data like params of a model.
type TVMByteArray uintptr


// nativeCPtr returns the type freed unitptr for TVMByteArray.
func (tbytearray TVMByteArray) nativeCPtr() (retVal uintptr) {
	retVal = (uintptr)(tbytearray)
    return
}

// SetData is used to intialize TVMByteArray from a golang string object.
//
// This method initialize both data and data size of the underlaying object.
// This function handles freeing old data object if any before allocating new.
//
// `val` is the golang string object from which the TVMByteArray is initialized.
func (tbytearray TVMByteArray) setData(val string) {
	C._TVMByteArraySetData(C.uintptr_t(tbytearray), *(*C._gostring_)(unsafe.Pointer(&val)))
}

// GetData returns the golang string corresponding to the TVMByteArray.
func (tbytearray TVMByteArray) GetData() (retVal string) {
	val := C._TVMByteArrayGetData(C.uintptr_t(tbytearray))
	retVal = goStringFromNative(*(*string)(unsafe.Pointer(&val)))
    return
}

// NewTVMByteArray initilizes the native TVMByteArray object with given byte slice
//
//`val` is the golang byte array used to initialize.
//
// returns pointer to newly created TVMByteArray.
func NewTVMByteArray(val []uint8) (retVal *TVMByteArray) {

    handle := new(TVMByteArray)
    *handle = TVMByteArray(C._NewTVMByteArray())

    finalizer := func(ahandle *TVMByteArray) {
        ahandle.deleteTVMByteArray()
        ahandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    (*handle).setData(string(val))

    retVal = handle
    return
}

// deleteTVMByteArray releases the allocated native object of TVMByteArray.
//
// This delete handles freeing of underlaying native data object too.
func (tbytearray TVMByteArray) deleteTVMByteArray() {
	C._DeleteTVMByteArray(C.uintptr_t(tbytearray.nativeCPtr()))
}

// TVMModule type in golang hold pointer for the TVMModule handle.
//
// TVMModule initialization happen through TVMModLoadFromFile api in TVM runtime.
type TVMModule uintptr

// nativeCPtr returns type freed uintptr for the TVMModule.
func (tvmmodule TVMModule) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(tvmmodule)
    return
}

// TVMFunction type in golang hold pointer for the TVMFunction handle.
type TVMFunction uintptr

// nativeCPtr returns type freed uintptr for the TVMFunction.
func (tvmfunction TVMFunction) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(tvmfunction)
    return
}

// FuncListGlobalNames is used to query global callable packed function names from TVM.
//
// returns slice of string holding function names and error if any.
func FuncListGlobalNames() (retVal []string, err error) {
    var str string

    ret := (int32)(C._TVMFuncListGlobalNames(C.native_voidp(&str)))

    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    str = goStringFromNative(*(*string)(unsafe.Pointer(&str)))
    bin := binary.LittleEndian
    size := bin.Uint64([]byte(str[:8]))
    str = str[8:]
    retVal = make([]string, size)
    for i := range retVal {
        len := bin.Uint64([]byte(str[:8]))
        str = str[8:]
        retVal[i] = str[:len]
        str = str[len:]
    }

    return
}

// getTVMLastError returns the detailed error string for any api called in TVM runtime.
//
// This is useful when any api returns non zero value.
//
// Returns golang string for the corresponding native error message.
func getTVMLastError() (retVal string) {
    errStr := C._TVMGetLastError()
    retVal = goStringFromNative(*(*string)(unsafe.Pointer(&errStr)))
    return
}

// ModLoadFromFile loads the given module in TVM runtime.
//
// `modpath` is the path to tvm module.
//
// `args` is an optional arguments of ["dll", "dylib", "dso", "so"] with default value "so"
//
// returns pointer to TVMModule and err or if any.
func ModLoadFromFile(modpath string, args ...interface{}) (retVal *TVMModule, err error) {
    modtype := "so"

    if len(args) > 0 {
       modtype  = args[0].(string)
    }

    var modp uintptr

    ret := (int32)(C._TVMModLoadFromFile(*(*C._gostring_)(unsafe.Pointer(&modpath)),
                                         *(*C._gostring_)(unsafe.Pointer(&modtype)),
                                         C.native_voidp(&modp)))
    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    handle := new(TVMModule)
    *handle = TVMModule(modp)

    finalizer := func(mhandle *TVMModule) {
        nativeTVMModFree(*mhandle)
        mhandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    retVal = handle
    return
}

// GetGlobalFunction is to get handle to the given global function name.
//
// `funcname` is the name of global packed function.
//
// returns a function closure with signature
//         func (args ...interface{}) (interface{}, error) and  error if any.
//
// The closure function can be used to call TVMFunction with arguments directly.
//
// Variadic arguments can be any type which can be embed into TVMValue.
func GetGlobalFunction(funcname string) (retVal func (args ...interface{}) (interface{}, error), err error) {
    var funp uintptr

    ret := (int32)(C._TVMFuncGetGlobal(*(*C._gostring_)(unsafe.Pointer(&funcname)),
                                       C.native_voidp(&funp)))

    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    handle := new(TVMFunction)
    *handle = TVMFunction(funp)

    finalizer := func(fhandle *TVMFunction) {
        nativeTVMFuncFree(*fhandle)
        fhandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    funccall := func (args ...interface{}) (interface{}, error) {
        return callNativeFunction(*handle, args)
    }

    retVal = funccall
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

// data type to pTVMType mapping
var dtypeMap = map[string] pTVMType {
    "int32": pTVMType{0, 32, 1},
    "int64": pTVMType{0, 64, 1},
    "uint32": pTVMType{1, 32, 1},
    "uint64": pTVMType{1, 64, 1},
    "float32": pTVMType{2, 32, 1},
    "float64": pTVMType{2, 64, 1},
    "handle": pTVMType{3, 64, 1},
}

// dtypeFromTVMType return the pTVMType corresponding to given dtype
//
// `dtype` string for the given data type.
func dtypeFromTVMType(tvmtype pTVMType) (retVal string, err error) {
    for k, v := range dtypeMap {
        if v.code == tvmtype.code && v.bits == tvmtype.bits && v.lanes == tvmtype.lanes {
            retVal = k
            return
        }
    }

    err = fmt.Errorf("Cannot map TVMType:%v to dtype", tvmtype)
    return
}

// dtypeToTVMType return the pTVMType corresponding to given dtype
//
// `dtype` string for the given data type.
func dtypeToTVMType(args ...interface{}) (tvmtype pTVMType, err error) {
    dtype := args[0].(string)

    lanes := 1
    if len(args) == 2 {
        lanes = args[1].(int)
    }

    for k, v := range dtypeMap {
        if k == dtype {
            tvmtype = v
            tvmtype.lanes = uint16(lanes)
            return
        }
    }

    err = fmt.Errorf("Cannot map dtype:%v to TVMType", dtype)
    return
}

// EmptyArray is used to allocate TVM empty array of given epecification. 
//
// `shape` is int64 slice holding shape of the TVMArray
//
// `args` is variadic args for
//
//        `args[0]` is string for data type. Default value is 'float32'
//
//        `args[1]` is TVMContext. Default value is '{KDLCPU, 0}'
//
// returns pointer to TVMArray on successful execution and error if any.
func EmptyArray(shape []int64, args ...interface{}) (tvmArray *TVMArray, err error) {
    typeName := "float32"
    ctx := TVMContext{KDLCPU, 0}

    if len(args) > 0 {
        typeName = args[0].(string)
    }

    tvmType, err := dtypeToTVMType(typeName)

    if err != nil {
        return
    }

    if len(args) > 1 {
        ctx = args[1].(TVMContext)
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

// callNativeFunction is routine which calls gotvm native wrapper with given arguments.
//
// `handle` is the handle for TVMFunction.
//
// `args` are the variadic arguments to the TVMFunction.
//
// returns the interface for the return value from TVM if any and error if any.
func callNativeFunction(handle TVMFunction, args []interface{}) (retVal interface{}, err error) {
        argsIn := make([]TVMValue, len(args))

        var typeCodes []int32

        if len(args) != 0 {
           typeCodes = make([]int32, len(args))
        } else {
            typeCodes = make([]int32, 1)
        }

        for ii := range args {
            argsIn[ii] = newTVMValue()
            if typeCodes[ii], err = argsIn[ii].setValue(args[ii]); err != nil {
                return
            }
        }

        defer func() {
            for ii := range argsIn {
                argsIn[ii].clearVStr(args[ii])
                argsIn[ii].deleteTVMValue()
            }
        }()

        argsOut := []TVMValue{newTVMValue()}
        retTypeCode := KNull

        defer argsOut[0].deleteTVMValue()

        err = nativeTVMFuncCall(handle, argsIn, typeCodes, argsOut, &retTypeCode)
        if err != nil {
            return
        }

        if retTypeCode != KNull {
            retVal, err = argsOut[0].getFinalizedValue(retTypeCode)
            return
        }

        return
}

// GetFunction returns the function pointer from the module for given function name.
//
// `tvmmodule` is handle for TVMModule
//
// `funcname` function name in module.
//
// `args` variadic args of `queryImport`
//
// returns function closure with signature
//         func (args ...interface{}) (interface{}, error) and error if any.
//
// The closure function can be used to call TVMFunction with arguments directly.
// 
// Variadic arguments can be any type which can be embed into TVMValue.
func (tvmmodule *TVMModule) GetFunction (
      funcname string, args ...interface{}) (
      retVal func (args ...interface{}) (interface{}, error), err error){
    queryImports := int32(1)
    if len(args) > 0 {
        queryImports = int32(args[1].(int))
    }

    var funp uintptr

    ret := (int32)(C._TVMModGetFunction(C.uintptr_t(*tvmmodule),
                                        *(*C._gostring_)(unsafe.Pointer(&funcname)),
                                        C.int(queryImports), C.native_voidp(&funp)))

    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    handle := new(TVMFunction)
    *handle = TVMFunction(funp)

    finalizer := func(fhandle *TVMFunction) {
        nativeTVMFuncFree(*fhandle)
        fhandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    funccall := func (args ...interface{}) (interface{}, error) {
        return callNativeFunction(*handle, args)
    }

    retVal = funccall
    return
}

// nativeTVMModFree free the module handle allocated in TVM runtime.
//
// `modp` is the Module handle to be freed.
func nativeTVMModFree(modp TVMModule) (retVal int32) {
    retVal = (int32) (C.TVMModFree(C.TVMModuleHandle(modp.nativeCPtr())))
    return
}

// nativeTVMFuncFree free the function handle allocated in TVM runtime.
//
// `funp` is the Function handle to be freed.
func nativeTVMFuncFree(funp TVMFunction) (retVal int32) {
    retVal = (int32) (C.TVMFuncFree(C.TVMFunctionHandle(funp.nativeCPtr())))
    return
}

// nativeTVMFuncCall executes the function with given arguments
//
// `funp` TVMFunction handle to the packed function.
//
// `argValues` is the slice of TVMValue which are arguments to the packed function.
//
// `typeCodes` is the alice of argument type codes corresponding to argValues.
//
// `retValues` is return argument which is slice of return values from the packed function.
//
// `retTypeCode` is int32 holding type codes for retValue
//
// Returns err indicating native error if any.
func nativeTVMFuncCall(funp TVMFunction, argValues []TVMValue, typeCodes []int32,
                 retValues []TVMValue, retTypeCode *int32) (err error) {
    numArgs := int32(len(argValues))

    nargValues := C._TVMValueNativeAllocate(C.int(int32(len(argValues))))

    for ii := range argValues {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(nargValues)),
                             C.native_voidp(unsafe.Pointer(argValues[ii].nativeCPtr())),
                             C.int(int32(ii)))
    }

    nretValues := C._TVMValueNativeAllocate(C.int(int32(len(retValues))))

    for ii := range retValues {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(nretValues)),
                             C.native_voidp(unsafe.Pointer(retValues[ii].nativeCPtr())),
                             C.int(int32(ii)))
    }

	result := (int32)(C._TVMFuncCall(C.uintptr_t(funp),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nargValues))),
                                     C.native_voidp(&(typeCodes[0])), C.int(numArgs),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nretValues))),
                                     C.native_voidp(retTypeCode)))

    for ii := range argValues {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(argValues[ii].nativeCPtr())),
                             C.native_voidp(unsafe.Pointer(nargValues)),
                             C.int(int32(ii)))
    }

    C._TVMValueNativeFree(C.native_voidp(unsafe.Pointer(nargValues)))


    for ii := range retValues {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(retValues[ii].nativeCPtr())),
                             C.native_voidp(unsafe.Pointer(nretValues)),
                             C.int(int32(ii)))
    }

    C._TVMValueNativeFree(C.native_voidp(unsafe.Pointer(nretValues)))

    if result != 0 {
	    err = errors.New(getTVMLastError())
    }

    return
}

