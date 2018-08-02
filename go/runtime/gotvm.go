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
import "unsafe"
import "encoding/binary"
import "fmt"

// Variables from this package are the enums exported from TVM.
// All enums are wrapped here as golang require package exports to be started with a upper case.

// GoTVMVersion is gotvm package version information.
var GoTVMVersion            = "0.1"
// DLPackVersion is the dlpack version of tvm runtime.
var DLPackVersion int       = int(C.DLPACK_VERSION)
// TVMVersion is the TVM runtime version.
var TVMVersion              = _TVMVersion()
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

func _TVMVersion() string {
    version := C._TVM_VERSION()
    fmt.Printf("Welcome to gotvm\n")
    return goStringFromNative(*(*string)(unsafe.Pointer(&version)))
}

// Native string map for go string
type nativeGoString struct { p uintptr; n int32 }

func goStringFromNative (s string) string {
  p := *(*nativeGoString)(unsafe.Pointer(&s))
  r := string((*[0x7fffffff]byte)(unsafe.Pointer(p.p))[:p.n])
  C._native_free(unsafe.Pointer(p.p))
  return r
}

// TVMValue Typemap for union exposed by TVM runtime API.
//
// gotvm maps it to a uintptr and then dynamically allocates memory by NewTVMValue method.
type TVMValue uintptr

// NativeCPtr return the unitptr corresponding to TVMValue type.
func (tvmval TVMValue) NativeCPtr() uintptr {
    return (uintptr)(tvmval)
}

// SetVInt64 initializes the TVMValue object with given int64 value.
//
// `val` is the int64 value to initialize the TVMValue
func (tvmval TVMValue) SetVInt64(val int64) {
	C._TVMValueSetInt64(C.uintptr_t(tvmval), C.native_long_long(val))
}


// GetVInt64 returns the int64 value inside the TVMValue.
func (tvmval TVMValue) GetVInt64() int64 {
	return (int64)(C._TVMValueGetInt64(C.uintptr_t(tvmval)))
}

// SetVFloat64 initializes the TVMValue object with given float64 value.
//
// `val` is the float64 value to initialize the TVMValue.
func (tvmval TVMValue) SetVFloat64(val float64) {
	C._TVMValueSetFloat64(C.uintptr_t(tvmval), C.double(val))
}

// GetVFloat64 returns the float64 value inside TVMValue.
func (tvmval TVMValue) GetVFloat64() float64 {
	return (float64)(C._TVMValueGetFloat64(C.uintptr_t(tvmval)))
}

// SetVHandle initializes the handle inside the TVMValue.
//
// Can be used to store any uintptr type object like
// module handle, function handle and any object's NativeCPtr.
//
// `val` is the uintptr type of given handle.
func (tvmval TVMValue) SetVHandle(val uintptr) {
	C._TVMValueSetHandle(C.uintptr_t(tvmval), C.uintptr_t(val))
}

// GetVHandle returns the uintptr handle 
func (tvmval TVMValue) GetVHandle() uintptr {
	return (uintptr)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// SetVStr intializes the TVMValue with given golang string object.
//
// Native wrapper allocate memory to store the golang string which need to be cleaned
// by callint UnSetVStr.
//
// `val` is the golang string object used to initialize the TVMValue.
func (tvmval TVMValue) SetVStr(val string) {
	C._TVMValueSetStr(C.uintptr_t(tvmval), *(*C._gostring_)(unsafe.Pointer(&val)))
}


// GetVStr returns the golang string for the native string inside TVMValue.
func (tvmval TVMValue) GetVStr() string {
	str := C._TVMValueGetStr(C.uintptr_t(tvmval))
    return goStringFromNative(*(*string)(unsafe.Pointer(&str)))
}

// UnSetVStr release the memory allocated in SetVStr
func (tvmval TVMValue) UnSetVStr() {
	C._TVMValueUnSetStr(C.uintptr_t(tvmval))
}

// SetVAHandle is used to set TVMArray handle in TVMValue.
//
// Application can call the SetVHandle with NativeCPtr instead too.
// This is a wrapper to accept TVMArray directly.
func (tvmval TVMValue) SetVAHandle(ptvmarray TVMArray) {
    tvmval.SetVHandle(ptvmarray.NativeCPtr())
}

// GetVAHandle is used to get TVMArray handle in TVMValue.
func (tvmval TVMValue) GetVAHandle() TVMArray {
	return (TVMArray)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// SetVMHandle is used to set TVMModule handle in TVMValue.
//
// Application can call the SetVHandle with NativeCPtr instead too.
// This is a wrapper to accept TVMModule directly.
func (tvmval TVMValue) SetVMHandle(tvmmodule TVMModule) {
    tvmval.SetVHandle(tvmmodule.NativeCPtr())
}

// GetVMHandle is used to get TVMModule handle in TVMValue.
func (tvmval TVMValue) GetVMHandle() TVMModule {
	return (TVMModule)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// SetVFHandle is used to set TVMFunction handle in TVMValue.
//
// Application can call the SetVHandle with NativeCPtr instead.
// This is a wrapper to accept TVMFunction directly.
func (tvmval TVMValue) SetVFHandle(tvmfunction TVMFunction) {
    tvmval.SetVHandle(tvmfunction.NativeCPtr())
}

// GetVFHandle is used to get TVMFunction handle in TVMValue.
func (tvmval TVMValue) GetVFHandle() TVMFunction {
	return (TVMFunction)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// SetVBHandle is used to set TVMByteArray handle in TVMValue.
//
// Application can call the SetVHandle with NativeCPtr instead.
// This is a wrapper to accept TVMByteArray directly.
func (tvmval TVMValue) SetVBHandle(tbytearray TVMByteArray) {
    tvmval.SetVHandle(tbytearray.NativeCPtr())
}

// GetVBHandle is used to get TVMByteArray handle in TVMValue.
func (tvmval TVMValue) GetVBHandle() TVMByteArray {
	return (TVMByteArray)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// NewTVMValue initialize the TVMValue native object.
//
// Before calling any setter or getter on any uderlaying objects of TVMValue
// it should be initialized by thi API.
func NewTVMValue() (TVMValue) {
	return (TVMValue(C._NewTVMValue()))
}

// Delete free the native TVMValue object which is allocated in NewTVMValue.
//
// Delete doesn't free any objects initialized by setter.
// It's application responsibility to free it exclusively.
func (tvmval TVMValue) Delete() {
	C._DeleteTVMValue(C.uintptr_t(tvmval.NativeCPtr()))
}

// TVMArray type in golang hold pointer for the TVMArray object from dlpack.
//
// TVMArray initialization happen through TVMArrayAlloc
type TVMArray uintptr

// NativeCPtr returns type freed uintptr for the TVMArray.
func (ptvmarray TVMArray) NativeCPtr() uintptr {
    return (uintptr)(ptvmarray)
}

// GetData returns the unitptr of for the data inside TVMArray.
func (ptvmarray TVMArray) GetData() uintptr {
	return (uintptr)(C._DLTensorGetData(C.uintptr_t(ptvmarray)))
}

// TVMByteArray type wraps the TVMByteArray of C runtime API.
// 
// This can be used to hold raw data like params of a model.
type TVMByteArray uintptr


// NativeCPtr returns the type freed unitptr for TVMByteArray.
func (tbytearray TVMByteArray) NativeCPtr() uintptr {
	return (uintptr)(tbytearray)
}

// SetData is used to intialize TVMByteArray from a golang string object.
//
// This method initialize both data and data size of the underlaying object.
// This function handles freeing old data object if any before allocating new.
//
// `val` is the golang string object from which the TVMByteArray is initialized.
func (tbytearray TVMByteArray) SetData(val string) {
	C._TVMByteArraySetData(C.uintptr_t(tbytearray), *(*C._gostring_)(unsafe.Pointer(&val)))
}

// GetData returns the golang string corresponding to the TVMByteArray.
func (tbytearray TVMByteArray) GetData() string {
	val := C._TVMByteArrayGetData(C.uintptr_t(tbytearray))
	return goStringFromNative(*(*string)(unsafe.Pointer(&val)))
}

// NewTVMByteArray initilizes the native TVMByteArray object.
func NewTVMByteArray() TVMByteArray {
	return TVMByteArray(C._NewTVMByteArray())
}

// Delete releases the allocated native object of TVMByteArray.
//
// This Delete handles freeing of underlaying native data object too.
func (tbytearray TVMByteArray) Delete() {
	C._DeleteTVMByteArray(C.uintptr_t(tbytearray.NativeCPtr()))
}

// TVMModule type in golang hold pointer for the TVMModule handle.
//
// TVMModule initialization happen through TVMModLoadFromFile api in TVM runtime.
type TVMModule uintptr

// NativeCPtr returns type freed uintptr for the TVMModule.
func (tvmmodule TVMModule) NativeCPtr() uintptr {
    return (uintptr)(tvmmodule)
}

// TVMFunction type in golang hold pointer for the TVMFunction handle.
//
// TVMFunction initialization happen through TVMModGetFunction or TVMFuncGetGlobal.
type TVMFunction uintptr

// NativeCPtr returns type freed uintptr for the TVMFunction.
func (tvmfunction TVMFunction) NativeCPtr() uintptr {
    return (uintptr)(tvmfunction)
}

// TVMFuncListGlobalNames is used to query global callable packed function names from TVM.
//
// `names` return argument which holds golang slice of strings for all the global function names.
//
// `ret` indicates the status of this api execution.
func TVMFuncListGlobalNames(names *[]string) (ret int32) {
    var str string

    ret = (int32)(C._TVMFuncListGlobalNames(C.native_voidp(&str)))

    str = goStringFromNative(*(*string)(unsafe.Pointer(&str)))
    bin := binary.LittleEndian
    size := bin.Uint64([]byte(str[:8]))
    str = str[8:]
    r := make([]string, size)
    for i := range r {
        len := bin.Uint64([]byte(str[:8]))
        str = str[8:]
        r[i] = str[:len]
        str = str[len:]
    }
    *names = r

    return ret
}

// TVMGetLastError returns the detailed error string for any api called in TVM runtime.
//
// This is useful when any api returns non zero value.
//
// Returns golang string for the corresponding native error message.
func TVMGetLastError() string {
    errStr := C._TVMGetLastError()
    return goStringFromNative(*(*string)(unsafe.Pointer(&errStr)))
}

// TVMModLoadFromFile loads the given module in TVM runtime.
//
// `modpath` is the path to tvm module.
//
// `modtype` is the module type from ["dll", "dylib", "dso", "so"]
//
// `modp` is the return argument which is handle to loaded module TVMModule.
//
// `ret` indicates the status of this api execution.
func TVMModLoadFromFile(modpath string, modtype string, modp *TVMModule) int32 {
    return (int32)(C._TVMModLoadFromFile(*(*C._gostring_)(unsafe.Pointer(&modpath)),
                                         *(*C._gostring_)(unsafe.Pointer(&modtype)),
                                         C.native_voidp(modp)))
}

// TVMFuncGetGlobal is to get handle to the given global function name.
//
// `funcname` is the name of global packed function.
//
// `funp` is the return argument holding function handle on success.
//
// `ret` indicates the status of this api execution.
func TVMFuncGetGlobal(funcname string, funp *TVMFunction) int32 {
    return (int32)(C._TVMFuncGetGlobal(*(*C._gostring_)(unsafe.Pointer(&funcname)),
                                       C.native_voidp(funp)))
}

// TVMArrayAlloc is used to allocate TVMArray from given attributes.
//
// `shape` is int64 slice holding shape of the TVMArray to be created.
//
// `ndim` is the rank of the TVMArray to be created.
//
// `dtypeCode`, `dtype_bits` and `dtype_lanes` describe the data type in TVMArray.
//
// `deviceType` indicates the device on whose memory the TVMArray to allocated.
//
// `deviceID` indicates device index if multiple devices of same type present.
//
// `pTvmArray` return argument holding newly allocated TVMArray.
//
// `ret` indicates the status of this api execution.
func TVMArrayAlloc(shape []int64, ndim int32,
                   dtypeCode int32, dtypeBits int32, dtypeLanes int32,
                   deviceType int32, deviceID int32, pTvmArray *TVMArray) int32 {
    var ret int32

    var newTvmArray uintptr

    ret = (int32)(C._TVMArrayAlloc(C.native_voidp(&(shape[0])), C.int(ndim),
                                   C.int(dtypeCode), C.int(dtypeBits), C.int(dtypeLanes),
                                   C.int(deviceType), C.int(deviceID), C.native_voidp(&newTvmArray)))

    *pTvmArray = TVMArray(newTvmArray)

    return ret
}

// TVMArrayFree is used to release the TVMArray.
//
// `ptvmarray` is the TVMArray handle.
//
// `ret` indicates the status of this api execution.
func TVMArrayFree(ptvmarray TVMArray) int32 {
    return (int32)(C._TVMArrayFree(C.native_voidp(ptvmarray.NativeCPtr())))
}

// TVMModGetFunction returns the function pointer from the imported module.
//
// `modp` is the handle for the module TVMModule.
//
// `funcname` is the name of the function in module modp.
//
// `queryImports` indicates to query the imported modules of this module.
//
// `funp` is return argument which is a handle to packed function TVMFunction.
//
// `ret` indicates the status of this api execution.
func TVMModGetFunction(modp TVMModule, funcname string, queryImports int32, funp *TVMFunction) int32 {
    return (int32)(C._TVMModGetFunction(C.uintptr_t(modp), *(*C._gostring_)(unsafe.Pointer(&funcname)),
                                        C.int(queryImports), C.native_voidp(funp)))
}

// TVMModFree free the module handle allocated in TVM runtime.
//
// `modp` is the Module handle to be freed.
func TVMModFree(modp TVMModule) int32 {
    return (int32) (C.TVMModFree(C.TVMModuleHandle(modp.NativeCPtr())))
}

// TVMFuncFree free the function handle allocated in TVM runtime.
//
// `funp` is the Function handle to be freed.
func TVMFuncFree(funp TVMFunction) int32 {
    return (int32) (C.TVMFuncFree(C.TVMFunctionHandle(funp.NativeCPtr())))
}

// TVMFuncCall executes the function with given arguments
//
// `funp` TVMFunction handle to the packed function.
//
// `argValues` is the slice of TVMValue which are arguments to the packed function.
//
// `typeCodes` is the argument type codes corresponding to arg_values.
//
// `numArgs` indicates number of arguments in arg_values.
//
// `retValues` is return argument which is slice of return values from the packed function.
//
// `retTypeCode` is alice of int32 holding type codes for ret_valModule
//
// `ret` indicates the status of this api execution.
func TVMFuncCall(funp TVMFunction, argValues []TVMValue, typeCodes *int32, numArgs int32,
                 retValues []TVMValue, retTypeCode *int32) int32 {
    nargValues := C._TVMValueNativeAllocate(C.int(int32(len(argValues))))

    for ii := range argValues {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(nargValues)),
                             C.native_voidp(unsafe.Pointer(argValues[ii].NativeCPtr())),
                             C.int(int32(ii)))
    }

    nretValues := C._TVMValueNativeAllocate(C.int(int32(len(retValues))))

    for ii := range retValues {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(nretValues)),
                             C.native_voidp(unsafe.Pointer(retValues[ii].NativeCPtr())),
                             C.int(int32(ii)))
    }

	result := (int32)(C._TVMFuncCall(C.uintptr_t(funp),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nargValues))),
                                     C.native_voidp(typeCodes), C.int(numArgs),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nretValues))),
                                     C.native_voidp(retTypeCode)))

    for ii := range argValues {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(argValues[ii].NativeCPtr())),
                             C.native_voidp(unsafe.Pointer(nargValues)),
                             C.int(int32(ii)))
    }

    C._TVMValueNativeFree(C.native_voidp(unsafe.Pointer(nargValues)))


    for ii := range retValues {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(retValues[ii].NativeCPtr())),
                             C.native_voidp(unsafe.Pointer(nretValues)),
                             C.int(int32(ii)))
    }

    C._TVMValueNativeFree(C.native_voidp(unsafe.Pointer(nretValues)))

	return result
}
