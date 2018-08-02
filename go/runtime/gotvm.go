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
import "errors"

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

// nativeCPtr return the unitptr corresponding to TVMValue type.
func (tvmval TVMValue) nativeCPtr() uintptr {
    return (uintptr)(tvmval)
}

// setVInt64 initializes the TVMValue object with given int64 value.
//
// `val` is the int64 value to initialize the TVMValue
func (tvmval TVMValue) setVInt64(val int64) {
	C._TVMValueSetInt64(C.uintptr_t(tvmval), C.native_long_long(val))
}


// getVInt64 returns the int64 value inside the TVMValue.
func (tvmval TVMValue) getVInt64() int64 {
	return (int64)(C._TVMValueGetInt64(C.uintptr_t(tvmval)))
}

// setVFloat64 initializes the TVMValue object with given float64 value.
//
// `val` is the float64 value to initialize the TVMValue.
func (tvmval TVMValue) setVFloat64(val float64) {
	C._TVMValueSetFloat64(C.uintptr_t(tvmval), C.double(val))
}

// getVFloat64 returns the float64 value inside TVMValue.
func (tvmval TVMValue) getVFloat64() float64 {
	return (float64)(C._TVMValueGetFloat64(C.uintptr_t(tvmval)))
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
func (tvmval TVMValue) getVHandle() uintptr {
	return (uintptr)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
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
func (tvmval TVMValue) getVStr() string {
	str := C._TVMValueGetStr(C.uintptr_t(tvmval))
    return goStringFromNative(*(*string)(unsafe.Pointer(&str)))
}

// unSetVStr release the memory allocated in setVStr
func (tvmval TVMValue) unSetVStr() {
	C._TVMValueUnSetStr(C.uintptr_t(tvmval))
}

// setVAHandle is used to set TVMArray handle in TVMValue.
//
// Application can call the setVHandle with nativeCPtr instead too.
// This is a wrapper to accept TVMArray directly.
func (tvmval TVMValue) setVAHandle(ptvmarray TVMArray) {
    tvmval.setVHandle(ptvmarray.nativeCPtr())
}

// getVAHandle is used to get TVMArray handle in TVMValue.
func (tvmval TVMValue) getVAHandle() TVMArray {
	return (TVMArray)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// setVMHandle is used to set TVMModule handle in TVMValue.
//
// Application can call the setVHandle with nativeCPtr instead too.
// This is a wrapper to accept TVMModule directly.
func (tvmval TVMValue) setVMHandle(tvmmodule TVMModule) {
    tvmval.setVHandle(tvmmodule.nativeCPtr())
}

// getVMHandle is used to get TVMModule handle in TVMValue.
func (tvmval TVMValue) getVMHandle() TVMModule {
	return (TVMModule)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// setVFHandle is used to set TVMFunction handle in TVMValue.
//
// Application can call the setVHandle with nativeCPtr instead.
// This is a wrapper to accept TVMFunction directly.
func (tvmval TVMValue) setVFHandle(tvmfunction TVMFunction) {
    tvmval.setVHandle(tvmfunction.nativeCPtr())
}

// getVFHandle is used to get TVMFunction handle in TVMValue.
func (tvmval TVMValue) getVFHandle() TVMFunction {
	return (TVMFunction)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// setVBHandle is used to set TVMByteArray handle in TVMValue.
//
// Application can call the setVHandle with nativeCPtr instead.
// This is a wrapper to accept TVMByteArray directly.
func (tvmval TVMValue) setVBHandle(tbytearray TVMByteArray) {
    tvmval.setVHandle(tbytearray.nativeCPtr())
}

// getVBHandle is used to get TVMByteArray handle in TVMValue.
func (tvmval TVMValue) getVBHandle() TVMByteArray {
	return (TVMByteArray)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

// SetValue is used to set the given value in TVMValue.
//
// `val` is value of types accepted by TVMValue container or native union.
func (tvmval TVMValue) SetValue(val interface{}) (int32, error) {
    switch val.(type) {
        case string:
            tvmval.setVStr(val.(string))
            return KStr, nil
        case int64:
            tvmval.setVInt64(val.(int64))
            return KDLInt, nil
        case float64:
            tvmval.setVFloat64(val.(float64))
            return KDLFloat, nil
        case TVMModule:
            tvmval.setVMHandle(val.(TVMModule))
            return KModuleHandle, nil
        case TVMFunction:
            tvmval.setVFHandle(val.(TVMFunction))
            return KFuncHandle, nil
        case TVMByteArray:
            tvmval.setVBHandle(val.(TVMByteArray))
            return KBytes, nil
        case TVMArray:
            tvmval.setVAHandle(val.(TVMArray))
            return KArrayHandle, nil
        default:
            return KNull, errors.New("Type not defined for TVMValue");
    }
    return KNull, nil
}

// GetValue is used to get the given from TVMValue container or union.
//
// `tvmtype` is types accepted by TVMValue container or native union.
func (tvmval TVMValue) GetValue(tvmtype int32) interface{} {
    switch tvmtype {
        case KDLInt:
            return tvmval.getVInt64()
        case KDLFloat:
            return tvmval.getVFloat64()
        case KStr:
            return tvmval.getVStr()
        case KModuleHandle:
            return tvmval.getVMHandle()
        case KFuncHandle:
            return tvmval.getVFHandle()
        case KBytes:
            return tvmval.getVBHandle()
        case KArrayHandle:
            return tvmval.getVAHandle()
    }
    return nil
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
	C._DeleteTVMValue(C.uintptr_t(tvmval.nativeCPtr()))
}

// TVMArray type in golang hold pointer for the TVMArray object from dlpack.
//
// TVMArray initialization happen through TVMArrayAlloc
type TVMArray uintptr

// nativeCPtr returns type freed uintptr for the TVMArray.
func (ptvmarray TVMArray) nativeCPtr() uintptr {
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


// nativeCPtr returns the type freed unitptr for TVMByteArray.
func (tbytearray TVMByteArray) nativeCPtr() uintptr {
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
	C._DeleteTVMByteArray(C.uintptr_t(tbytearray.nativeCPtr()))
}

// TVMModule type in golang hold pointer for the TVMModule handle.
//
// TVMModule initialization happen through TVMModLoadFromFile api in TVM runtime.
type TVMModule uintptr

// nativeCPtr returns type freed uintptr for the TVMModule.
func (tvmmodule TVMModule) nativeCPtr() uintptr {
    return (uintptr)(tvmmodule)
}

// TVMFunction type in golang hold pointer for the TVMFunction handle.
//
// TVMFunction initialization happen through TVMModGetFunction or TVMFuncGetGlobal.
type TVMFunction uintptr

// nativeCPtr returns type freed uintptr for the TVMFunction.
func (tvmfunction TVMFunction) nativeCPtr() uintptr {
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
// `dtypeCode`, `dtypeBits` and `dtypeLanes` describe the data type in TVMArray.
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
    return (int32)(C._TVMArrayFree(C.native_voidp(ptvmarray.nativeCPtr())))
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
    return (int32) (C.TVMModFree(C.TVMModuleHandle(modp.nativeCPtr())))
}

// TVMFuncFree free the function handle allocated in TVM runtime.
//
// `funp` is the Function handle to be freed.
func TVMFuncFree(funp TVMFunction) int32 {
    return (int32) (C.TVMFuncFree(C.TVMFunctionHandle(funp.nativeCPtr())))
}

// TVMFuncCall executes the function with given arguments
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
// `result` indicates the status of this api execution.
func TVMFuncCall(funp TVMFunction, argValues []TVMValue, typeCodes []int32,
                 retValues []TVMValue, retTypeCode *int32) int32 {
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

	return result
}


func clearVStr(val interface{}, tvmval TVMValue) {
    switch val.(type) {
        case string:
            tvmval.unSetVStr()
    }
}

// TVMFunctionExec executes the given function of given module with given parameters.
//
// `module` TVMModule handle to the module. Can be `nil` for Global functions.
//
// `funcname` string representing the function name.
//
// `args` variadic arguments of gilang of different types which can be embed on TVMValue.
//
// returns TVMValue, retTypeCode, err (if any). Application should Delete the TVMValue if valid.
func TVMFunctionExec(module interface{}, funcname string, args ...interface{}) (TVMValue, int32, error) {
    argsIn := make([]TVMValue, len(args))

    var typeCodes []int32

    if len(args) != 0 {
        typeCodes = make([]int32, len(args))
    } else {
        typeCodes = make([]int32, 1)
    }

    var err error

    for ii := range args {
        argsIn[ii] = NewTVMValue()
        if typeCodes[ii], err = argsIn[ii].SetValue(args[ii]); err != nil {
            return TVMValue(C.NULL), 0, err
        }
    }

    defer func() {
        for ii := range argsIn {
            clearVStr(args[ii], argsIn[ii])
            argsIn[ii].Delete()
        }
    }()

    var funp TVMFunction

    switch module.(type) {
        case TVMModule:
            if TVMModGetFunction(module.(TVMModule), funcname, 1, &funp) != 0 {
                return TVMValue(C.NULL), 0, errors.New(TVMGetLastError())
            }
        default:
            if TVMFuncGetGlobal(funcname, &funp) != 0 {
                return TVMValue(C.NULL), 0, errors.New(TVMGetLastError())
            }
    }
    defer TVMFuncFree(funp)

    argsOut := []TVMValue{NewTVMValue()}
    retTypeCode := KNull

    if TVMFuncCall(funp, argsIn, typeCodes, argsOut, &retTypeCode) != 0 {
        argsOut[0].Delete()
        return TVMValue(C.NULL), 0, errors.New(TVMGetLastError())
    }

    if retTypeCode != KNull {
        return argsOut[0], retTypeCode, nil
    }

    argsOut[0].Delete()

    return TVMValue(C.NULL), retTypeCode, nil
}
