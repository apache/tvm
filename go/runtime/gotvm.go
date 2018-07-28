/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package
 * \file gotvm.go
 */

package gotvm

//#include "gotvm.h"
import "C"
import "unsafe"
import "encoding/binary"
import "fmt"

var GOTVM_VERSION string =  "0.1"

// Import enums
var DLPACK_VERSION int      = int(C.DLPACK_VERSION)
var KDLSDAccel int32        = int32(C.kDLSDAccel)
var KDLVulkan int32         = int32(C.kDLVulkan)
var KOpenGL int32           = int32(C.kOpenGL)
var KExtDev int32           = int32(C.kExtDev)
var KHandle int32           = int32(C.kHandle)
var KNull int32             = int32(C.kNull)
var KTVMType int32          = int32(C.kTVMType)
var KTVMContext int32       = int32(C.kTVMContext)
var KArrayHandle int32      = int32(C.kArrayHandle)
var KNodeHandle int32       = int32(C.kNodeHandle)
var KModuleHandle int32     = int32(C.kModuleHandle)
var KFuncHandle int32       = int32(C.kFuncHandle)
var KStr int32              = int32(C.kStr)
var KBytes int32            = int32(C.kBytes)
var KNDArrayContainer int32 = int32(C.kNDArrayContainer)
var KExtBegin int32         = int32(C.kExtBegin)
var KNNVMFirst int32        = int32(C.kNNVMFirst)
var KNNVMLast int32         = int32(C.kNNVMLast)
var KExtReserveEnd int32    = int32(C.kExtReserveEnd)
var KExtEnd int32           = int32(C.kExtEnd)
var KDLCPU int32            = int32(C.kDLCPU)
var KDLGPU int32            = int32(C.kDLGPU)
var KDLCPUPinned int32      = int32(C.kDLCPUPinned)
var KDLOpenCL int32         = int32(C.kDLOpenCL)
var KDLMetal int32          = int32(C.kDLMetal)
var KDLVPI int32            = int32(C.kDLVPI)
var KDLROCM int32           = int32(C.kDLROCM)
var KDLInt int32            = int32(C.kDLInt)
var KDLUInt int32           = int32(C.kDLUInt)
var KDLFloat int32          = int32(C.kDLFloat)

// API - TVM_VERSION
var TVM_VERSION string = _TVM_VERSION()

func _TVM_VERSION() string {
    version := C._TVM_VERSION()
    fmt.Printf("Welcome to gotvm\n")
    return _gostring_from_native(*(*string)(unsafe.Pointer(&version)))
}

// Native string map for go string
type native_gostring struct { p uintptr; n int32 }

func _gostring_from_native (s string) string {
  p := *(*native_gostring)(unsafe.Pointer(&s))
  r := string((*[0x7fffffff]byte)(unsafe.Pointer(p.p))[:p.n])
  C._native_free(unsafe.Pointer(p.p))
  return r
}

// TVMValue
type PTVMValue uintptr

func (p PTVMValue) Nativecptr() uintptr {
    return (uintptr)(p)
}

func (tvmval PTVMValue) SetV_int64(val int64) {
	C._TVMValueSetInt64(C.uintptr_t(tvmval), C.native_long_long(val))
}

func (tvmval PTVMValue) GetV_int64() int64 {
	return (int64)(C._TVMValueGetInt64(C.uintptr_t(tvmval)))
}


func (tvmval PTVMValue) SetV_float64(val float64) {
	C._TVMValueSetFloat64(C.uintptr_t(tvmval), C.double(val))
}

func (tvmval PTVMValue) GetV_float64() float64 {
	return (float64)(C._TVMValueGetFloat64(C.uintptr_t(tvmval)))
}

func (tvmval PTVMValue) SetV_handle(val uintptr) {
	C._TVMValueSetHandle(C.uintptr_t(tvmval), C.uintptr_t(val))
}

func (tvmval PTVMValue) GetV_handle() uintptr {
	return (uintptr)(C._TVMValueGetHandle(C.uintptr_t(tvmval)))
}

func (tvmval PTVMValue) SetV_str(val string) {
	C._TVMValueSetStr(C.uintptr_t(tvmval), *(*C._gostring_)(unsafe.Pointer(&val)))
}

func (tvmval PTVMValue) GetV_str() string {
	str := C._TVMValueGetStr(C.uintptr_t(tvmval))
    return _gostring_from_native(*(*string)(unsafe.Pointer(&str)))
}

func (tvmval PTVMValue) UnSetV_str() {
	C._TVMValueUnSetStr(C.uintptr_t(tvmval))
}

func (tvmval PTVMValue) SetV_aHandle(pdltensor DLTensor) {
    tvmval.SetV_handle(pdltensor.Nativecptr())
}

func NewTVMValue() (TVMValue) {
	return (TVMValue)(PTVMValue(C._NewTVMValue()))
}

func DeleteTVMValue(tvmval TVMValue) {
	C._DeleteTVMValue(C.uintptr_t(tvmval.Nativecptr()))
}

type TVMValue interface {
    Nativecptr() uintptr
    SetV_int64(int64)
    GetV_int64() (int64)
    SetV_float64(float64)
    GetV_float64() (float64)
    SetV_handle(uintptr)
    GetV_handle() (uintptr)
    SetV_str(string)
    GetV_str() (string)
    UnSetV_str()
    SetV_aHandle(DLTensor)
/*  SetV_type(TVMType)
    GetV_type() (TVMType)
    SetV_ctx(TVMContext)
    GetV_ctx() (TVMContext)*/
}

// DLTensor
type PDLTensor uintptr

func (p PDLTensor) Nativecptr() uintptr {
    return (uintptr)(p)
}

func (pdltensor PDLTensor) GetData() uintptr {
	return (uintptr)(C._DLTensorGetData(C.uintptr_t(pdltensor)))
}

func NewDLTensor() (dltensor DLTensor) {
	return (DLTensor)(PDLTensor(C._NewDLTensor()))
}

func DeleteDLTensor(dltensor DLTensor) {
	C._DeleteDLTensor(C.uintptr_t(dltensor.Nativecptr()))
}


type DLTensor interface {
    Nativecptr() uintptr
    GetData() uintptr
/*
    SetData(val uintptr)
    SetCtx(val DLContext)
    GetCtx() DLContext
    SetNdim(val int32)
    GetNdim() int32
    SetDtype(val DLDataType)
    GetDtype() DLDataType
    SetShape(val *int64)
    GetShape() *int64
    SetStrides(val *int64)
    GetStrides() *int64
    SetByte_offset(val uint64)
    GetByte_offset() uint64
*/
}

// TVMByteArray
type PTVMByteArray uintptr

func (p PTVMByteArray) Nativecptr() uintptr {
	return (uintptr)(p)
}

func (tbytearray PTVMByteArray) SetData(val string) {
	C._TVMByteArraySetData(C.uintptr_t(tbytearray), *(*C._gostring_)(unsafe.Pointer(&val)))
}

func (tbytearray PTVMByteArray) GetData() string {
	val := C._TVMByteArrayGetData(C.uintptr_t(tbytearray))
	return _gostring_from_native(*(*string)(unsafe.Pointer(&val)))
}

func (tbytearray PTVMByteArray) SetSize(val int64) {
	C._TVMByteArraySetSize(C.uintptr_t(tbytearray), C.native_long_long(val))
}

func (tbytearray PTVMByteArray) GetSize() int64 {
	return (int64)(C._TVMByteArrayGetSize(C.uintptr_t(tbytearray)))
}

func NewTVMByteArray() TVMByteArray {
	return (TVMByteArray)(PTVMByteArray(C._NewTVMByteArray()))
}

func DeleteTVMByteArray(tbytearray TVMByteArray) {
	C._DeleteTVMByteArray(C.uintptr_t(tbytearray.Nativecptr()))
}

type TVMByteArray interface {
	Nativecptr() uintptr
	SetData(val string)
	GetData() string
	SetSize(val int64)
	GetSize() int64
}

// API - TVMFuncListGlobalNames
func TVMFuncListGlobalNames(names *[]string) (ret int32) {
    var str string

    ret = (int32)(C._TVMFuncListGlobalNames(C.native_voidp(&str)))

    str = _gostring_from_native(*(*string)(unsafe.Pointer(&str)))
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

// API - TVMGetLastError
func TVMGetLastError() string {
    err_str := C._TVMGetLastError()
    return _gostring_from_native(*(*string)(unsafe.Pointer(&err_str)))
}

// API - TVMModLoadFromFile
func TVMModLoadFromFile(modpath string, modtype string, modp *uintptr) int32 {
    return (int32)(C._TVMModLoadFromFile(*(*C._gostring_)(unsafe.Pointer(&modpath)),
                                         *(*C._gostring_)(unsafe.Pointer(&modtype)),
                                         C.native_voidp(modp)))
}

// API - TVMFuncGetGlobal
func TVMFuncGetGlobal(funcname string, funp *uintptr) int32 {
    return (int32)(C._TVMFuncGetGlobal(*(*C._gostring_)(unsafe.Pointer(&funcname)),
                                       C.native_voidp(funp)))
}

// API - TVMArrayAlloc
func TVMArrayAlloc(shape *int64, ndim int32,
                   dtype_code int32, dtype_bits int32, dtype_lanes int32,
                   device_type int32, device_id int32, dltensor *DLTensor) int32 {
    var ret int32

    DeleteDLTensor(*dltensor)
    var newdltensor uintptr

    ret = (int32)(C._TVMArrayAlloc(C.native_voidp(shape), C.int(ndim),
                                   C.int(dtype_code), C.int(dtype_bits), C.int(dtype_lanes),
                                   C.int(device_type), C.int(device_id), C.native_voidp(&newdltensor)))

    *dltensor = (DLTensor)(PDLTensor(newdltensor))

    return ret
}

// API - TVMArrayFree
func TVMArrayFree(pdltensor DLTensor) int32 {
    return (int32)(C._TVMArrayFree(C.native_voidp(pdltensor.Nativecptr())))
}

// API - TVMModGetFunction
func TVMModGetFunction(modp uintptr, funcname string, query_imports int32, funp *uintptr) int32 {
    return (int32)(C._TVMModGetFunction(C.uintptr_t(modp), *(*C._gostring_)(unsafe.Pointer(&funcname)),
                                        C.int(query_imports), C.native_voidp(funp)))
}

// API - TVMFuncCall
func TVMFuncCall(funp uintptr, arg_values []TVMValue, type_codes *int32, num_args int32,
                 ret_values []TVMValue, ret_type_code *int32) int32 {
    narg_values := C._TVMValueNativeAllocate(C.int(int32(len(arg_values))))

    for ii := range arg_values {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(narg_values)),
                             C.native_voidp(unsafe.Pointer(arg_values[ii].Nativecptr())),
                             C.int(int32(ii)))
    }

    nret_values := C._TVMValueNativeAllocate(C.int(int32(len(ret_values))))

    for ii := range ret_values {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(nret_values)),
                             C.native_voidp(unsafe.Pointer(ret_values[ii].Nativecptr())),
                             C.int(int32(ii)))
    }

	result := (int32)(C._TVMFuncCall(C.uintptr_t(funp),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(narg_values))),
                                     C.native_voidp(type_codes), C.int(num_args),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nret_values))),
                                     C.native_voidp(ret_type_code)))

    for ii := range arg_values {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(arg_values[ii].Nativecptr())),
                             C.native_voidp(unsafe.Pointer(narg_values)),
                             C.int(int32(ii)))
    }

    C._TVMValueNativeFree(C.native_voidp(unsafe.Pointer(narg_values)))


    for ii := range ret_values {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(ret_values[ii].Nativecptr())),
                             C.native_voidp(unsafe.Pointer(nret_values)),
                             C.int(int32(ii)))
    }

    C._TVMValueNativeFree(C.native_voidp(unsafe.Pointer(nret_values)))

	return result
}
