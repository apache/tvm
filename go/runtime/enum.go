/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for all enums imported from tvm runtime interface.
 * \file enum.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "fmt"
    "unsafe"
)

// Variables from this package are the enums exported from TVM.
// All enums are wrapped here as golang require package exports to be started with a upper case.

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
