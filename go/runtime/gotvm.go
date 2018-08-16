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
    "fmt"
)

// GoTVMVersion is gotvm package version information.
var GoTVMVersion            = "0.1"
// DLPackVersion is the dlpack version of tvm runtime.
var DLPackVersion int       = int(C.DLPACK_VERSION)
// TVMVersion is the TVM runtime version.
var TVMVersion              = getTVMVersion()


func getTVMVersion() (retStr string) {
    version := C._TVM_VERSION()
    fmt.Printf("Welcome to gotvm\n")
    retStr = goStringFromNative(*(*string)(unsafe.Pointer(&version)))
    return
}
