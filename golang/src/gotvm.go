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

// DLPackVersion is the dlpack version of tvm runtime.
var DLPackVersion           = int(C.DLPACK_VERSION)
// TVMVersion is the TVM runtime version.
var TVMVersion              = getTVMVersion()

func getTVMVersion() (retStr string) {
    retStr = C.GoString(C._TVM_VERSION())
    return
}
