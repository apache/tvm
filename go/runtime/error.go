/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for error related API interface.
 * \file error.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
)

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
