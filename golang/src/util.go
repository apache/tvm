/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for common utilities
 * \file util.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
)

// Native string map for go string
type nativeGoString struct { p uintptr; n int32 }

func goStringFromNative (s string) (retStr string) {
    p := *(*nativeGoString)(unsafe.Pointer(&s))
    retStr = string((*[0x7fffffff]byte)(unsafe.Pointer(p.p))[:p.n])
    C.free(unsafe.Pointer(p.p))
    return
}
