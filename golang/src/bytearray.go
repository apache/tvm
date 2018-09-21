/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for TVMByteArray interface.
 * \file bytearray.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
)

// ByteArray type wraps the TVMByteArray of C runtime API.
// 
// This can be used to hold raw data like params of a model.
type ByteArray uintptr

// nativeCPtr returns the type freed unitptr for ByteArray.
func (tbytearray ByteArray) nativeCPtr() (retVal uintptr) {
	retVal = (uintptr)(tbytearray)
    return
}

// SetData is used to intialize ByteArray from a golang string object.
//
// This method initialize both data and data size of the underlaying object.
// This function handles freeing old data object if any before allocating new.
//
// `val` is the golang string object from which the ByteArray is initialized.
func (tbytearray ByteArray) setData(val string) {
	C._TVMByteArraySetData(C.uintptr_t(tbytearray), C.CString(val), C.int(len(val)))
}

// getData returns the golang byte slice corresponding to the ByteArray.
func (tbytearray ByteArray) getData() (retVal []byte) {
	val := C._TVMByteArrayGetData(C.uintptr_t(tbytearray))
	blen := C._TVMByteArrayGetDataLen(C.uintptr_t(tbytearray))
	retVal = C.GoBytes(unsafe.Pointer(val), C.int(blen))
    return
}

// newByteArray initilizes the native TVMByteArray object with given byte slice
//
//`val` is the golang byte array used to initialize.
//
// returns newly created ByteArray.
func newByteArray(val []byte) (retVal ByteArray) {
    handle := ByteArray(C._NewTVMByteArray())
    handle.setData(string(val))
    retVal = handle
    return
}

// deleteTVMByteArray releases the allocated native object of ByteArray.
//
// This delete handles freeing of underlaying native data object too.
func (tbytearray ByteArray) deleteTVMByteArray() {
	C._DeleteTVMByteArray(C.uintptr_t(tbytearray.nativeCPtr()))
}
