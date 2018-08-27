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
    "runtime"
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
	C._TVMByteArraySetData(C.uintptr_t(tbytearray), *(*C._gostring_)(unsafe.Pointer(&val)))
}

// GetData returns the golang byte slice corresponding to the ByteArray.
func (tbytearray ByteArray) GetData() (retVal []byte) {
	val := C._TVMByteArrayGetData(C.uintptr_t(tbytearray))
	retVal = []byte(goStringFromNative(*(*string)(unsafe.Pointer(&val))))
    return
}

// NewByteArray initilizes the native TVMByteArray object with given byte slice
//
//`val` is the golang byte array used to initialize.
//
// returns pointer to newly created ByteArray.
func NewByteArray(val []byte) (retVal *ByteArray) {

    handle := new(ByteArray)
    *handle = ByteArray(C._NewTVMByteArray())

    finalizer := func(ahandle *ByteArray) {
        ahandle.deleteTVMByteArray()
        ahandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    (*handle).setData(string(val))

    retVal = handle
    return
}

// deleteTVMByteArray releases the allocated native object of ByteArray.
//
// This delete handles freeing of underlaying native data object too.
func (tbytearray ByteArray) deleteTVMByteArray() {
	C._DeleteTVMByteArray(C.uintptr_t(tbytearray.nativeCPtr()))
}
