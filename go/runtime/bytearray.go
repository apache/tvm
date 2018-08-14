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

// TVMByteArray type wraps the TVMByteArray of C runtime API.
// 
// This can be used to hold raw data like params of a model.
type TVMByteArray uintptr

// nativeCPtr returns the type freed unitptr for TVMByteArray.
func (tbytearray TVMByteArray) nativeCPtr() (retVal uintptr) {
	retVal = (uintptr)(tbytearray)
    return
}

// SetData is used to intialize TVMByteArray from a golang string object.
//
// This method initialize both data and data size of the underlaying object.
// This function handles freeing old data object if any before allocating new.
//
// `val` is the golang string object from which the TVMByteArray is initialized.
func (tbytearray TVMByteArray) setData(val string) {
	C._TVMByteArraySetData(C.uintptr_t(tbytearray), *(*C._gostring_)(unsafe.Pointer(&val)))
}

// GetData returns the golang string corresponding to the TVMByteArray.
func (tbytearray TVMByteArray) GetData() (retVal string) {
	val := C._TVMByteArrayGetData(C.uintptr_t(tbytearray))
	retVal = goStringFromNative(*(*string)(unsafe.Pointer(&val)))
    return
}

// NewTVMByteArray initilizes the native TVMByteArray object with given byte slice
//
//`val` is the golang byte array used to initialize.
//
// returns pointer to newly created TVMByteArray.
func NewTVMByteArray(val []uint8) (retVal *TVMByteArray) {

    handle := new(TVMByteArray)
    *handle = TVMByteArray(C._NewTVMByteArray())

    finalizer := func(ahandle *TVMByteArray) {
        ahandle.deleteTVMByteArray()
        ahandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    (*handle).setData(string(val))

    retVal = handle
    return
}

// deleteTVMByteArray releases the allocated native object of TVMByteArray.
//
// This delete handles freeing of underlaying native data object too.
func (tbytearray TVMByteArray) deleteTVMByteArray() {
	C._DeleteTVMByteArray(C.uintptr_t(tbytearray.nativeCPtr()))
}
