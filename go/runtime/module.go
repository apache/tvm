/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for TVMModule interface.
 * \file module.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
    "errors"
    "runtime"
)

// Module type in golang hold pointer for the TVMModule handle.
//
// Module initialization happen through TVMModLoadFromFile api in TVM runtime.
type Module uintptr

// nativeCPtr returns type freed uintptr for the Module.
func (tvmmodule Module) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(tvmmodule)
    return
}

// LoadModuleFromFile loads the given module in TVM runtime.
//
// `modpath` is the path to tvm module.
//
// `args` is an optional arguments of ["dll", "dylib", "dso", "so"] with default value "so"
//
// returns pointer to Module and err or if any.
func LoadModuleFromFile(modpath string, args ...interface{}) (retVal *Module, err error) {
    modtype := "so"

    if len(args) > 0 {
       modtype  = args[0].(string)
    }

    var modp uintptr

    ret := (int32)(C._TVMModLoadFromFile(*(*C._gostring_)(unsafe.Pointer(&modpath)),
                                         *(*C._gostring_)(unsafe.Pointer(&modtype)),
                                         C.native_voidp(&modp)))
    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    handle := new(Module)
    *handle = Module(modp)

    finalizer := func(mhandle *Module) {
        nativeTVMModFree(*mhandle)
        mhandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    retVal = handle
    return
}

// nativeTVMModFree free the module handle allocated in TVM runtime.
//
// `modp` is the Module handle to be freed.
func nativeTVMModFree(modp Module) (retVal int32) {
    retVal = (int32) (C.TVMModFree(C.TVMModuleHandle(modp.nativeCPtr())))
    return
}
