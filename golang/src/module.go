/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief gotvm package source for TVMModule interface.
 * \file module.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "errors"
    "runtime"
    "unsafe"
)

// Module type in golang hold pointer for the TVMModule handle.
//
// Module initialization happen through TVMModLoadFromFile api in TVM runtime.
type Module uintptr

// nativeCPtr returns type freed uintptr for the Module.
func (tvmmodule *Module) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(*tvmmodule)
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

    cmodpath := C.CString(modpath)
    cmodtype := C.CString(modtype)

    ret := (int32)(C.TVMModLoadFromFile(cmodpath,
                                        cmodtype,
                                        (*C.TVMModuleHandle)(unsafe.Pointer(&modp))))

    C.free(unsafe.Pointer(cmodpath))
    C.free(unsafe.Pointer(cmodtype))

    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    handle := new(Module)
    *handle = Module(modp)
    finalizer := func(mhandle *Module) {
        nativeTVMModFree(mhandle)
        mhandle = nil
    }
    runtime.SetFinalizer(handle, finalizer)
    retVal = handle
    return
}

// nativeTVMModFree free the module handle allocated in TVM runtime.
//
// `modp` is the Module handle to be freed.
func nativeTVMModFree(modp *Module) (retVal int32) {
    retVal = (int32) (C.TVMModFree(C.TVMModuleHandle(modp.nativeCPtr())))
    return
}

// GetFunction returns the function pointer from the module for given function name.
//
// `tvmmodule` is handle for Module
//
// `funcname` function name in module.
//
// `args` variadic args of `queryImport`
//
// returns function closure with signature
//         func (args ...interface{}) (interface{}, error) and error if any.
//
// The closure function can be used to call Function with arguments directly.
//
// Variadic arguments can be any type which can be embed into Value.
func (tvmmodule *Module) GetFunction (
      funcname string, args ...interface{}) (
      retVal *Function, err error){
    queryImports := int32(1)
    if len(args) > 0 {
        queryImports = int32(args[1].(int))
    }

    var funp uintptr
    cfuncname := C.CString(funcname)
    ret := (int32)(C.TVMModGetFunction((C.TVMModuleHandle)(*tvmmodule),
                                       cfuncname,
                                       C.int(queryImports),
                                       (*C.TVMFunctionHandle)(unsafe.Pointer(&funp))))
    C.free(unsafe.Pointer(cfuncname))

    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    handle := new(Function)
    *handle = Function(funp)
    finalizer := func(fhandle *Function) {
        nativeTVMFuncFree(fhandle)
        fhandle = nil
    }
    runtime.SetFinalizer(handle, finalizer)
    retVal = handle
    return
}
