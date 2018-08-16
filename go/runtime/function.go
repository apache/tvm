/*!
 *  Copyright (c) 2018 by Contributors
 * \brief gotvm package source for TVMFunction interface.
 * \file function.go
 */

package gotvm

//#include "gotvm.h"
import "C"

import (
    "unsafe"
    "encoding/binary"
    "errors"
    "runtime"
)

// Function type in golang hold pointer for the TVMFunction handle.
type Function uintptr

// nativeCPtr returns type freed uintptr for the Function.
func (tvmfunction Function) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(tvmfunction)
    return
}

// FuncListGlobalNames is used to query global callable packed function names from TVM.
//
// returns slice of string holding function names and error if any.
func FuncListGlobalNames() (retVal []string, err error) {
    var str string

    ret := (int32)(C._TVMFuncListGlobalNames(C.native_voidp(&str)))

    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    str = goStringFromNative(*(*string)(unsafe.Pointer(&str)))
    bin := binary.LittleEndian
    size := bin.Uint64([]byte(str[:8]))
    str = str[8:]
    retVal = make([]string, size)
    for i := range retVal {
        len := bin.Uint64([]byte(str[:8]))
        str = str[8:]
        retVal[i] = str[:len]
        str = str[len:]
    }

    return
}

// GetGlobalFunction is to get handle to the given global function name.
//
// `funcname` is the name of global packed function.
//
// returns a function closure with signature
//         func (args ...interface{}) (interface{}, error) and  error if any.
//
// The closure function can be used to call Function with arguments directly.
//
// Variadic arguments can be any type which can be embed into Value.
func GetGlobalFunction(funcname string) (retVal func (args ...interface{}) (*Value, error), err error) {
    var funp uintptr

    ret := (int32)(C._TVMFuncGetGlobal(*(*C._gostring_)(unsafe.Pointer(&funcname)),
                                       C.native_voidp(&funp)))

    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    handle := new(Function)
    *handle = Function(funp)

    finalizer := func(fhandle *Function) {
        nativeTVMFuncFree(*fhandle)
        fhandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    funccall := func (args ...interface{}) (*Value, error) {
        return callNativeFunction(*handle, args)
    }

    retVal = funccall
    return
}

// callNativeFunction is routine which calls gotvm native wrapper with given arguments.
//
// `handle` is the handle for Function.
//
// `args` are the variadic arguments to the Function.
//
// returns the interface for the return value from TVM if any and error if any.
func callNativeFunction(handle Function, args []interface{}) (retVal *Value, err error) {
        argsIn := make([]Value, len(args))

        var typeCodes []int32

        if len(args) != 0 {
           typeCodes = make([]int32, len(args))
        } else {
            typeCodes = make([]int32, 1)
        }

        for ii := range args {
            argsIn[ii] = newTVMValue()
            if typeCodes[ii], err = argsIn[ii].setValue(args[ii]); err != nil {
                return
            }
        }

        defer func() {
            for ii := range argsIn {
                argsIn[ii].clearVStr(args[ii])
                argsIn[ii].deleteTVMValue()
            }
        }()

        argsOut := []Value{newTVMValue()}
        retTypeCode := KNull

        finalizerValue := func(vhandle *Value) {
            (*vhandle).deleteTVMValue()
            vhandle = nil
        }

        vhandle := new(Value)
        *vhandle = Value(argsOut[0].nativeCPtr())
        runtime.SetFinalizer(vhandle, finalizerValue)

        err = nativeTVMFuncCall(handle, argsIn, typeCodes, argsOut, &retTypeCode)
        if err != nil {
            return
        }

        if retTypeCode != KNull {
            retVal = vhandle
            return
        }

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
      retVal func (args ...interface{}) (*Value, error), err error){
    queryImports := int32(1)
    if len(args) > 0 {
        queryImports = int32(args[1].(int))
    }

    var funp uintptr

    ret := (int32)(C._TVMModGetFunction(C.uintptr_t(*tvmmodule),
                                        *(*C._gostring_)(unsafe.Pointer(&funcname)),
                                        C.int(queryImports), C.native_voidp(&funp)))

    if ret != 0 {
        err = errors.New(getTVMLastError())
        return
    }

    handle := new(Function)
    *handle = Function(funp)

    finalizer := func(fhandle *Function) {
        nativeTVMFuncFree(*fhandle)
        fhandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    funccall := func (args ...interface{}) (*Value, error) {
        return callNativeFunction(*handle, args)
    }

    retVal = funccall
    return
}

// nativeTVMFuncFree free the function handle allocated in TVM runtime.
//
// `funp` is the Function handle to be freed.
func nativeTVMFuncFree(funp Function) (retVal int32) {
    retVal = (int32) (C.TVMFuncFree(C.TVMFunctionHandle(funp.nativeCPtr())))
    return
}

// nativeTVMFuncCall executes the function with given arguments
//
// `funp` Function handle to the packed function.
//
// `argValues` is the slice of Value which are arguments to the packed function.
//
// `typeCodes` is the alice of argument type codes corresponding to argValues.
//
// `retValues` is return argument which is slice of return values from the packed function.
//
// `retTypeCode` is int32 holding type codes for retValue
//
// Returns err indicating native error if any.
func nativeTVMFuncCall(funp Function, argValues []Value, typeCodes []int32,
                 retValues []Value, retTypeCode *int32) (err error) {
    numArgs := int32(len(argValues))

    nargValues := C._TVMValueNativeAllocate(C.int(int32(len(argValues))))

    for ii := range argValues {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(nargValues)),
                             C.native_voidp(unsafe.Pointer(argValues[ii].nativeCPtr())),
                             C.int(int32(ii)))
    }

    nretValues := C._TVMValueNativeAllocate(C.int(int32(len(retValues))))

    for ii := range retValues {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(nretValues)),
                             C.native_voidp(unsafe.Pointer(retValues[ii].nativeCPtr())),
                             C.int(int32(ii)))
    }

	result := (int32)(C._TVMFuncCall(C.uintptr_t(funp),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nargValues))),
                                     C.native_voidp(&(typeCodes[0])), C.int(numArgs),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nretValues))),
                                     C.native_voidp(retTypeCode)))

    for ii := range argValues {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(argValues[ii].nativeCPtr())),
                             C.native_voidp(unsafe.Pointer(nargValues)),
                             C.int(int32(ii)))
    }

    C._TVMValueNativeFree(C.native_voidp(unsafe.Pointer(nargValues)))


    for ii := range retValues {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(retValues[ii].nativeCPtr())),
                             C.native_voidp(unsafe.Pointer(nretValues)),
                             C.int(int32(ii)))
    }

    C._TVMValueNativeFree(C.native_voidp(unsafe.Pointer(nretValues)))

    if result != 0 {
	    err = errors.New(getTVMLastError())
    }

    return
}
