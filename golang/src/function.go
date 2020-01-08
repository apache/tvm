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
    "reflect"
    "fmt"
)

// Function type in golang hold pointer for the TVMFunction handle.
type Function uintptr

// nativeCPtr returns type freed uintptr for the Function.
func (tvmfunction Function) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(tvmfunction)
    return
}

// Invoke calls the TVM packed function referred by the handle with given arguments.
func (tvmfunction *Function) Invoke(args ...interface{}) (retVal *Value, err error) {
    funccall := func (fargs ...interface{}) (*Value, error) {
        return callNativeFunction(tvmfunction, fargs)
    }
    // Check is any args are contain any ValueArray
    // Possible is it's a args forward from one packed function to another.
    valueArrayFound := false
    for ii := range args {
        switch args[ii].(type) {
            case []*Value:
                valueArrayFound = true
        }
    }

    if !valueArrayFound {
        return funccall(args...)
    }
    if len(args) != 1 {
        err = fmt.Errorf("Not supported if packed function args are a mix of []Value and other types")
        return
    }

    valArray := args[0].([]*Value)
    if len(valArray) > 0 {
        newArgs := make([]interface{}, len(valArray))
        for ii := range valArray {
            newVal := newTVMValue()
            newVal.moveFrom(valArray[ii])
            newArgs[ii] = newVal
        }

        return funccall(newArgs...)
    }
    return funccall()
}

// FuncListGlobalNames is used to query global callable packed function names from TVM.
//
// returns slice of string holding function names and error if any.
func FuncListGlobalNames() (retVal []string, err error) {
    var str string
    ret := (int32)(C._TVMFuncListGlobalNames(unsafe.Pointer((&str))))
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
func GetGlobalFunction(funcname string) (retVal *Function, err error) {
    var funp uintptr

    cfuncname := C.CString(funcname)
    ret := (int32)(C.TVMFuncGetGlobal(cfuncname,
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

// callNativeFunction is routine which calls gotvm native wrapper with given arguments.
//
// `handle` is the handle for Function.
//
// `args` are the variadic arguments to the Function.
//
// returns the interface for the return value from TVM if any and error if any.
func callNativeFunction(handle *Function, args []interface{}) (retVal *Value, err error) {
    argsIn := make([]*Value, len(args))
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

    retVal = newTVMValue()
    argsOut := []*Value{retVal}
    retTypeCode := KNull
    err = nativeTVMFuncCall(handle, argsIn, typeCodes, argsOut, &retTypeCode)
    if err != nil {
        retVal = nil
        return
    }
    retVal.isLocal = false
    retVal.dtype = retTypeCode
    return
}

// nativeTVMFuncFree free the function handle allocated in TVM runtime.
//
// `funp` is the Function handle to be freed.
func nativeTVMFuncFree(funp *Function) (retVal int32) {
    retVal = (int32) (C.TVMFuncFree(C.TVMFunctionHandle(funp.nativeCPtr())))
    return
}

// nativeToGoSlice converts native TVMValue array to Golang slice of TVMValue
//
//
func nativeToGoSlice(nargValues (*C.void), argValues []*Value, typeCodes []int32) {
    for ii := range argValues {
        C._TVMValueNativeGet(unsafe.Pointer(argValues[ii].nativeCPtr()),
                             unsafe.Pointer(nargValues),
                             C.int(int32(ii)))
        argValues[ii].dtype = typeCodes[ii]
    }
}

// nativeFromGoSlice converts golang slice of TVMValue to native TVMValue array.
//
//
func nativeFromGoSlice(argValues []*Value) (nptr (*C.void)) {
    nargValues := ((uintptr)(C.malloc(C.ulong(C.sizeof_TVMValue * len(argValues)))))
    for ii := range argValues {
        C._TVMValueNativeSet(unsafe.Pointer(nargValues),
                             unsafe.Pointer(argValues[ii].nativeCPtr()),
                             C.int(int32(ii)))
    }
    nptr = (*C.void)(unsafe.Pointer(nargValues))
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
func nativeTVMFuncCall(funp *Function, argValues []*Value, typeCodes []int32,
                 retValues []*Value, retTypeCode *int32) (err error) {
    nargValues := nativeFromGoSlice(argValues)
    nretValues := nativeFromGoSlice(retValues)
	result := (int32)(C.TVMFuncCall(C.TVMFunctionHandle(*funp),
                                    (*C.TVMValue)(unsafe.Pointer(nargValues)),
                                    (*C.int)(unsafe.Pointer(&(typeCodes[0]))),
                                    C.int(len(argValues)),
                                    (*C.TVMValue)(unsafe.Pointer(nretValues)),
                                    (*C.int)(unsafe.Pointer(retTypeCode))))
    nativeToGoSlice(nargValues, argValues, typeCodes)
    nativeToGoSlice(nretValues, retValues, (*[1<<31] int32)(unsafe.Pointer(retTypeCode))[:1:1])
    C.free(unsafe.Pointer(nargValues))
    C.free(unsafe.Pointer(nretValues))

    if result != 0 {
	    err = errors.New(getTVMLastError())
    }
    return
}

// goCallBack is a structure holding the go callback function pointer.
// This wrapping is necessary as cgo doesn't support
// passing golang functions type conversion to native.
type goCallBack struct {
    cb func (args ...*Value) (interface{}, error)
}

//export goTVMCallback
func goTVMCallback(args C.native_voidp, typeCodes C.native_voidp, numArgs int32,
                   retArg C.native_voidp, resourceHandle C.native_voidp) (ret int32){
    fcb := (*goCallBack)(resourceHandle)
    // Make Value Sice from native TVMValue pointer.
    argValues := make([]*Value, numArgs)

    for ii := range argValues {
        argValues[ii] = newTVMValue()
        argValues[ii].isLocal = false
    }

    // Prepare arguments for golang callback function
    nativeToGoSlice((*C.void)(unsafe.Pointer(args)), argValues,
                    (*[1<<31] int32)(unsafe.Pointer(typeCodes))[:numArgs:numArgs])
    cbargs := argValues

    // Execute the callback
    retVal, err := fcb.cb(cbargs...)
    if err != nil {
        errStr := err.Error()
        setTVMLastError(errStr)
        return -1
    }

    // It's possible a packed function directly return
    // the return value of another packed function.
    //
    // Inside a packed func :
    //      ```return pfunc.Invoke(args)```
    //
    // In this case pfunc returns nil which is
    // returned as an interface holding nil *Value.
    // Which becomes a valid retVal holding nil *Value.
    isRetNull := false
    switch retVal.(type) {
        case *Value:
            pRet := retVal.(*Value)
            if pRet == nil {
                isRetNull = true
            }
    }

    // Handle return value from callback function
    if retVal != nil && !isRetNull {
        var retTypeCode int32
        retValues := []*Value{newTVMValue()}

        retTypeCode, err = retValues[0].setValue(retVal)
        if err != nil {
            errStr := err.Error()
            setTVMLastError(errStr)
            return -1
        }
        nretValues := nativeFromGoSlice(retValues)

        // Handle KStr, KBytes: Local finalizers shouldn't try freeing them.
        retValues[0].isLocal = false

        apiRet := (int32) (C.TVMCFuncSetReturn(C.TVMRetValueHandle(retArg),
                                               (*C.TVMValue)(unsafe.Pointer(nretValues)),
                                               (*C.int)(unsafe.Pointer(&retTypeCode)), 1))
        C.free(unsafe.Pointer(nretValues))
        if apiRet != 0 {
            errStr := string("TVMCFuncSetReturn failed ")
            setTVMLastError(errStr)
        }
    }
    return
}

// ConvertFunction converts given golang function to TVM packed function.
//
// `args[0]` function pointer for a type ```func (args ...interface{}) (interface{})```
//
// Returns Function handle and err if any.
func ConvertFunction(args ...interface{}) (retVal *Function, err error) {
    function := args[0].(func (args ...*Value) (interface{}, error))
    fcb := &goCallBack{cb:function}
    var funp uintptr

    result := (int32) (C._ConvertFunction(unsafe.Pointer(fcb),
                                          unsafe.Pointer(&funp)))
    if result != 0 {
	    err = errors.New(getTVMLastError())
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

// RegisterFunction registers the golang func in TVM runtime global space.
//
// `args[0]` function pointer for a type ```func (args ...interface{}) (interface{})```
//
// `args[1]` Optional argument of function name with which it will be registered.
//           If not passed we use function name from reflection.
//
// Returns err indicating native error if any.
func RegisterFunction(args ...interface{}) (err error) {
    fhandle, err := ConvertFunction(args...)
    if err != nil {
        return
    }

    funcname := runtime.FuncForPC(reflect.ValueOf(args[0]).Pointer()).Name()
    if len(args) > 1 {
        funcname = args[1].(string)
    }

    cfuncname := C.CString(funcname)
    result := (int32) (C.TVMFuncRegisterGlobal(cfuncname,
                                               C.TVMFunctionHandle(*fhandle),
                                               0)); // Override = False
    C.free(unsafe.Pointer(cfuncname))
    if result != 0 {
	    err = errors.New(getTVMLastError())
    }
    // Clear the finalizer as we don't need to control it anymore.
    runtime.SetFinalizer(fhandle, nil)
    return
}
