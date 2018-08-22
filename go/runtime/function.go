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
    "reflect"
)

// Function type in golang hold pointer for the TVMFunction handle.
type Function uintptr

// nativeCPtr returns type freed uintptr for the Function.
func (tvmfunction Function) nativeCPtr() (retVal uintptr) {
    retVal = (uintptr)(tvmfunction)
    return
}

// Invoke calls the TVM packed function referred by the handle with given arguments.
func (tvmfunction Function) Invoke(args ...interface{}) (retVal *Value, err error) {
    funccall := func (args ...interface{}) (*Value, error) {
        return callNativeFunction(tvmfunction, args)
    }

    return funccall(args...)
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
func GetGlobalFunction(funcname string) (retVal *Function, err error) {
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

// nativeTVMFuncFree free the function handle allocated in TVM runtime.
//
// `funp` is the Function handle to be freed.
func nativeTVMFuncFree(funp Function) (retVal int32) {
    retVal = (int32) (C.TVMFuncFree(C.TVMFunctionHandle(funp.nativeCPtr())))
    return
}

// nativeToGoSlice converts native TVMValue array to Golang slice of TVMValue
//
//
func nativeToGoSlice(nargValues C.native_voidp, argValues []Value) {
    for ii := range argValues {
        C._TVMValueNativeGet(C.native_voidp(unsafe.Pointer(argValues[ii].nativeCPtr())),
                             C.native_voidp(unsafe.Pointer(nargValues)),
                             C.int(int32(ii)))
    }
}

// nativeFromGoSlice converts golang slice of TVMValue to native TVMValue array.
//
//
func nativeFromGoSlice(argValues []Value) (nptr C.native_voidp) {
    nargValues := C._TVMValueNativeAllocate(C.int(int32(len(argValues))))

    for ii := range argValues {
        C._TVMValueNativeSet(C.native_voidp(unsafe.Pointer(nargValues)),
                             C.native_voidp(unsafe.Pointer(argValues[ii].nativeCPtr())),
                             C.int(int32(ii)))
    }

    nptr = nargValues

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

    nargValues := nativeFromGoSlice(argValues)
    nretValues := nativeFromGoSlice(retValues)

	result := (int32)(C._TVMFuncCall(C.uintptr_t(funp),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nargValues))),
                                     C.native_voidp(&(typeCodes[0])), C.int(len(argValues)),
                                     C.uintptr_t((uintptr)(unsafe.Pointer(nretValues))),
                                     C.native_voidp(retTypeCode)))


    nativeToGoSlice(nargValues, argValues)
    nativeToGoSlice(nretValues, retValues)

    C._TVMValueNativeFree(nargValues)
    C._TVMValueNativeFree(nretValues)

    if result != 0 {
	    err = errors.New(getTVMLastError())
    }

    return
}

// goCallBack is a structure holding the go callback function pointer.
// This wrapping is necessary as cgo doesn't support
// passing golang functions type conversion to native.
type goCallBack struct {
    cb func (args ...interface{}) (interface{}, error)
}

//export goTVMCallback
func goTVMCallback(args C.native_voidp, typeCodes C.native_voidp, numArgs int32,
                   retArg C.native_voidp, resourceHandle C.native_voidp) (ret int32){

    fcb := (*goCallBack)(resourceHandle)

    // Make Value Sice from native TVMValue pointer.
    argValues := make([]Value, numArgs)

    for ii := range argValues {
        argValues[ii] = newTVMValue()
    }

    defer func() {
        for ii := range argValues {
            argValues[ii].clearVStr(argValues[ii])
            argValues[ii].deleteTVMValue()
        }
    }()

    // Prepare arguments for golang callback function
    cbargs := make([]interface{}, numArgs)
    typeCodesSlice := (*[1<<31] int32)(unsafe.Pointer(typeCodes))[:numArgs:numArgs]

    nativeToGoSlice(C.native_voidp(unsafe.Pointer(args)), argValues)

    for ii := range argValues {
        value, err := argValues[ii].getValue(typeCodesSlice[ii])
        if err != nil {
            errStr := err.Error()
            C._TVMAPISetLastError(*(*C._gostring_)(unsafe.Pointer(&errStr)))
            return -1
        }
        cbargs[ii] = value
    }

    // Execute the callback
    retVal, err := fcb.cb(cbargs...)

    if err != nil {
        errStr := err.Error()
        C._TVMAPISetLastError(*(*C._gostring_)(unsafe.Pointer(&errStr)))
        return -1
    }

    // Handle return value from callback function
    if retVal != nil {
        var retTypeCode int32
        retValues := []Value{newTVMValue()}
        defer retValues[0].deleteTVMValue()

        retTypeCode, err = retValues[0].setValue(retVal)
        if err != nil {
            errStr := err.Error()
            C._TVMAPISetLastError(*(*C._gostring_)(unsafe.Pointer(&errStr)))
            return -1
        }

        nretValues := nativeFromGoSlice(retValues)

        apiRet := (int32) (C._TVMCFuncSetReturn(retArg,
                                                C.native_voidp(unsafe.Pointer(nretValues)),
                                                C.native_voidp(unsafe.Pointer(&retTypeCode)), 1))
        C._TVMValueNativeFree(nretValues)

        if apiRet != 0 {
            errStr := string("TVMCFuncSetReturn failed ")
            C._TVMAPISetLastError(*(*C._gostring_)(unsafe.Pointer(&errStr)))
        }
    }

    return
}

// ConvertFunction converts given golang function to TVM packed function.
//
// `args[0]` function pointer for a type ```func (args ...interface{}) (interface{})```
//
// Returns Function handle and err if any.
func ConvertFunction(args ...interface{}) (fhandle Function, err error) {
    function := args[0].(func (args ...interface{}) (interface{}, error))

    fcb := &goCallBack{cb:function}

    var funp uintptr

    result := (int32) (C._ConvertFunction((C.native_voidp)(unsafe.Pointer(fcb)),
                                          C.native_voidp(&funp)))

    if result != 0 {
	    err = errors.New(getTVMLastError())
    }

    fhandle = Function(funp)

    handle := new(Function)
    *handle = Function(funp)

    finalizer := func(fhandle *Function) {
        nativeTVMFuncFree(*fhandle)
        fhandle = nil
    }

    runtime.SetFinalizer(handle, finalizer)

    fhandle = *handle

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

    result := (int32) (C._RegisterFunction(*(*C._gostring_)(unsafe.Pointer(&funcname)),
                                           C.uintptr_t(fhandle)));

    if result != 0 {
	    err = errors.New(getTVMLastError())
    }

    return
}
