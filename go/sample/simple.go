/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application deployment over tvm.
 * \file simple.go
 */

package main

import (
    "fmt"
    "unsafe"
    "./gotvm"
)

// NNVM compiled model paths.
const (
    modLib    = "./deploy.so"
)

// main
func main() {
  // Welcome
  fmt.Printf("TVM Go Interface : v%v\n", gotvm.GoTVMVersion)
  fmt.Printf("TVM Version   : v%v\n", gotvm.TVMVersion)
  fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPackVersion)

  // Query global functions available
  funcNames := []string{}
  if gotvm.TVMFuncListGlobalNames(&funcNames) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  fmt.Printf("Global Functions:%v\n", funcNames)

  // Import tvm module (dso)
  var modp gotvm.TVMModule

  if gotvm.TVMModLoadFromFile(modLib, "so", &modp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  fmt.Printf("Module Imported\n")

  // Get function pointer for tvm.graph_runtime.create
  var funp gotvm.TVMFunction
  if gotvm.TVMFuncGetGlobal("tvm.graph_runtime.create", &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // TVMArray allocation attributes
  var ndim int32 = 1
  dtypeCode := gotvm.KDLFloat
  var dtypeBits int32 = 32
  var dtypeLanes int32 = 1
  deviceType := gotvm.KDLCPU
  var deviceID int32
  tshapeIn  := []int64{4}

  // Allocate input TVMArray : inX
  var inX gotvm.TVMArray

  if gotvm.TVMArrayAlloc(&(tshapeIn[0]), ndim, dtypeCode, dtypeBits, dtypeLanes,
                         deviceType, deviceID, &inX) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Allocate input TVMArray : inY
  var inY gotvm.TVMArray
  if gotvm.TVMArrayAlloc(&(tshapeIn[0]), ndim, dtypeCode, dtypeBits, dtypeLanes,
                         deviceType, deviceID, &inY) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Allocate output TVMArray
  var out gotvm.TVMArray
  tshapeOut := []int64{4}

  if gotvm.TVMArrayAlloc(&(tshapeOut[0]), ndim, dtypeCode, dtypeBits, dtypeLanes,
                         deviceType, deviceID, &out) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  fmt.Printf("Input and Output TVMArrays allocated\n")

  // Fill Input Data : inX , inY
  // We use unsafe package to access underlying array to any type.
  inXSlice := (*[1<<15] float32)(unsafe.Pointer(inX.GetData()))[:4:4]
  inYSlice := (*[1<<15] float32)(unsafe.Pointer(inY.GetData()))[:4:4]

  for ii := 0; ii < 4 ; ii++ {
      inXSlice[ii] = float32(ii)
      inYSlice[ii] = float32(ii+5)
  }

  fmt.Printf("X: %v\n", inXSlice)
  fmt.Printf("Y: %v\n", inYSlice)

  // Get module function : myadd
  if gotvm.TVMModGetFunction(modp, "myadd", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Fill arguments for myadd
  argsIn := []gotvm.TVMValue{gotvm.NewTVMValue(),
                             gotvm.NewTVMValue(),
                             gotvm.NewTVMValue()}

  argsIn[0].SetVAHandle(inX);
  argsIn[1].SetVAHandle(inY);
  argsIn[2].SetVAHandle(out);
  typeCodes := []int32{gotvm.KArrayHandle, gotvm.KArrayHandle, gotvm.KArrayHandle}

  argsOut := []gotvm.TVMValue{}

  // Call module function myadd
  var ret int32
  if gotvm.TVMFuncCall(funp, argsIn, &(typeCodes[0]), 3, argsOut, &ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Module function myadd executed\n")

  for ii := range argsIn {
    argsIn[ii].Delete()
  }

  // We use unsafe package to access underlying array to any type.
  outSlice := (*[1<<15] float32)(unsafe.Pointer(out.GetData()))[:4:4]
  fmt.Printf("Result:%v\n", outSlice)

  // Free all the allocations safely

  gotvm.TVMArrayFree(inX)
  gotvm.TVMArrayFree(inY)
  gotvm.TVMArrayFree(out)
}
