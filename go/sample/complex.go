/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application deployment over tvm.
 * \file complex.go
 */

package main

import (
    "fmt"
    "unsafe"
    "io/ioutil"
    "math/rand"
    "./gotvm"
)

// NNVM compiled model paths.
const (
    modLib    = "./mobilenet.so"
    modJSON   = "./mobilenet.json"
    modParams = "./mobilenet.params"
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
    fmt.Printf("Please copy tvm compiled modules here and update the sample.go accordingly.")
    fmt.Printf("You may need to update modLib, modJSON, modParams, tshapeIn, tshapeOut")
    return
  }
  defer gotvm.TVMModFree(modp)

  fmt.Printf("Module Imported:%p\n", modp)

  // Get function pointer for tvm.graph_runtime.create
  var funp gotvm.TVMFunction

  if gotvm.TVMFuncGetGlobal("tvm.graph_runtime.create", &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Read graph json
  bytes, err := ioutil.ReadFile(modJSON)
  if err != nil {
    fmt.Print(err)
  }
  jsonStr := string(bytes)

  // Fill arguments for tvm.graph_runtime.create
  argsIn := []gotvm.TVMValue{gotvm.NewTVMValue(),
                              gotvm.NewTVMValue(),
                              gotvm.NewTVMValue(),
                              gotvm.NewTVMValue()}

  argsIn[0].SetVStr(jsonStr);
  argsIn[1].SetVMHandle(modp);
  argsIn[2].SetVInt64((int64)(gotvm.KDLCPU));
  argsIn[3].SetVInt64((int64)(0));
  typeCodes := []int32{gotvm.KStr, gotvm.KModuleHandle, gotvm.KDLInt, gotvm.KDLInt}

  graphrt := gotvm.NewTVMValue()
  argsOut := []gotvm.TVMValue{graphrt}

 // Load module on tvm runtime - call tvm.graph_runtime.create
  ret := new(int32)
  if gotvm.TVMFuncCall(funp, argsIn, &(typeCodes[0]), 4, argsOut, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  argsIn[0].UnSetVStr()
  gotvm.TVMFuncFree(funp)

  graphmod := graphrt.GetVMHandle()
  graphrt.Delete()

  for ii := range argsIn {
    argsIn[ii].Delete()
  }

  fmt.Printf("Graph runtime Created\n")

  // TVMArray allocation attributes
  var ndim int32 = 4
  dtypeCode := gotvm.KDLFloat
  var dtypeBits int32 = 32
  var dtypeLanes int32 = 1
  deviceType := gotvm.KDLCPU
  var deviceID int32
  tshapeIn  := []int64{1, 224, 224, 3}

  // Allocate input TVMArray
  var inX gotvm.TVMArray

  if gotvm.TVMArrayAlloc(tshapeIn, ndim, dtypeCode, dtypeBits, dtypeLanes,
                         deviceType, deviceID, &inX) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  defer gotvm.TVMArrayFree(inX)

  // Allocate output TVMArray
  ndim = 2

  var out gotvm.TVMArray

  tshapeOut := []int64{1, 1001}

  if gotvm.TVMArrayAlloc(tshapeOut, ndim, dtypeCode, dtypeBits, dtypeLanes,
                         deviceType, deviceID, &out) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  defer gotvm.TVMArrayFree(out)

  fmt.Printf("Input and Output TVMArrays allocated\n")

  // Get module function from graph runtime : load_params
  if gotvm.TVMModGetFunction(graphmod, "load_params", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Read params
  bytes, err = ioutil.ReadFile(modParams)
  if err != nil {
    fmt.Print(err)
  }
  paramsStr := string(bytes)

  paramsByteArray := gotvm.NewTVMByteArray()
  paramsByteArray.SetData(paramsStr)

  // Fill arguments for load_params
  argsIn = []gotvm.TVMValue{gotvm.NewTVMValue()}
  argsIn[0].SetVBHandle(paramsByteArray);
  typeCodes = []int32{gotvm.KBytes}

  argsOut = []gotvm.TVMValue{}

  // Call runtime function load_params
  if gotvm.TVMFuncCall(funp, argsIn, &(typeCodes[0]), 1, argsOut, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Module params loaded\n")

  gotvm.TVMFuncFree(funp)
  paramsByteArray.Delete()
  for ii := range argsIn {
    argsIn[ii].Delete()
  }

  // Get module function from graph runtime : set_input
  if gotvm.TVMModGetFunction(graphmod, "set_input", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Fill arguments for set_input
  argsIn = []gotvm.TVMValue{gotvm.NewTVMValue(),
                             gotvm.NewTVMValue()}

  argsIn[0].SetVStr("input");

  // Set some data in input TVMArray
  // We use unsafe package to access underlying array to any type.
  inSlice := (*[1<<31] float32)(unsafe.Pointer(inX.GetData()))[:(224*224*3):(224*224*3)]
  rand.Seed(10)
  rand.Shuffle(len(inSlice), func(i, j int) { inSlice[i], inSlice[j] = rand.Float32(), rand.Float32() })

  argsIn[1].SetVAHandle(inX);
  typeCodes = []int32{gotvm.KStr, gotvm.KArrayHandle}

  argsOut = []gotvm.TVMValue{}

  // Call runtime function set_input
  if gotvm.TVMFuncCall(funp, argsIn, &(typeCodes[0]), 2, argsOut, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Module input is set\n")

  argsIn[0].UnSetVStr()
  gotvm.TVMFuncFree(funp)

  for ii := range argsIn {
    argsIn[ii].Delete()
  }

  // Get module function from graph runtime : run
  if gotvm.TVMModGetFunction(graphmod, "run", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Fill arguments for get_output
  argsIn = []gotvm.TVMValue{}
  argsOut = []gotvm.TVMValue{}
  typeCodes = []int32{gotvm.KDLInt}

  // Call runtime function get_output
  if gotvm.TVMFuncCall(funp, argsIn, &(typeCodes[0]), 0, argsOut, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  gotvm.TVMFuncFree(funp)
  fmt.Printf("Module Executed \n")

  // Get module function from graph runtime : get_output
  if gotvm.TVMModGetFunction(graphmod, "get_output", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Fill arguments for get_output
  argsIn = []gotvm.TVMValue{gotvm.NewTVMValue(),
                             gotvm.NewTVMValue()}

  argsIn[0].SetVInt64((int64)(0));
  argsIn[1].SetVAHandle(out);
  typeCodes = []int32{gotvm.KDLInt, gotvm.KArrayHandle}

  argsOut = []gotvm.TVMValue{}

  // Call runtime function get_output
  if gotvm.TVMFuncCall(funp, argsIn, &(typeCodes[0]), 2, argsOut, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Got Module Output \n")

  gotvm.TVMFuncFree(funp)
  for ii := range argsIn {
    argsIn[ii].Delete()
  }

  // We use unsafe package to access underlying array to any type.
  outSlice := (*[1<<15] float32)(unsafe.Pointer(out.GetData()))[:1001:1001]
  fmt.Printf("Result:%v\n", outSlice[:10])
}
