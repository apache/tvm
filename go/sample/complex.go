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
    "./tvmgo"
)

// NNVM compiled model paths.
const (
    MOD_LIB    = "./mobilenet.so"
    MOD_JSON   = "./mobilenet.json"
    MOD_PARAMS = "./mobilenet.params"
)

// main
func main() {
  // Welcome
  fmt.Printf("TVM Go Interface : v%v\n", tvmgo.TVMGO_VERSION)
  fmt.Printf("TVM Version   : v%v\n", tvmgo.TVM_VERSION)
  fmt.Printf("DLPACK Version: v%v\n\n", tvmgo.DLPACK_VERSION)

  // Query global functions available
  func_names := []string{}
  if tvmgo.TVMFuncListGlobalNames(&func_names) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  fmt.Printf("Global Functions:%v\n", func_names)

  // Import tvm module (dso)
  mod := tvmgo.NewTVMValue()
  modp := mod.GetV_handle()
  if tvmgo.TVMModLoadFromFile(MOD_LIB, "so", &modp) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    fmt.Printf("Please copy tvm compiled modules here and update the sample.go accordingly.")
    fmt.Printf("You may need to update MOD_LIB, MOD_JSON, MOD_PARAMS, tshape_in, tshape_out")
    return
  }

  fmt.Printf("Module Imported\n")

  // Get function pointer for tvm.graph_runtime.create
  fun := tvmgo.NewTVMValue()
  funp := fun.GetV_handle()
  if tvmgo.TVMFuncGetGlobal("tvm.graph_runtime.create", &funp) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Read graph json
  bytes, err := ioutil.ReadFile(MOD_JSON)
  if err != nil {
    fmt.Print(err)
  }
  json_str := string(bytes)

  // Fill arguments for tvm.graph_runtime.create
  args_in := []tvmgo.TVMValue{tvmgo.NewTVMValue(),
                              tvmgo.NewTVMValue(),
                              tvmgo.NewTVMValue(),
                              tvmgo.NewTVMValue()}

  args_in[0].SetV_str(json_str);
  args_in[1].SetV_handle(modp);
  args_in[2].SetV_int64((int64)(tvmgo.KDLCPU));
  args_in[3].SetV_int64((int64)(0));
  type_codes := []int32{tvmgo.KStr, tvmgo.KModuleHandle, tvmgo.KDLInt, tvmgo.KDLInt}

  graphrt := tvmgo.NewTVMValue()
  args_out := []tvmgo.TVMValue{graphrt}

 // Load module on tvm runtime - call tvm.graph_runtime.create
  ret := new(int32)
  if tvmgo.TVMFuncCall(funp, args_in, &(type_codes[0]), 4, args_out, ret) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  for ii := range args_in {
    tvmgo.DeleteTVMValue(args_in[ii])
  }

  fmt.Printf("Graph runtime Created\n")

  // DLTensor allocation attributes
  var ndim int32 = 4
  dtype_code := tvmgo.KDLFloat
  var dtype_bits int32 = 32
  var dtype_lanes int32 = 1
  device_type := tvmgo.KDLCPU
  var device_id int32 = 0
  tshape_in  := []int64{1, 224, 224, 3}

  // Allocate input DLTensor
  in_x := tvmgo.NewDLTensor()
  if tvmgo.TVMArrayAlloc(&(tshape_in[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &in_x) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Allocate output DLTensor
  ndim = 2

  out := tvmgo.NewDLTensor()
  tshape_out := []int64{1, 1001}

  if tvmgo.TVMArrayAlloc(&(tshape_out[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &out) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  fmt.Printf("Input and Output DLTensors allocated\n")

  // Get module function from graph runtime : load_params
  if tvmgo.TVMModGetFunction(graphrt.GetV_handle(), "load_params", 1, &funp) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Read params
  bytes, err = ioutil.ReadFile(MOD_PARAMS)
  if err != nil {
    fmt.Print(err)
  }
  params_str := string(bytes)

  params_byte_array := tvmgo.NewTVMByteArray()
  params_byte_array.SetData(params_str)
  params_byte_array.SetSize(int64(len(params_str)))

  // Fill arguments for load_params
  args_in = []tvmgo.TVMValue{tvmgo.NewTVMValue()}
  args_in[0].SetV_handle(params_byte_array.Swigcptr());
  type_codes = []int32{tvmgo.KBytes}

  args_out = []tvmgo.TVMValue{}

  // Call runtime function load_params
  if tvmgo.TVMFuncCall(funp, args_in, &(type_codes[0]), 1, args_out, ret) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }
  fmt.Printf("Module params loaded\n")

  tvmgo.DeleteTVMByteArray(params_byte_array)
  for ii := range args_in {
    tvmgo.DeleteTVMValue(args_in[ii])
  }

  // Get module function from graph runtime : set_input
  if tvmgo.TVMModGetFunction(graphrt.GetV_handle(), "set_input", 1, &funp) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Fill arguments for set_input
  args_in = []tvmgo.TVMValue{tvmgo.NewTVMValue(),
                             tvmgo.NewTVMValue()}

  args_in[0].SetV_str("input");

  // Set some data in input DLTensor
  // We use unsafe package to access underlying array to any type.
  in_slice := (*[1<<31] float32)(unsafe.Pointer(in_x.GetData()))[:(224*224*3):(224*224*3)]
  rand.Seed(10)
  rand.Shuffle(len(in_slice), func(i, j int) { in_slice[i], in_slice[j] = rand.Float32(), rand.Float32() })

  args_in[1].SetV_handle(in_x.Swigcptr());
  type_codes = []int32{tvmgo.KStr, tvmgo.KArrayHandle}

  args_out = []tvmgo.TVMValue{}

  // Call runtime function set_input
  if tvmgo.TVMFuncCall(funp, args_in, &(type_codes[0]), 2, args_out, ret) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }
  fmt.Printf("Module input is set\n")

  for ii := range args_in {
    tvmgo.DeleteTVMValue(args_in[ii])
  }

  // Get module function from graph runtime : run
  if tvmgo.TVMModGetFunction(graphrt.GetV_handle(), "run", 1, &funp) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Fill arguments for get_output
  args_in = []tvmgo.TVMValue{}
  args_out = []tvmgo.TVMValue{}
  type_codes = []int32{tvmgo.KDLInt}

  // Call runtime function get_output
  if tvmgo.TVMFuncCall(funp, args_in, &(type_codes[0]), 0, args_out, ret) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }
  fmt.Printf("Module Executed \n")

  // Get module function from graph runtime : get_output
  if tvmgo.TVMModGetFunction(graphrt.GetV_handle(), "get_output", 1, &funp) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Fill arguments for get_output
  args_in = []tvmgo.TVMValue{tvmgo.NewTVMValue(),
                             tvmgo.NewTVMValue()}

  args_in[0].SetV_int64((int64)(0));
  args_in[1].SetV_handle(out.Swigcptr());
  type_codes = []int32{tvmgo.KDLInt, tvmgo.KArrayHandle}

  args_out = []tvmgo.TVMValue{}

  // Call runtime function get_output
  if tvmgo.TVMFuncCall(funp, args_in, &(type_codes[0]), 2, args_out, ret) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }
  fmt.Printf("Got Module Output \n")

  for ii := range args_in {
    tvmgo.DeleteTVMValue(args_in[ii])
  }

  // We use unsafe package to access underlying array to any type.
  out_slice := (*[1<<15] float32)(unsafe.Pointer(out.GetData()))[:1001:1001]
  fmt.Printf("Result:%v\n", out_slice[:10])

  // Free all the allocations safely
  tvmgo.DeleteTVMValue(mod)
  tvmgo.DeleteTVMValue(fun)
  tvmgo.DeleteTVMValue(graphrt)

  tvmgo.TVMArrayFree(in_x)
  tvmgo.TVMArrayFree(out)
}
