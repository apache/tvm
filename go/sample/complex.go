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
    MOD_LIB    = "./mobilenet.so"
    MOD_JSON   = "./mobilenet.json"
    MOD_PARAMS = "./mobilenet.params"
)

// main
func main() {
  // Welcome
  fmt.Printf("TVM Go Interface : v%v\n", gotvm.GOTVM_VERSION)
  fmt.Printf("TVM Version   : v%v\n", gotvm.TVM_VERSION)
  fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPACK_VERSION)

  // Query global functions available
  func_names := []string{}
  if gotvm.TVMFuncListGlobalNames(&func_names) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  fmt.Printf("Global Functions:%v\n", func_names)

  // Import tvm module (dso)
  mod := gotvm.NewTVMValue()
  modp := mod.GetV_handle()
  if gotvm.TVMModLoadFromFile(MOD_LIB, "so", &modp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    fmt.Printf("Please copy tvm compiled modules here and update the sample.go accordingly.")
    fmt.Printf("You may need to update MOD_LIB, MOD_JSON, MOD_PARAMS, tshape_in, tshape_out")
    return
  }

  fmt.Printf("Module Imported:%p\n", modp)

  // Get function pointer for tvm.graph_runtime.create
  fun := gotvm.NewTVMValue()
  funp := fun.GetV_handle()
  if gotvm.TVMFuncGetGlobal("tvm.graph_runtime.create", &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Read graph json
  bytes, err := ioutil.ReadFile(MOD_JSON)
  if err != nil {
    fmt.Print(err)
  }
  json_str := string(bytes)

  // Fill arguments for tvm.graph_runtime.create
  args_in := []gotvm.TVMValue{gotvm.NewTVMValue(),
                              gotvm.NewTVMValue(),
                              gotvm.NewTVMValue(),
                              gotvm.NewTVMValue()}

  args_in[0].SetV_str(json_str);
  args_in[1].SetV_handle(modp);
  args_in[2].SetV_int64((int64)(gotvm.KDLCPU));
  args_in[3].SetV_int64((int64)(0));
  type_codes := []int32{gotvm.KStr, gotvm.KModuleHandle, gotvm.KDLInt, gotvm.KDLInt}

  graphrt := gotvm.NewTVMValue()
  args_out := []gotvm.TVMValue{graphrt}

 // Load module on tvm runtime - call tvm.graph_runtime.create
  ret := new(int32)
  if gotvm.TVMFuncCall(funp, args_in, &(type_codes[0]), 4, args_out, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  args_in[0].UnSetV_str()

  for ii := range args_in {
    gotvm.DeleteTVMValue(args_in[ii])
  }

  fmt.Printf("Graph runtime Created\n")

  // DLTensor allocation attributes
  var ndim int32 = 4
  dtype_code := gotvm.KDLFloat
  var dtype_bits int32 = 32
  var dtype_lanes int32 = 1
  device_type := gotvm.KDLCPU
  var device_id int32 = 0
  tshape_in  := []int64{1, 224, 224, 3}

  // Allocate input DLTensor
  in_x := gotvm.NewDLTensor()
  if gotvm.TVMArrayAlloc(&(tshape_in[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &in_x) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Allocate output DLTensor
  ndim = 2

  out := gotvm.NewDLTensor()
  tshape_out := []int64{1, 1001}

  if gotvm.TVMArrayAlloc(&(tshape_out[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &out) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  fmt.Printf("Input and Output DLTensors allocated\n")

  // Get module function from graph runtime : load_params
  if gotvm.TVMModGetFunction(graphrt.GetV_handle(), "load_params", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Read params
  bytes, err = ioutil.ReadFile(MOD_PARAMS)
  if err != nil {
    fmt.Print(err)
  }
  params_str := string(bytes)

  params_byte_array := gotvm.NewTVMByteArray()
  params_byte_array.SetData(params_str)
  params_byte_array.SetSize(int64(len(params_str)))

  // Fill arguments for load_params
  args_in = []gotvm.TVMValue{gotvm.NewTVMValue()}
  args_in[0].SetV_handle(params_byte_array.Nativecptr());
  type_codes = []int32{gotvm.KBytes}

  args_out = []gotvm.TVMValue{}

  // Call runtime function load_params
  if gotvm.TVMFuncCall(funp, args_in, &(type_codes[0]), 1, args_out, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Module params loaded\n")

  gotvm.DeleteTVMByteArray(params_byte_array)
  for ii := range args_in {
    gotvm.DeleteTVMValue(args_in[ii])
  }

  // Get module function from graph runtime : set_input
  if gotvm.TVMModGetFunction(graphrt.GetV_handle(), "set_input", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Fill arguments for set_input
  args_in = []gotvm.TVMValue{gotvm.NewTVMValue(),
                             gotvm.NewTVMValue()}

  args_in[0].SetV_str("input");

  // Set some data in input DLTensor
  // We use unsafe package to access underlying array to any type.
  in_slice := (*[1<<31] float32)(unsafe.Pointer(in_x.GetData()))[:(224*224*3):(224*224*3)]
  rand.Seed(10)
  rand.Shuffle(len(in_slice), func(i, j int) { in_slice[i], in_slice[j] = rand.Float32(), rand.Float32() })

  args_in[1].SetV_aHandle(in_x);
  type_codes = []int32{gotvm.KStr, gotvm.KArrayHandle}

  args_out = []gotvm.TVMValue{}

  // Call runtime function set_input
  if gotvm.TVMFuncCall(funp, args_in, &(type_codes[0]), 2, args_out, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Module input is set\n")

  args_in[0].UnSetV_str()

  for ii := range args_in {
    gotvm.DeleteTVMValue(args_in[ii])
  }

  // Get module function from graph runtime : run
  if gotvm.TVMModGetFunction(graphrt.GetV_handle(), "run", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Fill arguments for get_output
  args_in = []gotvm.TVMValue{}
  args_out = []gotvm.TVMValue{}
  type_codes = []int32{gotvm.KDLInt}

  // Call runtime function get_output
  if gotvm.TVMFuncCall(funp, args_in, &(type_codes[0]), 0, args_out, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Module Executed \n")

  // Get module function from graph runtime : get_output
  if gotvm.TVMModGetFunction(graphrt.GetV_handle(), "get_output", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Fill arguments for get_output
  args_in = []gotvm.TVMValue{gotvm.NewTVMValue(),
                             gotvm.NewTVMValue()}

  args_in[0].SetV_int64((int64)(0));
  args_in[1].SetV_aHandle(out);
  type_codes = []int32{gotvm.KDLInt, gotvm.KArrayHandle}

  args_out = []gotvm.TVMValue{}

  // Call runtime function get_output
  if gotvm.TVMFuncCall(funp, args_in, &(type_codes[0]), 2, args_out, ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Got Module Output \n")

  for ii := range args_in {
    gotvm.DeleteTVMValue(args_in[ii])
  }

  // We use unsafe package to access underlying array to any type.
  out_slice := (*[1<<15] float32)(unsafe.Pointer(out.GetData()))[:1001:1001]
  fmt.Printf("Result:%v\n", out_slice[:10])

  // Free all the allocations safely
  gotvm.DeleteTVMValue(mod)
  gotvm.DeleteTVMValue(fun)
  gotvm.DeleteTVMValue(graphrt)

  gotvm.TVMArrayFree(in_x)
  gotvm.TVMArrayFree(out)
}
