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
    MOD_LIB    = "./deploy.so"
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
    return
  }

  fmt.Printf("Module Imported\n")

  // Get function pointer for tvm.graph_runtime.create
  fun := gotvm.NewTVMValue()
  funp := fun.GetV_handle()
  if gotvm.TVMFuncGetGlobal("tvm.graph_runtime.create", &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // DLTensor allocation attributes
  var ndim int32 = 1
  dtype_code := gotvm.KDLFloat
  var dtype_bits int32 = 32
  var dtype_lanes int32 = 1
  device_type := gotvm.KDLCPU
  var device_id int32 = 0
  tshape_in  := []int64{4}

  // Allocate input DLTensor : in_x
  in_x := gotvm.NewDLTensor()

  if gotvm.TVMArrayAlloc(&(tshape_in[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &in_x) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Allocate input DLTensor : in_y
  in_y := gotvm.NewDLTensor()
  if gotvm.TVMArrayAlloc(&(tshape_in[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &in_y) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Allocate output DLTensor
  out := gotvm.NewDLTensor()
  tshape_out := []int64{4}

  if gotvm.TVMArrayAlloc(&(tshape_out[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &out) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  fmt.Printf("Input and Output DLTensors allocated\n")

  // Fill Input Data : in_x , in_y
  // We use unsafe package to access underlying array to any type.
  in_x_slice := (*[1<<15] float32)(unsafe.Pointer(in_x.GetData()))[:4:4]
  in_y_slice := (*[1<<15] float32)(unsafe.Pointer(in_y.GetData()))[:4:4]

  for ii := 0; ii < 4 ; ii++ {
      in_x_slice[ii] = float32(ii)
      in_y_slice[ii] = float32(ii+5)
  }

  fmt.Printf("X: %v\n", in_x_slice)
  fmt.Printf("Y: %v\n", in_y_slice)

  // Get module function : myadd
  if gotvm.TVMModGetFunction(modp, "myadd", 1, &funp) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }

  // Fill arguments for myadd
  args_in := []gotvm.TVMValue{gotvm.NewTVMValue(),
                             gotvm.NewTVMValue(),
                             gotvm.NewTVMValue()}

  args_in[0].SetV_aHandle(in_x);
  args_in[1].SetV_aHandle(in_y);
  args_in[2].SetV_aHandle(out);
  type_codes := []int32{gotvm.KArrayHandle, gotvm.KArrayHandle, gotvm.KArrayHandle}

  args_out := []gotvm.TVMValue{}

  // Call module function myadd
  var ret int32
  if gotvm.TVMFuncCall(funp, args_in, &(type_codes[0]), 3, args_out, &ret) != 0 {
    fmt.Printf("%v", gotvm.TVMGetLastError())
    return
  }
  fmt.Printf("Module function myadd executed\n")

  for ii := range args_in {
    gotvm.DeleteTVMValue(args_in[ii])
  }

  // We use unsafe package to access underlying array to any type.
  out_slice := (*[1<<15] float32)(unsafe.Pointer(out.GetData()))[:4:4]
  fmt.Printf("Result:%v\n", out_slice)

  // Free all the allocations safely
  gotvm.DeleteTVMValue(mod)
  gotvm.DeleteTVMValue(fun)

  gotvm.TVMArrayFree(in_x)
  gotvm.TVMArrayFree(in_y)
  gotvm.TVMArrayFree(out)
}
