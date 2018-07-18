/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Sample golang application deployment over tvm.
 * \file simple.go
 */

package main

import (
    "fmt"
    "unsafe"
    "./tvmgo"
)

// NNVM compiled model paths.
const (
    MOD_LIB    = "./deploy.so"
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

  // DLTensor allocation attributes
  var ndim int32 = 1
  dtype_code := tvmgo.KDLFloat
  var dtype_bits int32 = 32
  var dtype_lanes int32 = 1
  device_type := tvmgo.KDLCPU
  var device_id int32 = 0
  tshape_in  := []int64{4}

  // Allocate input DLTensor : in_x
  in_x := tvmgo.NewDLTensor()
  if tvmgo.TVMArrayAlloc(&(tshape_in[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &in_x) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Allocate input DLTensor : in_y
  in_y := tvmgo.NewDLTensor()
  if tvmgo.TVMArrayAlloc(&(tshape_in[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &in_y) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Allocate output DLTensor
  out := tvmgo.NewDLTensor()
  tshape_out := []int64{4}

  if tvmgo.TVMArrayAlloc(&(tshape_out[0]), ndim, dtype_code, dtype_bits, dtype_lanes,
                         device_type, device_id, &out) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
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
  if tvmgo.TVMModGetFunction(modp, "myadd", 1, &funp) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }

  // Fill arguments for myadd
  args_in := []tvmgo.TVMValue{tvmgo.NewTVMValue(),
                             tvmgo.NewTVMValue(),
                             tvmgo.NewTVMValue()}

  args_in[0].SetV_handle(in_x.Swigcptr());
  args_in[1].SetV_handle(in_y.Swigcptr());
  args_in[2].SetV_handle(out.Swigcptr());
  type_codes := []int32{tvmgo.KArrayHandle, tvmgo.KArrayHandle, tvmgo.KArrayHandle}

  args_out := []tvmgo.TVMValue{}

  // Call module function myadd
  var ret int32
  if tvmgo.TVMFuncCall(funp, args_in, &(type_codes[0]), 3, args_out, &ret) != 0 {
    fmt.Printf("%v", tvmgo.TVMGetLastError())
    return
  }
  fmt.Printf("Module function myadd executed\n")

  for ii := range args_in {
    tvmgo.DeleteTVMValue(args_in[ii])
  }

  // We use unsafe package to access underlying array to any type.
  out_slice := (*[1<<15] float32)(unsafe.Pointer(out.GetData()))[:4:4]
  fmt.Printf("Result:%v\n", out_slice)

  // Free all the allocations safely
  tvmgo.DeleteTVMValue(mod)
  tvmgo.DeleteTVMValue(fun)

  tvmgo.TVMArrayFree(in_x)
  tvmgo.TVMArrayFree(in_y)
  tvmgo.TVMArrayFree(out)
}
