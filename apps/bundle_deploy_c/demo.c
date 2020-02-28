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
 * \brief Example code on load and run TVM modules
 * \file c_deploy.c
 */
#include <tvm/runtime/c_runtime_api.h>

#include <stdio.h>
#include <assert.h>

void Verify(TVMModuleHandle mod, const char * fname) {
  // Get the function from the module.
  TVMFunctionHandle f;
  TVMModGetFunction(mod, fname, 0, &f);

  /* CHECK(f != nullptr); */

  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* x;
  DLTensor* y;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[1] = {10};
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &x);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &y);
  for (int i = 0; i < shape[0]; ++i) {
    ((float*)(x->data))[i] = i;
    printf("%f\n", ((float*)(x->data))[i]);
  }

  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  // The signature of the function is specified in tvm.build
  TVMValue arg_values[2];
  arg_values[0].v_handle = x;
  arg_values[1].v_handle = y;
  int type_codes[2] = {kDLFloat, kDLFloat,};
  TVMValue ret_val[1];
  ret_val[0].v_handle = y;
  int ret_type_code[1] = {kDLFloat,};
  TVMFuncCall(f, arg_values, type_codes, 1, ret_val, ret_type_code);

  // Print out the output
  for (int i = 0; i < shape[0]; ++i) {
    /* assert(((float*)(y->data))[i] == (i + 1.0f)); */
    printf("%f vs %f\n", ((float*)(y->data))[i], i + 1.0f);
  }

  printf("Finish verification...\n");

  // Release memory
  TVMArrayFree(x);
  TVMArrayFree(y);
}

int main(void) {
  // Normally we can directly
  TVMModuleHandle mod_dylib;
  TVMModLoadFromFile("lib/test_addone_dll.so", "", &mod_dylib);
  /* LOG(INFO) << "Verify dynamic loading from test_addone_dll.so"; */
  Verify(mod_dylib, "addone");

  // For libraries that are directly packed as system lib and linked together with the app
  // We can directly use GetSystemLib to get the system wide library.
  /* LOG(INFO) << "Verify load function from system lib"; */
  TVMModuleHandle mod_syslib;
  TVMFuncGetGlobal("runtime.SystemLib", &mod_syslib);
  /* = (*tvm::runtime::Registry::Get("runtime.SystemLib"))(); */
  Verify(mod_syslib, "addonesys");
  
  return 0;
}
