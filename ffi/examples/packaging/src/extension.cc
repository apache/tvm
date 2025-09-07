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
 * \file example.cc
 * \brief Example of a tvm-ffi based library that registers various functions.
 *
 * It is a simple example that demonstrates how to package a tvm-ffi library into a python wheel.
 * The library is written in C++ and can be compiled into a shared library.
 * The shared library can then be loaded into python and used to call the functions.
 */
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

namespace my_ffi_extension {

namespace ffi = tvm::ffi;

/*!
 * \brief Raises a runtime error
 *
 * This is an example function to show how to raise and propagate
 * an error across the language boundary.
 *
 * \param msg The message to raise the error with
 */
void RaiseError(ffi::String msg) { TVM_FFI_THROW(RuntimeError) << msg; }

void AddOne(ffi::Tensor x, ffi::Tensor y) {
  // implementation of a library function
  TVM_FFI_ICHECK(x->ndim == 1) << "x must be a 1D tensor";
  DLDataType f32_dtype{kDLFloat, 32, 1};
  TVM_FFI_ICHECK(x->dtype == f32_dtype) << "x must be a float tensor";
  TVM_FFI_ICHECK(y->ndim == 1) << "y must be a 1D tensor";
  TVM_FFI_ICHECK(y->dtype == f32_dtype) << "y must be a float tensor";
  TVM_FFI_ICHECK(x->shape[0] == y->shape[0]) << "x and y must have the same shape";
  for (int i = 0; i < x->shape[0]; ++i) {
    static_cast<float*>(y->data)[i] = static_cast<float*>(x->data)[i] + 1;
  }
}

// expose global symbol add_one
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, my_ffi_extension::AddOne);

// The static initialization block is
// called once when the library is loaded.
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  // In this particular example, we use the reflection mechanisms to
  // register the functions directly into the global function table.
  //
  // This is an alternative approach to TVM_FFI_DLL_EXPORT_TYPED_FUNC
  // that exports the function directly as C symbol that follows tvm-ffi abi.
  //
  // - For functions that are expected to be static part of tvm_ffi_example project,
  //   one can use reflection mechanisms to register the globa function.
  // - For functions that are compiled and dynamically loaded at runtime, consider
  //   using the normal export mechanism so they won't be exposed to the global function table.
  //
  // Make sure to have a unique name across all registered functions,
  // always prefix with a package namespace name to avoid name collision.
  //
  // The function can then be found via tvm_ffi.get_global_func(name)
  // If the function is expected to stay throughout the lifetime of the program/
  //
  // When registering via reflection mechanisms, the library do not need to be loaded via
  // tvm::ffi::Module::LoadFromFile, instead, just load the dll or simply bundle into the
  // final project
  refl::GlobalDef().def("my_ffi_extension.raise_error", RaiseError);
});
}  // namespace my_ffi_extension
