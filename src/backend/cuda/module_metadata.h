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

/*! \file module_metadata.h
 *  \brief Shared CUDA compilation and versioned module serialization contract.
 */
#ifndef TVM_BACKEND_CUDA_MODULE_METADATA_H_
#define TVM_BACKEND_CUDA_MODULE_METADATA_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>

#include "../../runtime/metadata.h"
#include "../../support/bytes_io.h"

namespace tvm {
namespace backend {
namespace cuda {

inline void CompileSource(ffi::Bytes* code, ffi::String* format,
                          ffi::Map<ffi::String, ffi::String>* source) {
  if (*format != "cuda") return;
  auto config = source->Get("cuda.compile_config");
  TVM_FFI_CHECK(config.has_value(), ValueError)
      << "CUDA source artifact has no CompileConfig; regenerate it with an explicit configuration";
  auto compile = ffi::Function::GetGlobalRequired("tvm_callback_cuda_compile");
  ffi::String text(code->data(), code->size());
  auto result = compile(text, config.value()).cast<ffi::Array<ffi::Any>>();
  source->Set("cuda", text);
  source->Set("cuda.compile_config", result[2].cast<ffi::String>());
  source->Set("cuda.compile_log", result[3].cast<ffi::String>());
  *code = result[0].cast<ffi::Bytes>();
  *format = result[1].cast<ffi::String>();
  TVM_FFI_CHECK(*format == "ptx" || *format == "cubin" || *format == "fatbin", ValueError)
      << "CUDA compiler returned an unsupported binary format: " << *format;
}

inline ffi::Bytes SaveModule(ffi::String format,
                             ffi::Map<ffi::String, runtime::FunctionInfo> functions,
                             ffi::Bytes code, ffi::Map<ffi::String, ffi::String> source) {
  std::string buffer;
  support::BytesOutStream stream(&buffer);
  stream.Write(ffi::String("cuda.module.v1"));
  stream.Write(format);
  stream.Write(functions);
  stream.Write(code);
  stream.Write(source);
  return ffi::Bytes(std::move(buffer));
}

inline void LoadModule(const ffi::Bytes& bytes, ffi::String* format,
                       ffi::Map<ffi::String, runtime::FunctionInfo>* functions, ffi::Bytes* code,
                       ffi::Map<ffi::String, ffi::String>* source) {
  support::BytesInStream stream(bytes);
  TVM_FFI_ICHECK(stream.Read(format));
  bool versioned = *format == "cuda.module.v1";
  if (versioned) TVM_FFI_ICHECK(stream.Read(format));
  TVM_FFI_ICHECK(stream.Read(functions));
  TVM_FFI_ICHECK(stream.Read(code));
  if (versioned) TVM_FFI_ICHECK(stream.Read(source));
  // Old compiled binaries remain loadable. Old source payloads fail explicitly
  // in CompileSource because they cannot be reproduced without their options.
}

}  // namespace cuda
}  // namespace backend
}  // namespace tvm
#endif  // TVM_BACKEND_CUDA_MODULE_METADATA_H_
