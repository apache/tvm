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

/*
 * \file wasm_runtime.cc
 * \brief TVM wasm runtime library pack.
 */

// configurations for tvm logging
#define TVM_LOG_STACK_TRACE 0
#define TVM_LOG_DEBUG 0
#define TVM_LOG_CUSTOMIZE 1
#define TVM_FFI_USE_LIBBACKTRACE 0
#define TVM_FFI_ALWAYS_LOG_BEFORE_THROW 1
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/ffi/any.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include "src/runtime/contrib/sort/sort.cc"
#include "src/runtime/cpu_device_api.cc"
#include "src/runtime/device_api.cc"
#include "src/runtime/file_utils.cc"
#include "src/runtime/logging.cc"
#include "src/runtime/profiling.cc"
#include "src/runtime/rpc/rpc_channel.cc"
#include "src/runtime/rpc/rpc_endpoint.cc"
#include "src/runtime/rpc/rpc_event_impl.cc"
#include "src/runtime/rpc/rpc_local_session.cc"
#include "src/runtime/rpc/rpc_module.cc"
#include "src/runtime/rpc/rpc_session.cc"
#include "src/runtime/tensor.cc"
#include "src/runtime/workspace_pool.cc"
// relax setup
#include "3rdparty/tvm-ffi/src/ffi/backtrace.cc"
#include "3rdparty/tvm-ffi/src/ffi/container.cc"
#include "3rdparty/tvm-ffi/src/ffi/dtype.cc"
#include "3rdparty/tvm-ffi/src/ffi/error.cc"
#include "3rdparty/tvm-ffi/src/ffi/extra/env_c_api.cc"
#include "3rdparty/tvm-ffi/src/ffi/extra/env_context.cc"
#include "3rdparty/tvm-ffi/src/ffi/extra/library_module.cc"
#include "3rdparty/tvm-ffi/src/ffi/extra/library_module_system_lib.cc"
#include "3rdparty/tvm-ffi/src/ffi/extra/module.cc"
#include "3rdparty/tvm-ffi/src/ffi/function.cc"
#include "3rdparty/tvm-ffi/src/ffi/object.cc"
#include "3rdparty/tvm-ffi/src/ffi/tensor.cc"
#include "3rdparty/tvm-ffi/src/ffi/testing/testing.cc"
#include "src/runtime/memory/memory_manager.cc"
#include "src/runtime/nvtx.cc"
#include "src/runtime/vm/attn_backend.cc"
#include "src/runtime/vm/builtin.cc"
#include "src/runtime/vm/bytecode.cc"
#include "src/runtime/vm/executable.cc"
#include "src/runtime/vm/kv_state.cc"
#include "src/runtime/vm/lm_support.cc"
#include "src/runtime/vm/paged_kv_cache.cc"
#include "src/runtime/vm/rnn_state.cc"
#include "src/runtime/vm/tensor_cache_support.cc"
#include "src/runtime/vm/vm.cc"

// --- Implementations of backend and wasm runtime API. ---

int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task) {
  TVMParallelGroupEnv env;
  env.num_task = 1;
  flambda(0, &env, cdata);
  return 0;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) { return 0; }

// --- Environment ffi::Functions for testing ---
namespace tvm {
namespace runtime {
namespace detail {
// Override logging mechanism
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  std::cerr << "[FATAL] " << file << ":" << lineno << ": " << message << std::endl;
  abort();
}

void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  static const char* level_strings_[] = {
      "[DEBUG] ",
      "[INFO] ",
      "[WARNING] ",
      "[ERROR] ",
  };
  std::cout << level_strings_[level] << file << ":" << lineno << ": " << message << std::endl;
}

}  // namespace detail

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvmjs.testing.call",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    (args[0].cast<ffi::Function>()).CallPacked(args.Slice(1), ret);
                  })
      .def_packed(
          "tvmjs.testing.log_info_str",
          [](ffi::PackedArgs args, ffi::Any* ret) { LOG(INFO) << args[0].cast<ffi::String>(); })
      .def("tvmjs.testing.add_one", [](int x) { return x + 1; })
      .def_packed("tvmjs.testing.wrap_callback", [](ffi::PackedArgs args, ffi::Any* ret) {
        ffi::Function pf = args[0].cast<ffi::Function>();
        *ret = ffi::TypedFunction<void()>([pf]() { pf(); });
      });
}

void ArrayDecodeStorage(Tensor cpu_arr, TVMFFIByteArray* bytes, const std::string& format,
                        const std::string& dtype) {
  ICHECK_NE(bytes, nullptr);
  const char* byte_data = bytes->data;
  const size_t byte_size = bytes->size;
  if (format == "f32-to-bf16" && dtype == "float32") {
    const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(byte_data);
    uint32_t* data = static_cast<uint32_t*>(cpu_arr->data);
    ICHECK(cpu_arr.IsContiguous());
    size_t size = 1;
    for (int i = 0; i < cpu_arr->ndim; ++i) {
      size *= cpu_arr->shape[i];
    }
    ICHECK_EQ(size, byte_size / 2);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<uint32_t>(bf16[i]) << 16;
    }
  } else {
    cpu_arr.CopyFromBytes(byte_data, byte_size);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "tvmjs.array.decode_storage", [](ffi::PackedArgs args, ffi::Any* ret) {
        Tensor cpu_arr = args[0].cast<Tensor>();
        TVMFFIByteArray* bytes = args[1].cast<TVMFFIByteArray*>();
        std::string format = args[2].cast<ffi::String>().operator std::string();
        std::string dtype = args[3].cast<ffi::String>().operator std::string();
        ArrayDecodeStorage(cpu_arr, bytes, format, dtype);
      });
}

// Concatenate n TVMArrays
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tvmjs.runtime.ArrayConcat",
                               [](ffi::PackedArgs args, ffi::Any* ret) {
                                 std::vector<Any> data;
                                 for (int i = 0; i < args.size(); ++i) {
                                   // Get i-th TVMArray
                                   auto* arr_i = args[i].as<ffi::ArrayObj>();
                                   ICHECK(arr_i != nullptr);
                                   for (size_t j = 0; j < arr_i->size(); ++j) {
                                     // Push back each j-th element of the i-th array
                                     data.push_back(arr_i->at(j));
                                   }
                                 }
                                 *ret = ffi::Array<Any>(data);
                               });
}

Tensor ConcatEmbeddings(const std::vector<Tensor>& embeddings) {
  // Get output shape
  int64_t hidden_size = embeddings[0]->shape[1];
  DLDataType dtype = embeddings[0]->dtype;
  DLDevice device = embeddings[0]->device;
  int seqLen = 0;
  for (int i = 0; i < embeddings.size(); ++i) {
    ICHECK_EQ(embeddings[i]->ndim, 2);
    ICHECK_EQ(embeddings[i]->shape[1], hidden_size);
    seqLen += embeddings[i]->shape[0];
  }

  // Create output
  std::vector<int64_t> shape;
  shape.push_back(seqLen);
  shape.push_back(hidden_size);
  Tensor result = Tensor::Empty(shape, dtype, device);

  // Copy
  int offset = 0;
  for (int i = 0; i < embeddings.size(); i++) {
    const DLTensor& copy_src = *(embeddings[i].operator->());
    const DLTensor* p_copy_dst = result.operator->();
    DLTensor copy_dst = *p_copy_dst;
    copy_dst.shape = embeddings[i]->shape;
    copy_dst.byte_offset =
        offset * hidden_size * ((embeddings[i]->dtype.bits * embeddings[i]->dtype.lanes + 7) / 8);
    Tensor::CopyFromTo(&copy_src, &copy_dst);
    offset += embeddings[i]->shape[0];
  }

  return result;
}

// Concatenate n Tensors
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvmjs.runtime.ConcatEmbeddings",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    std::vector<Tensor> embeddings;
                    for (int i = 0; i < args.size(); ++i) {
                      embeddings.push_back(args[i].cast<Tensor>());
                    }
                    Tensor result = ConcatEmbeddings(std::move(embeddings));
                    *ret = result;
                  })
      .def("tvmjs.runtime.TensorCopyFromBytes",
           [](Tensor nd, TVMFFIByteArray* bytes) { nd.CopyFromBytes(bytes->data, bytes->size); })
      .def("tvmjs.runtime.TensorCopyToBytes", [](Tensor nd) -> ffi::Bytes {
        size_t size = ffi::GetDataSize(*(nd.operator->()));
        std::string bytes;
        bytes.resize(size);
        nd.CopyToBytes(bytes.data(), size);
        return ffi::Bytes(bytes);
      });
}

}  // namespace runtime
}  // namespace tvm
