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

#include <tvm/runtime/logging.h>

#include "src/runtime/contrib/sort/sort.cc"
#include "src/runtime/cpu_device_api.cc"
#include "src/runtime/device_api.cc"
#include "src/runtime/file_utils.cc"
#include "src/runtime/library_module.cc"
#include "src/runtime/logging.cc"
#include "src/runtime/module.cc"
#include "src/runtime/ndarray.cc"
#include "src/runtime/profiling.cc"
#include "src/runtime/rpc/rpc_channel.cc"
#include "src/runtime/rpc/rpc_endpoint.cc"
#include "src/runtime/rpc/rpc_event_impl.cc"
#include "src/runtime/rpc/rpc_local_session.cc"
#include "src/runtime/rpc/rpc_module.cc"
#include "src/runtime/rpc/rpc_session.cc"
#include "src/runtime/system_library.cc"
#include "src/runtime/workspace_pool.cc"
// relax setup
#include "ffi/src/ffi/container.cc"
#include "ffi/src/ffi/dtype.cc"
#include "ffi/src/ffi/error.cc"
#include "ffi/src/ffi/function.cc"
#include "ffi/src/ffi/ndarray.cc"
#include "ffi/src/ffi/object.cc"
#include "ffi/src/ffi/testing.cc"
#include "ffi/src/ffi/traceback.cc"
#include "src/runtime/memory/memory_manager.cc"
#include "src/runtime/nvtx.cc"
#include "src/runtime/vm/attn_backend.cc"
#include "src/runtime/vm/builtin.cc"
#include "src/runtime/vm/bytecode.cc"
#include "src/runtime/vm/executable.cc"
#include "src/runtime/vm/kv_state.cc"
#include "src/runtime/vm/lm_support.cc"
#include "src/runtime/vm/ndarray_cache_support.cc"
#include "src/runtime/vm/paged_kv_cache.cc"
#include "src/runtime/vm/rnn_state.cc"
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

TVM_FFI_REGISTER_GLOBAL("tvmjs.testing.call")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      (args[0].cast<ffi::Function>()).CallPacked(args.Slice(1), ret);
    });

TVM_FFI_REGISTER_GLOBAL("tvmjs.testing.log_info_str")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      LOG(INFO) << args[0].cast<String>();
    });

TVM_FFI_REGISTER_GLOBAL("tvmjs.testing.add_one").set_body_typed([](int x) { return x + 1; });

TVM_FFI_REGISTER_GLOBAL("tvmjs.testing.wrap_callback")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      ffi::Function pf = args[0].cast<ffi::Function>();
      *ret = ffi::TypedFunction<void()>([pf]() { pf(); });
    });

void ArrayDecodeStorage(NDArray cpu_arr, std::string bytes, std::string format, std::string dtype) {
  if (format == "f32-to-bf16" && dtype == "float32") {
    std::vector<uint16_t> buffer(bytes.length() / 2);
    std::memcpy(buffer.data(), bytes.data(), buffer.size() * 2);
    // decode bf16 to f32
    const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(buffer.data());
    uint32_t* data = static_cast<uint32_t*>(cpu_arr->data);
    ICHECK(cpu_arr.IsContiguous());
    size_t size = 1;
    for (int i = 0; i < cpu_arr->ndim; ++i) {
      size *= cpu_arr->shape[i];
    }
    ICHECK_EQ(size, bytes.length() / 2);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<uint32_t>(bf16[i]) << 16;
    }
  } else {
    cpu_arr.CopyFromBytes(bytes.data(), bytes.length());
  }
}

TVM_FFI_REGISTER_GLOBAL("tvmjs.array.decode_storage").set_body_typed(ArrayDecodeStorage);

// Concatenate n TVMArrays
TVM_FFI_REGISTER_GLOBAL("tvmjs.runtime.ArrayConcat")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
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
      *ret = Array<Any>(data);
    });

NDArray ConcatEmbeddings(const std::vector<NDArray>& embeddings) {
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
  NDArray result = NDArray::Empty(shape, dtype, device);

  // Copy
  int offset = 0;
  for (int i = 0; i < embeddings.size(); i++) {
    const DLTensor& copy_src = *(embeddings[i].operator->());
    const DLTensor* p_copy_dst = result.operator->();
    DLTensor copy_dst = *p_copy_dst;
    copy_dst.shape = embeddings[i]->shape;
    copy_dst.byte_offset =
        offset * hidden_size * ((embeddings[i]->dtype.bits * embeddings[i]->dtype.lanes + 7) / 8);
    NDArray::CopyFromTo(&copy_src, &copy_dst);
    offset += embeddings[i]->shape[0];
  }

  return result;
}

// Concatenate n NDArrays
TVM_FFI_REGISTER_GLOBAL("tvmjs.runtime.ConcatEmbeddings")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      std::vector<NDArray> embeddings;
      for (int i = 0; i < args.size(); ++i) {
        embeddings.push_back(args[i].cast<NDArray>());
      }
      NDArray result = ConcatEmbeddings(std::move(embeddings));
      *ret = result;
    });

TVM_FFI_REGISTER_GLOBAL("tvmjs.runtime.NDArrayCopyFromBytes")
    .set_body_typed([](NDArray nd, TVMFFIByteArray* bytes) {
      nd.CopyFromBytes(bytes->data, bytes->size);
    });

TVM_FFI_REGISTER_GLOBAL("tvmjs.runtime.NDArrayCopyToBytes")
    .set_body_typed([](NDArray nd) -> ffi::Bytes {
      size_t size = GetDataSize(*(nd.operator->()));
      std::string bytes;
      bytes.resize(size);
      nd.CopyToBytes(bytes.data(), size);
      return ffi::Bytes(bytes);
    });

}  // namespace runtime
}  // namespace tvm
