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

#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>

#include "src/runtime/c_runtime_api.cc"
#include "src/runtime/container.cc"
#include "src/runtime/contrib/sort/sort.cc"
#include "src/runtime/cpu_device_api.cc"
#include "src/runtime/file_utils.cc"
#include "src/runtime/library_module.cc"
#include "src/runtime/logging.cc"
#include "src/runtime/module.cc"
#include "src/runtime/ndarray.cc"
#include "src/runtime/object.cc"
#include "src/runtime/profiling.cc"
#include "src/runtime/registry.cc"
#include "src/runtime/rpc/rpc_channel.cc"
#include "src/runtime/rpc/rpc_endpoint.cc"
#include "src/runtime/rpc/rpc_event_impl.cc"
#include "src/runtime/rpc/rpc_local_session.cc"
#include "src/runtime/rpc/rpc_module.cc"
#include "src/runtime/rpc/rpc_session.cc"
#include "src/runtime/system_library.cc"
#include "src/runtime/workspace_pool.cc"
// relax setup
#include "src/runtime/memory/memory_manager.cc"
#include "src/runtime/nvtx.cc"
#include "src/runtime/relax_vm/builtin.cc"
#include "src/runtime/relax_vm/bytecode.cc"
#include "src/runtime/relax_vm/executable.cc"
#include "src/runtime/relax_vm/kv_state.cc"
#include "src/runtime/relax_vm/lm_support.cc"
#include "src/runtime/relax_vm/ndarray_cache_support.cc"
#include "src/runtime/relax_vm/paged_kv_cache.cc"
#include "src/runtime/relax_vm/rnn_state.cc"
#include "src/runtime/relax_vm/vm.cc"

// --- Implementations of backend and wasm runtime API. ---

int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task) {
  TVMParallelGroupEnv env;
  env.num_task = 1;
  flambda(0, &env, cdata);
  return 0;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) { return 0; }

// --- Environment PackedFuncs for testing ---
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

TVM_REGISTER_GLOBAL("testing.echo").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = args[0];
});

TVM_REGISTER_GLOBAL("testing.call").set_body([](TVMArgs args, TVMRetValue* ret) {
  (args[0].operator PackedFunc())
      .CallPacked(TVMArgs(args.values + 1, args.type_codes + 1, args.num_args - 1), ret);
});

TVM_REGISTER_GLOBAL("testing.ret_string").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = args[0].operator String();
});

TVM_REGISTER_GLOBAL("testing.log_info_str").set_body([](TVMArgs args, TVMRetValue* ret) {
  LOG(INFO) << args[0].operator String();
});

TVM_REGISTER_GLOBAL("testing.log_fatal_str").set_body([](TVMArgs args, TVMRetValue* ret) {
  LOG(FATAL) << args[0].operator String();
});

TVM_REGISTER_GLOBAL("testing.add_one").set_body_typed([](int x) { return x + 1; });

TVM_REGISTER_GLOBAL("testing.wrap_callback").set_body([](TVMArgs args, TVMRetValue* ret) {
  PackedFunc pf = args[0];
  *ret = runtime::TypedPackedFunc<void()>([pf]() { pf(); });
});

// internal function used for debug and testing purposes
TVM_REGISTER_GLOBAL("testing.object_use_count").set_body([](TVMArgs args, TVMRetValue* ret) {
  runtime::ObjectRef obj = args[0];
  // substract the current one because we always copy
  // and get another value.
  *ret = (obj.use_count() - 1);
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

TVM_REGISTER_GLOBAL("tvmjs.array.decode_storage").set_body_typed(ArrayDecodeStorage);

// Concatenate n TVMArrays
TVM_REGISTER_GLOBAL("tvmjs.runtime.ArrayConcat").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::vector<ObjectRef> data;
  for (int i = 0; i < args.size(); ++i) {
    // Get i-th TVMArray
    ICHECK_EQ(args[i].type_code(), kTVMObjectHandle);
    Object* ptr = static_cast<Object*>(args[i].value().v_handle);
    ICHECK(ptr->IsInstance<ArrayNode>());
    auto* arr_i = static_cast<const ArrayNode*>(ptr);
    for (size_t j = 0; j < arr_i->size(); ++j) {
      // Push back each j-th element of the i-th array
      data.push_back(arr_i->at(j));
    }
  }
  *ret = Array<ObjectRef>(data);
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
TVM_REGISTER_GLOBAL("tvmjs.runtime.ConcatEmbeddings").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::vector<NDArray> embeddings;
  for (int i = 0; i < args.size(); ++i) {
    ICHECK_EQ(args[i].type_code(), kTVMNDArrayHandle);
    embeddings.push_back(args[i]);
  }
  NDArray result = ConcatEmbeddings(std::move(embeddings));
  *ret = result;
});

}  // namespace runtime
}  // namespace tvm
