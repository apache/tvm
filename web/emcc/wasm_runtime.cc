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
#include "src/runtime/relax_vm/builtin.cc"
#include "src/runtime/relax_vm/bytecode.cc"
#include "src/runtime/relax_vm/executable.cc"
#include "src/runtime/relax_vm/memory_manager.cc"
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

/*!
 * A NDArray cache to store pre-loaded arrays in the system.
 */
class NDArrayCache {
 public:
  static NDArrayCache* Global() {
    static NDArrayCache* inst = new NDArrayCache();
    return inst;
  }

  static void Update(String name, NDArray arr, bool override) {
    NDArrayCache* pool = Global();
    if (!override) {
      ICHECK_EQ(pool->pool_.count(name), 0) << "Name " << name << " already exists in the cache";
    }
    pool->pool_.Set(name, arr);
  }

  static Optional<NDArray> Get(String name) {
    NDArrayCache* pool = Global();
    auto it = pool->pool_.find(name);
    if (it != pool->pool_.end()) {
      return (*it).second;
    } else {
      return NullOpt;
    }
  }

  static void Remove(String name) {
    NDArrayCache* pool = Global();
    pool->pool_.erase(name);
  }

  static void Clear() { Global()->pool_.clear(); }

 private:
  Map<String, NDArray> pool_;
};

TVM_REGISTER_GLOBAL("tvmjs.ndarray_cache.get").set_body_typed(NDArrayCache::Get);
TVM_REGISTER_GLOBAL("tvmjs.ndarray_cache.update").set_body_typed(NDArrayCache::Update);
TVM_REGISTER_GLOBAL("tvmjs.ndarray_cache.remove").set_body_typed(NDArrayCache::Remove);
TVM_REGISTER_GLOBAL("tvmjs.ndarray_cache.clear").set_body_typed(NDArrayCache::Clear);

void ArrayDecodeStorage(NDArray cpu_arr, std::string bytes, std::string format) {
  if (format == "f32-to-bf16") {
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

class ParamModuleNode : public runtime::ModuleNode {
 public:
  const char* type_key() const final { return "param_module"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_params") {
      auto params = params_;
      return PackedFunc([params](TVMArgs args, TVMRetValue* rv) { *rv = params; });
    } else {
      return PackedFunc();
    }
  }

  static Module Create(std::string prefix, int num_params) {
    Array<NDArray> params;
    for (int i = 0; i < num_params; ++i) {
      std::string name = prefix + "_" + std::to_string(i);
      auto opt = NDArrayCache::Get(name);
      if (opt) {
        params.push_back(opt.value());
      } else {
        LOG(FATAL) << "Cannot find " << name << " in cache";
      }
    }
    auto n = make_object<ParamModuleNode>();
    n->params_ = params;
    return Module(n);
  }

 private:
  Array<NDArray> params_;
};

TVM_REGISTER_GLOBAL("tvmjs.param_module_from_cache").set_body_typed(ParamModuleNode::Create);
}  // namespace runtime
}  // namespace tvm
