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
#include <dlpack/dlpack.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/vm.h>

#include <sstream>

#include "./utils.h"

namespace tvm {
namespace runtime {

class DSOLibraryCache {
 public:
  Module Open(const std::string& library_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    Module& lib = cache_[library_path];
    if (!lib.defined()) {
      lib = Module::LoadFromFile(library_path, "");
    }
    return lib;
  }

  std::unordered_map<std::string, Module> cache_;
  std::mutex mutex_;
};

Module LoadVMModule(std::string path, Device device) {
  static DSOLibraryCache cache;
  Module dso_mod = cache.Open(path);
  device = UseDefaultDeviceIfNone(device);
  PackedFunc vm_load_executable = dso_mod.GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_load_executable` does not exist";
  Module mod = vm_load_executable();
  PackedFunc vm_initialization = mod.GetFunction("vm_initialization");
  CHECK(vm_initialization != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_initialization` does not exist";
  vm_initialization(static_cast<int>(device.device_type), static_cast<int>(device.device_id),
                    static_cast<int>(AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
                    static_cast<int>(AllocatorType::kPooled));
  return mod;
}

NDArray DiscoEmptyNDArray(ShapeTuple shape, DataType dtype, Device device) {
  return NDArray::Empty(shape, dtype, UseDefaultDeviceIfNone(device));
}

const PackedFunc& GetCCLFunc(const char* name) {
  std::string ccl = DiscoWorker::ThreadLocal()->ccl;
  std::string pf_name = "runtime.disco." + ccl + "." + name;
  const PackedFunc* pf = tvm::runtime::Registry::Get(pf_name);
  CHECK(pf != nullptr) << "ValueError: Cannot find the `" << name << "` function for `" << ccl
                       << "` via `" << pf_name << "`";
  return *pf;
}

void AllReduce(NDArray send, ReduceKind reduce_kind, NDArray recv) {
  GetCCLFunc("allreduce")(send, static_cast<int>(reduce_kind), recv);
}

void AllGather(NDArray send, NDArray recv) { GetCCLFunc("allgather")(send, recv); }

TVM_DLL void BroadcastFromWorker0(NDArray send, NDArray recv) {
  GetCCLFunc("broadcast_from_worker0")(send, recv);
}

TVM_DLL void ScatterFromWorker0(Optional<NDArray> send, NDArray recv) {
  GetCCLFunc("scatter_from_worker0")(send, recv);
}

void GatherToWorker0(NDArray send, Optional<NDArray> recv) {
  GetCCLFunc("gather_to_worker0")(send, recv);
}

void RecvFromWorker0(NDArray buffer) { GetCCLFunc("recv_from_worker0")(buffer); }

int WorkerId() { return DiscoWorker::ThreadLocal()->worker_id; }

void SyncWorker() {
  if (DiscoWorker::ThreadLocal()->ccl != "") {
    GetCCLFunc("sync_worker")();
  }
}

TVM_REGISTER_GLOBAL("runtime.disco.load_vm_module").set_body_typed(LoadVMModule);
TVM_REGISTER_GLOBAL("runtime.disco.empty").set_body_typed(DiscoEmptyNDArray);
TVM_REGISTER_GLOBAL("runtime.disco.allreduce")
    .set_body_typed([](NDArray send, ShapeTuple reduce_kind, NDArray recv) {
      int kind = IntegerFromShapeTuple(reduce_kind);
      CHECK(0 <= kind && kind <= 4) << "ValueError: Unknown ReduceKind: " << kind;
      AllReduce(send, static_cast<ReduceKind>(kind), recv);
    });
TVM_REGISTER_GLOBAL("runtime.disco.allgather").set_body_typed(AllGather);
TVM_REGISTER_GLOBAL("runtime.disco.broadcast_from_worker0").set_body_typed(BroadcastFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.scatter_from_worker0").set_body_typed(ScatterFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.gather_to_worker0").set_body_typed(GatherToWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.recv_from_worker0").set_body_typed(RecvFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.worker_id").set_body_typed([]() -> ShapeTuple {
  return ShapeTuple({WorkerId()});
});

}  // namespace runtime
}  // namespace tvm
