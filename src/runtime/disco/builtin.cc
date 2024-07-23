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

void AllReduce(NDArray send, ReduceKind reduce_kind, bool in_group, NDArray recv) {
  GetCCLFunc("allreduce")(send, static_cast<int>(reduce_kind), in_group, recv);
}

void AllGather(NDArray send, bool in_group, NDArray recv) {
  GetCCLFunc("allgather")(send, in_group, recv);
}

TVM_DLL void BroadcastFromWorker0(NDArray send, bool in_group, NDArray recv) {
  GetCCLFunc("broadcast_from_worker0")(send, in_group, recv);
}

TVM_DLL void ScatterFromWorker0(Optional<NDArray> send, bool in_group, NDArray recv) {
  GetCCLFunc("scatter_from_worker0")(send, in_group, recv);
}

void GatherToWorker0(NDArray send, bool in_group, Optional<NDArray> recv) {
  GetCCLFunc("gather_to_worker0")(send, in_group, recv);
}

void RecvFromWorker0(NDArray buffer) { GetCCLFunc("recv_from_worker0")(buffer); }

void SendToNextGroup(NDArray buffer) { GetCCLFunc("send_to_next_group")(buffer); }

void RecvFromPrevGroup(NDArray buffer) { GetCCLFunc("recv_from_prev_group")(buffer); }

void SendToWorker(NDArray buffer, int receiver_id) {
  GetCCLFunc("send_to_worker")(buffer, receiver_id);
}

void RecvFromWorker(NDArray buffer, int sender_id) {
  GetCCLFunc("recv_from_worker")(buffer, sender_id);
}

int WorkerId() { return DiscoWorker::ThreadLocal()->worker_id; }

void SyncWorker() {
  if (DiscoWorker::ThreadLocal()->ccl != "") {
    GetCCLFunc("sync_worker")();
  }
}

TVM_REGISTER_GLOBAL("runtime.disco.load_vm_module").set_body_typed(LoadVMModule);

TVM_REGISTER_GLOBAL("runtime.disco.empty")
    .set_body_typed([](ShapeTuple shape, DataType dtype, Device device, bool worker0_only,
                       bool in_group) -> Optional<NDArray> {
      int worker_id = WorkerId();
      int group_size =
          DiscoWorker::ThreadLocal()->num_workers / DiscoWorker::ThreadLocal()->num_groups;
      bool is_worker0 = (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);
      if (worker0_only && !is_worker0) {
        return NullOpt;
      } else {
        return DiscoEmptyNDArray(shape, dtype, device);
      }
    });

TVM_REGISTER_GLOBAL("runtime.disco.allreduce")
    .set_body_typed([](NDArray send, ShapeTuple reduce_kind, bool in_group, NDArray recv) {
      int kind = IntegerFromShapeTuple(reduce_kind);
      CHECK(0 <= kind && kind <= 4) << "ValueError: Unknown ReduceKind: " << kind;
      AllReduce(send, static_cast<ReduceKind>(kind), in_group, recv);
    });
TVM_REGISTER_GLOBAL("runtime.disco.allgather").set_body_typed(AllGather);
TVM_REGISTER_GLOBAL("runtime.disco.broadcast_from_worker0").set_body_typed(BroadcastFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.scatter_from_worker0").set_body_typed(ScatterFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.gather_to_worker0").set_body_typed(GatherToWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.recv_from_worker0").set_body_typed(RecvFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.send_to_next_group").set_body_typed(SendToNextGroup);
TVM_REGISTER_GLOBAL("runtime.disco.recv_from_prev_group").set_body_typed(RecvFromPrevGroup);
TVM_REGISTER_GLOBAL("runtime.disco.send_to_worker").set_body_typed(SendToWorker);
TVM_REGISTER_GLOBAL("runtime.disco.recv_from_worker").set_body_typed(RecvFromWorker);
TVM_REGISTER_GLOBAL("runtime.disco.worker_id").set_body_typed([]() -> ShapeTuple {
  return ShapeTuple({WorkerId()});
});
TVM_REGISTER_GLOBAL("runtime.disco.worker_rank").set_body_typed([]() -> int64_t {
  return WorkerId();
});
TVM_REGISTER_GLOBAL("runtime.disco.device").set_body_typed([]() -> Device {
  return DiscoWorker::ThreadLocal()->default_device;
});
TVM_REGISTER_GLOBAL("runtime.disco.bind_worker_to_cpu_core").set_body_typed([](IntTuple cpu_ids) {
  int worker_id = WorkerId();
  ICHECK_LT(worker_id, static_cast<int>(cpu_ids.size()));
  const PackedFunc* f_set_thread_affinity =
      Registry::Get("tvm.runtime.threading.set_current_thread_affinity");
  ICHECK_NOTNULL(f_set_thread_affinity);
  (*f_set_thread_affinity)(IntTuple{cpu_ids[worker_id]});
});

}  // namespace runtime
}  // namespace tvm
