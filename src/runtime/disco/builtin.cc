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
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/vm/vm.h>

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

Module LoadVMModule(std::string path, Optional<Device> device) {
  static DSOLibraryCache cache;
  Module dso_mod = cache.Open(path);
  Device dev = UseDefaultDeviceIfNone(device);
  ffi::Function vm_load_executable = dso_mod.GetFunction("vm_load_executable");
  if (vm_load_executable == nullptr) {
    // not built by RelaxVM, return the dso_mod directly
    return dso_mod;
  }
  auto mod = vm_load_executable().cast<Module>();
  ffi::Function vm_initialization = mod.GetFunction("vm_initialization");
  CHECK(vm_initialization != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_initialization` does not exist";
  vm_initialization(static_cast<int>(dev.device_type), static_cast<int>(dev.device_id),
                    static_cast<int>(AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
                    static_cast<int>(AllocatorType::kPooled));
  return mod;
}

NDArray DiscoEmptyNDArray(ffi::Shape shape, DataType dtype, Optional<Device> device) {
  return NDArray::Empty(shape, dtype, UseDefaultDeviceIfNone(device));
}

ffi::Function GetCCLFunc(const char* name) {
  std::string ccl = DiscoWorker::ThreadLocal()->ccl;
  std::string pf_name = "runtime.disco." + ccl + "." + name;
  const auto pf = tvm::ffi::Function::GetGlobal(pf_name);
  CHECK(pf.has_value()) << "ValueError: Cannot find the `" << name << "` function for `" << ccl
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

TVM_FFI_REGISTER_GLOBAL("runtime.disco.load_vm_module").set_body_typed(LoadVMModule);

TVM_FFI_REGISTER_GLOBAL("runtime.disco.empty")
    .set_body_typed([](ffi::Shape shape, DataType dtype, Optional<Device> device, bool worker0_only,
                       bool in_group) -> Optional<NDArray> {
      int worker_id = WorkerId();
      int group_size =
          DiscoWorker::ThreadLocal()->num_workers / DiscoWorker::ThreadLocal()->num_groups;
      bool is_worker0 = (worker_id == 0 && !in_group) || (in_group && worker_id % group_size == 0);
      if (worker0_only && !is_worker0) {
        return std::nullopt;
      } else {
        return DiscoEmptyNDArray(shape, dtype, device);
      }
    });

TVM_FFI_REGISTER_GLOBAL("runtime.disco.allreduce")
    .set_body_typed([](NDArray send, ffi::Shape reduce_kind, bool in_group, NDArray recv) {
      int kind = IntegerFromShape(reduce_kind);
      CHECK(0 <= kind && kind <= 4) << "ValueError: Unknown ReduceKind: " << kind;
      AllReduce(send, static_cast<ReduceKind>(kind), in_group, recv);
    });
TVM_FFI_REGISTER_GLOBAL("runtime.disco.allgather").set_body_typed(AllGather);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.broadcast_from_worker0")
    .set_body_typed(BroadcastFromWorker0);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.scatter_from_worker0").set_body_typed(ScatterFromWorker0);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.gather_to_worker0").set_body_typed(GatherToWorker0);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.recv_from_worker0").set_body_typed(RecvFromWorker0);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.send_to_next_group").set_body_typed(SendToNextGroup);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.recv_from_prev_group").set_body_typed(RecvFromPrevGroup);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.send_to_worker").set_body_typed(SendToWorker);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.recv_from_worker").set_body_typed(RecvFromWorker);
TVM_FFI_REGISTER_GLOBAL("runtime.disco.worker_id").set_body_typed([]() -> ffi::Shape {
  return ffi::Shape({WorkerId()});
});
TVM_FFI_REGISTER_GLOBAL("runtime.disco.worker_rank").set_body_typed([]() -> int64_t {
  return WorkerId();
});
TVM_FFI_REGISTER_GLOBAL("runtime.disco.device").set_body_typed([]() -> Device {
  return DiscoWorker::ThreadLocal()->default_device;
});
TVM_FFI_REGISTER_GLOBAL("runtime.disco.bind_worker_to_cpu_core")
    .set_body_typed([](ffi::Shape cpu_ids) {
      int worker_id = WorkerId();
      ICHECK_LT(worker_id, static_cast<int>(cpu_ids.size()));
      const auto f_set_thread_affinity = tvm::ffi::Function::GetGlobalRequired(
          "tvm.runtime.threading.set_current_thread_affinity");
      f_set_thread_affinity(ffi::Shape{cpu_ids[worker_id]});
    });

}  // namespace runtime
}  // namespace tvm
