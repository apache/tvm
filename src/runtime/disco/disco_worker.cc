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
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include "../../support/process_id.h"
#include "./protocol.h"

namespace tvm {
namespace runtime {

TVM_DLL DiscoWorker* DiscoWorker::ThreadLocal() {
  DiscoWorker* ret = ThreadLocalDiscoWorker::Get()->worker;
  CHECK(ret) << "ValueError: The current thread is not a DiscoWorker thread";
  return ret;
}

void DiscoWorker::SetRegister(int reg_id, AnyView value) {
  ICHECK(0 <= reg_id && reg_id < static_cast<int>(register_file.size()));
  TVMRetValue& rv = register_file.at(reg_id);
  if (rv.type_index() == ffi::TypeIndex::kTVMFFINDArray &&
      value.type_index() == ffi::TypeIndex::kTVMFFINDArray) {
    NDArray dst = rv;
    NDArray src = value;
    dst.CopyFrom(src);
  } else {
    rv = value;
  }
}

struct DiscoWorker::Impl {
  static void MainLoop(DiscoWorker* self) {
    ThreadLocalDiscoWorker::Get()->worker = self;
    using namespace tvm;
    while (true) {
      TVMArgs args = self->channel->Recv();
      DiscoAction action = static_cast<DiscoAction>(args[0].operator int());
      int64_t reg_id = args[1];
      switch (action) {
        case DiscoAction::kShutDown: {
          Shutdown(self);
          return;
        }
        case DiscoAction::kKillReg: {
          GetReg(self, reg_id) = nullptr;
          break;
        }
        case DiscoAction::kGetGlobalFunc: {
          GetGlobalFunc(self, reg_id, args[2]);
          break;
        }
        case DiscoAction::kCallPacked: {
          int func_reg_id = args[2];
          CHECK_LT(func_reg_id, self->register_file.size());
          PackedFunc func = GetReg(self, func_reg_id);
          CHECK(func.defined());
          CallPacked(self, reg_id, func, args.Slice(3));
          break;
        }
        case DiscoAction::kCopyFromWorker0: {
          CopyFromWorker0(self, reg_id);
          break;
        }
        case DiscoAction::kCopyToWorker0: {
          CopyToWorker0(self, reg_id);
          break;
        }
        case DiscoAction::kSyncWorker: {
          SyncWorker(self, reg_id);
          break;
        }
        case DiscoAction::kDebugGetFromRemote: {
          int worker_id = args[2];
          DebugGetFromRemote(self, reg_id, worker_id);
          break;
        }
        case DiscoAction::kDebugSetRegister: {
          int worker_id = args[2];
          AnyView value = args[3];
          DebugSetRegister(self, reg_id, worker_id, value);
          break;
        }
      }
    }
  }

  static void Shutdown(DiscoWorker* self) {}

  static void GetGlobalFunc(DiscoWorker* self, int reg_id, const std::string& name) {
    const PackedFunc* pf = runtime::Registry::Get(name);
    CHECK(pf) << "ValueError: Cannot find global function: " << name;
    if (reg_id != 0) {
      GetReg(self, reg_id) = *pf;
    }
  }

  static NDArray GetNDArrayFromHost(DiscoWorker* self) {
    std::lock_guard<std::mutex> lock(self->worker_zero_data->queue_mutex_);
    NDArray array = self->worker_zero_data->host_arrays.front();
    self->worker_zero_data->host_arrays.pop();
    return array;
  }

  static void CopyFromWorker0(DiscoWorker* self, int reg_id) {
    if (self->worker_id == 0) {
      NDArray tgt = GetNDArrayFromHost(self);
      NDArray src = GetReg(self, reg_id);
      tgt.CopyFrom(src);
    }
  }

  static void CopyToWorker0(DiscoWorker* self, int reg_id) {
    if (self->worker_id == 0) {
      NDArray src = GetNDArrayFromHost(self);
      NDArray tgt = GetReg(self, reg_id);
      tgt.CopyFrom(src);
    }
  }

  static void SyncWorker(DiscoWorker* self, int worker_id) {
    if (worker_id == self->worker_id) {
      ::tvm::runtime::SyncWorker();
      AnyView packed_args[2];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoAction::kSyncWorker), worker_id);
      self->channel->Reply(ffi::PackedArgs(packed_args, 2));
    }
  }

  static void DebugGetFromRemote(DiscoWorker* self, int reg_id, int worker_id) {
    if (worker_id == self->worker_id) {
      TVMRetValue rv = GetReg(self, reg_id);
      if (rv.as<ObjectRef>()) {
        rv = DiscoDebugObject::Wrap(rv);
      }
      AnyView packed_args[2];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoAction::kDebugGetFromRemote), rv);
      self->channel->Reply(ffi::PackedArgs(packed_args, 2));
    }
  }

  static void DebugSetRegister(DiscoWorker* self, int reg_id, int worker_id, AnyView value) {
    if (worker_id == self->worker_id) {
      ::tvm::runtime::SyncWorker();
      self->SetRegister(reg_id, value);
      AnyView packed_args[1];
      ffi::PackedArgs::Fill(packed_args, static_cast<int>(DiscoAction::kDebugSetRegister));
      self->channel->Reply(ffi::PackedArgs(packed_args, 1));
    }
  }

  static void CallPacked(DiscoWorker* self, int64_t ret_reg_id, PackedFunc func,
                         const TVMArgs& args) {
    // NOTE: this action is not safe unless we know args is not
    // used else where in this case it is oK
    AnyView* args_vec = const_cast<AnyView*>(args.data());
    // translate args into remote calling convention
    for (int i = 0; i < args.size(); ++i) {
      if (auto opt_dref = args_vec[i].as<DRef>()) {
        DRef dref = opt_dref.value();
        args_vec[i] = GetReg(self, dref->reg_id);
      }
    }
    TVMRetValue rv;
    func.CallPacked(ffi::PackedArgs(args_vec, args.size()), &rv);
    GetReg(self, ret_reg_id) = std::move(rv);
  }

  static TVMRetValue& GetReg(DiscoWorker* self, int64_t reg_id) {
    if (reg_id >= static_cast<int64_t>(self->register_file.size())) {
      self->register_file.resize(reg_id + 1);
    }
    return self->register_file[reg_id];
  }
};

void DiscoWorker::MainLoop() { DiscoWorker::Impl::MainLoop(this); }

}  // namespace runtime
}  // namespace tvm
