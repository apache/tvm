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

struct ThreadLocalDiscoWorker {
  DiscoWorker* worker;

  static ThreadLocalDiscoWorker* Get() {
    thread_local static ThreadLocalDiscoWorker worker;
    return &worker;
  }
};

TVM_DLL DiscoWorker* DiscoWorker::ThreadLocal() {
  DiscoWorker* ret = ThreadLocalDiscoWorker::Get()->worker;
  CHECK(ret) << "ValueError: The current thread is not a DiscoWorker thread";
  return ret;
}

void DiscoWorker::SetRegister(int reg_id, TVMArgValue value) {
  ICHECK(0 <= reg_id && reg_id < static_cast<int>(register_file.size()));
  TVMRetValue& rv = register_file.at(reg_id);
  if (rv.type_code() == kTVMNDArrayHandle && value.type_code() == kTVMNDArrayHandle) {
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
          PackedFunc func = GetReg(self, func_reg_id);
          CallPacked(self, reg_id, func,
                     TVMArgs(args.values + 3, args.type_codes + 3, args.num_args - 3));
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
          TVMArgValue value = args[3];
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
    if (self->worker_zero_data != nullptr) {
      NDArray tgt = GetNDArrayFromHost(self);
      NDArray src = GetReg(self, reg_id);
      tgt.CopyFrom(src);
    }
  }

  static void CopyToWorker0(DiscoWorker* self, int reg_id) {
    if (self->worker_zero_data != nullptr) {
      NDArray src = GetNDArrayFromHost(self);
      NDArray tgt = GetReg(self, reg_id);
      tgt.CopyFrom(src);
    }
  }

  static void SyncWorker(DiscoWorker* self, int worker_id) {
    if (worker_id == self->worker_id) {
      ::tvm::runtime::SyncWorker();
      TVMValue values[2];
      int type_codes[2];
      PackArgs(values, type_codes, static_cast<int>(DiscoAction::kSyncWorker), worker_id);
      self->channel->Reply(TVMArgs(values, type_codes, 2));
    }
  }

  static void DebugGetFromRemote(DiscoWorker* self, int reg_id, int worker_id) {
    if (worker_id == self->worker_id) {
      TVMRetValue rv = GetReg(self, reg_id);
      if (rv.type_code() == kTVMNDArrayHandle || rv.type_code() == kTVMObjectHandle) {
        rv = DiscoDebugObject::Wrap(rv);
      }
      TVMValue values[2];
      int type_codes[2];
      PackArgs(values, type_codes, static_cast<int>(DiscoAction::kDebugGetFromRemote), rv);
      self->channel->Reply(TVMArgs(values, type_codes, 2));
    }
  }

  static void DebugSetRegister(DiscoWorker* self, int reg_id, int worker_id, TVMArgValue value) {
    if (worker_id == self->worker_id) {
      ::tvm::runtime::SyncWorker();
      self->SetRegister(reg_id, value);
      TVMValue values[1];
      int type_codes[1];
      PackArgs(values, type_codes, static_cast<int>(DiscoAction::kDebugSetRegister));
      self->channel->Reply(TVMArgs(values, type_codes, 1));
    }
  }

  static void CallPacked(DiscoWorker* self, int64_t ret_reg_id, PackedFunc func,
                         const TVMArgs& args) {
    TVMValue* values = const_cast<TVMValue*>(args.values);
    int* type_codes = const_cast<int*>(args.type_codes);
    int num_args = args.num_args;
    TVMArgsSetter setter(values, type_codes);
    for (int i = 0; i < num_args; ++i) {
      TVMArgValue val = TVMArgValue(values[i], type_codes[i]);
      if (val.IsObjectRef<DRef>()) {
        DRef dref = val;
        setter(i, GetReg(self, dref->reg_id));
      }
    }
    TVMRetValue rv;
    func.CallPacked(TVMArgs(values, type_codes, num_args), &rv);
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
