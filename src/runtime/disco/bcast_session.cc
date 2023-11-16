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
#include "./bcast_session.h"

#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <sstream>

namespace tvm {
namespace runtime {

struct BcastSessionObj::Internal {
  template <typename... Args>
  static void TVM_ALWAYS_INLINE BroadcastUnpacked(BcastSessionObj* self, DiscoAction action,
                                                  int64_t reg_id, Args&&... args) {
    constexpr int kNumArgs = 2 + sizeof...(Args);
    TVMValue values[kNumArgs];
    int type_codes[kNumArgs];
    PackArgs(values, type_codes, static_cast<int>(action), reg_id, std::forward<Args>(args)...);
    self->BroadcastPacked(TVMArgs(values, type_codes, kNumArgs));
  }

  static DRef MakeDRef(int reg_id, Session session) {
    ObjectPtr<DRefObj> p = make_object<DRefObj>();
    p->reg_id = reg_id;
    p->session = session;
    return DRef(std::move(p));
  }
};

DRef BcastSessionObj::GetGlobalFunc(const std::string& name) {
  int reg_id = AllocateReg();
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kGetGlobalFunc, reg_id, name);
  return BcastSessionObj::Internal::MakeDRef(reg_id, GetRef<Session>(this));
}

void BcastSessionObj::CopyFromWorker0(const NDArray& host_array, const DRef& remote_array) {
  this->AppendHostNDArray(host_array);
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kCopyFromWorker0,
                                               remote_array->reg_id);
}

void BcastSessionObj::CopyToWorker0(const NDArray& host_array, const DRef& remote_array) {
  this->AppendHostNDArray(host_array);
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kCopyToWorker0,
                                               remote_array->reg_id);
}

void BcastSessionObj::Shutdown() {
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kShutDown, 0);
}

void BcastSessionObj::InitCCL(String ccl, IntTuple device_ids) {
  const auto* pf = runtime::Registry::Get("runtime.disco." + ccl + ".init_ccl");
  CHECK(pf) << "ValueError: Cannot initialize CCL `" << ccl
            << "`, because cannot find function: runtime.disco." << ccl << ".init_ccl";
  (*pf)(GetRef<Session>(this), device_ids);
}

void BcastSessionObj::SyncWorker(int worker_id) {
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kSyncWorker, worker_id);
  TVMArgs args = this->RecvReplyPacked(worker_id);
  ICHECK_EQ(args.size(), 2);
  DiscoAction action = static_cast<DiscoAction>(args[0].operator int());
  int ret_worker_id = args[1];
  ICHECK(action == DiscoAction::kSyncWorker);
  ICHECK_EQ(ret_worker_id, worker_id);
}

DRef BcastSessionObj::CallWithPacked(const TVMArgs& args) {
  TVMValue* values = const_cast<TVMValue*>(args.values);
  int* type_codes = const_cast<int*>(args.type_codes);
  int num_args = args.num_args;
  int reg_id = AllocateReg();
  {
    TVMArgsSetter setter(values, type_codes);
    DRef func = args[2];
    setter(0, static_cast<int>(DiscoAction::kCallPacked));
    setter(1, reg_id);
    setter(2, func->reg_id);
  }
  {
    std::ostringstream os;
    int cnt = 0;
    for (int i = 3; i < num_args; ++i) {
      int type_code = type_codes[i];
      if (type_code != kDLInt && type_code != kDLUInt && type_code != kDLFloat &&
          type_code != kTVMDataType && type_code != kDLDevice && type_code != kTVMOpaqueHandle &&
          type_code != kTVMStr && type_code != kTVMNullptr && type_code != kTVMBytes &&
          type_code != kTVMObjectHandle) {
        os << "\n  Argument #" << i << " has unsupported type code: " << type_code << " ("
           << ArgTypeCode2Str(type_code) << ")";
        cnt += 1;
      }
    }
    if (cnt > 0) {
      LOG(FATAL) << "CallWithPacked() does not support " << cnt << " argument(s):" << os.str();
    }
  }
  this->BroadcastPacked(TVMArgs(values, type_codes, num_args));
  return BcastSessionObj::Internal::MakeDRef(reg_id, GetRef<Session>(this));
}

void BcastSessionObj::DeallocReg(int reg_id) {
  BcastSessionObj::Internal::BroadcastUnpacked(this, DiscoAction::kKillReg, reg_id);
  this->free_regs_.push_back(reg_id);
}

int BcastSessionObj::AllocateReg() {
  if (this->free_regs_.empty()) {
    return this->reg_count_++;
  }
  int reg_id = this->free_regs_.back();
  this->free_regs_.pop_back();
  return reg_id;
}

void BcastSessionObj::AppendHostNDArray(const NDArray& host_array) {
  std::lock_guard<std::mutex> lock(worker_zero_data_.queue_mutex_);
  worker_zero_data_.host_arrays.push(host_array);
}

}  // namespace runtime
}  // namespace tvm
