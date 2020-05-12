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

/*!
 * \file rpc_session.cc
 * \brief RPC session for remote function call.
 */
#include "rpc_session.h"

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>

#include <array>
#include <mutex>

namespace tvm {
namespace runtime {

bool RPCSession::IsAsync() const { return false; }

void RPCSession::SendException(FAsyncCallback callback, const char* msg) {
  TVMValue value;
  value.v_str = msg;
  int32_t tcode = kTVMStr;
  callback(RPCCode::kException, TVMArgs(&value, &tcode, 1));
}

void RPCSession::AsyncCallFunc(PackedFuncHandle func, const TVMValue* arg_values,
                               const int* arg_type_codes, int num_args, FAsyncCallback callback) {
  try {
    this->CallFunc(func, arg_values, arg_type_codes, num_args,
                   [&callback](TVMArgs args) { callback(RPCCode::kReturn, args); });
  } catch (const std::runtime_error& e) {
    this->SendException(callback, e.what());
  }
}

void RPCSession::AsyncCopyToRemote(void* local_from, size_t local_from_offset, void* remote_to,
                                   size_t remote_to_offset, size_t nbytes, TVMContext remote_ctx_to,
                                   DLDataType type_hint, RPCSession::FAsyncCallback callback) {
  TVMValue value;
  int32_t tcode = kTVMNullptr;
  value.v_handle = nullptr;

  try {
    this->CopyToRemote(local_from, local_from_offset, remote_to, remote_to_offset, nbytes,
                       remote_ctx_to, type_hint);
    callback(RPCCode::kReturn, TVMArgs(&value, &tcode, 1));
  } catch (const std::runtime_error& e) {
    this->SendException(callback, e.what());
  }
}

void RPCSession::AsyncCopyFromRemote(void* remote_from, size_t remote_from_offset, void* local_to,
                                     size_t local_to_offset, size_t nbytes,
                                     TVMContext remote_ctx_from, DLDataType type_hint,
                                     RPCSession::FAsyncCallback callback) {
  TVMValue value;
  int32_t tcode = kTVMNullptr;
  value.v_handle = nullptr;

  try {
    this->CopyFromRemote(remote_from, remote_from_offset, local_to, local_to_offset, nbytes,
                         remote_ctx_from, type_hint);
    callback(RPCCode::kReturn, TVMArgs(&value, &tcode, 1));
  } catch (const std::runtime_error& e) {
    this->SendException(callback, e.what());
  }
}

void RPCSession::AsyncStreamWait(TVMContext ctx, TVMStreamHandle stream,
                                 RPCSession::FAsyncCallback callback) {
  TVMValue value;
  int32_t tcode = kTVMNullptr;
  value.v_handle = nullptr;

  try {
    this->GetDeviceAPI(ctx)->StreamSync(ctx, stream);
    callback(RPCCode::kReturn, TVMArgs(&value, &tcode, 1));
  } catch (const std::runtime_error& e) {
    this->SendException(callback, e.what());
  }
}

class RPCSessTable {
 public:
  static constexpr int kMaxRPCSession = 32;
  // Get global singleton
  static RPCSessTable* Global() {
    static RPCSessTable inst;
    return &inst;
  }
  // Get session from table
  std::shared_ptr<RPCSession> Get(int index) {
    CHECK(index >= 0 && index < kMaxRPCSession);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<RPCSession> ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kMaxRPCSession; ++i) {
      if (tbl_[i].lock() == nullptr) {
        tbl_[i] = ptr;
        return i;
      }
    }
    LOG(FATAL) << "maximum number of RPC session reached";
    return 0;
  }

 private:
  // The mutex
  std::mutex mutex_;
  // Use weak_ptr intentionally
  // If the RPCSession get released, the pointer session will be released
  std::array<std::weak_ptr<RPCSession>, kMaxRPCSession> tbl_;
};

std::shared_ptr<RPCSession> RPCSession::Get(int table_index) {
  return RPCSessTable::Global()->Get(table_index);
}

void RPCSession::InsertToSessionTable(std::shared_ptr<RPCSession> sess) {
  CHECK_EQ(sess->table_index_, 0);
  sess->table_index_ = RPCSessTable::Global()->Insert(sess);
}

}  // namespace runtime
}  // namespace tvm
