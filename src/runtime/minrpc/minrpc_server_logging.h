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

#ifndef TVM_RUNTIME_MINRPC_MINRPC_SERVER_LOGGING_H_
#define TVM_RUNTIME_MINRPC_MINRPC_SERVER_LOGGING_H_

#include <memory>
#include <utility>

#include "minrpc_logger.h"
#include "minrpc_server.h"

namespace tvm {
namespace runtime {

/*!
 * \brief A minimum RPC server that logs the received commands.
 *
 * \tparam TIOHandler IO provider to provide io handling.
 */
template <typename TIOHandler>
class MinRPCServerWithLog {
 public:
  explicit MinRPCServerWithLog(TIOHandler* io)
      : ret_handler_(io),
        ret_handler_wlog_(&ret_handler_, &logger_),
        exec_handler_(io, &ret_handler_wlog_),
        exec_handler_ptr_(new MinRPCExecuteWithLog(&exec_handler_, &logger_)),
        next_(io, std::move(exec_handler_ptr_)) {}

  bool ProcessOnePacket() { return next_.ProcessOnePacket(); }

 private:
  Logger logger_;
  MinRPCReturns<TIOHandler> ret_handler_;
  MinRPCExecute<TIOHandler> exec_handler_;
  MinRPCReturnsWithLog ret_handler_wlog_;
  std::unique_ptr<MinRPCExecuteWithLog> exec_handler_ptr_;
  MinRPCServer<TIOHandler> next_;
};

/*!
 * \brief A minimum RPC server that only logs the outgoing commands and received responses.
 * (Does not process the packets or respond to them.)
 *
 * \tparam TIOHandler IO provider to provide io handling.
 */
template <typename TIOHandler, template <typename> class Allocator = detail::PageAllocator>
class MinRPCSniffer {
 public:
  using PageAllocator = Allocator<TIOHandler>;
  explicit MinRPCSniffer(TIOHandler* io)
      : io_(io),
        arena_(PageAllocator(io_)),
        ret_handler_(io_),
        ret_handler_wlog_(&ret_handler_, &logger_),
        exec_handler_(&ret_handler_wlog_),
        exec_handler_ptr_(new MinRPCExecuteWithLog(&exec_handler_, &logger_)),
        next_(io_, std::move(exec_handler_ptr_)) {}

  bool ProcessOnePacket() { return next_.ProcessOnePacket(); }

  void ProcessOneResponse() {
    RPCCode code;
    uint64_t packet_len = 0;

    if (!Read(&packet_len)) return;
    if (packet_len == 0) {
      OutputLog();
      return;
    }
    if (!Read(&code)) return;
    switch (code) {
      case RPCCode::kReturn: {
        int32_t num_args;
        int* type_codes;
        TVMValue* values;
        RPCReference::RecvPackedSeq(&values, &type_codes, &num_args, this);
        ret_handler_wlog_.ReturnPackedSeq(values, type_codes, num_args);
        break;
      }
      case RPCCode::kException: {
        ret_handler_wlog_.ReturnException("");
        break;
      }
      default: {
        OutputLog();
        break;
      }
    }
  }

  void OutputLog() { logger_.OutputLog(); }

  void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone) {
    logger_.Log("-> ");
    logger_.Log(RPCServerStatusToString(code));
    OutputLog();
  }

  template <typename T>
  T* ArenaAlloc(int count) {
    static_assert(std::is_trivial<T>::value && std::is_standard_layout<T>::value,
                  "need to be trival");
    return arena_.template allocate_<T>(count);
  }

  template <typename T>
  bool Read(T* data) {
    static_assert(std::is_trivial<T>::value && std::is_standard_layout<T>::value,
                  "need to be trival");
    return ReadRawBytes(data, sizeof(T));
  }

  template <typename T>
  bool ReadArray(T* data, size_t count) {
    static_assert(std::is_trivial<T>::value && std::is_standard_layout<T>::value,
                  "need to be trival");
    return ReadRawBytes(data, sizeof(T) * count);
  }

  void ReadObject(int* tcode, TVMValue* value) {
    this->ThrowError(RPCServerStatus::kUnknownTypeCode);
  }

 private:
  bool ReadRawBytes(void* data, size_t size) {
    uint8_t* buf = reinterpret_cast<uint8_t*>(data);
    size_t ndone = 0;
    while (ndone < size) {
      ssize_t ret = io_->PosixRead(buf, size - ndone);
      if (ret <= 0) {
        this->ThrowError(RPCServerStatus::kReadError);
        return false;
      }
      ndone += ret;
      buf += ret;
    }
    return true;
  }

  Logger logger_;
  TIOHandler* io_;
  support::GenericArena<PageAllocator> arena_;
  MinRPCReturnsNoOp<TIOHandler> ret_handler_;
  MinRPCReturnsWithLog ret_handler_wlog_;
  MinRPCExecuteNoOp exec_handler_;
  std::unique_ptr<MinRPCExecuteWithLog> exec_handler_ptr_;
  MinRPCServer<TIOHandler> next_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MINRPC_MINRPC_SERVER_LOGGING_H_
