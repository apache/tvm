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

#ifndef TVM_RUNTIME_MINRPC_MINRPC_LOGGER_H_
#define TVM_RUNTIME_MINRPC_MINRPC_LOGGER_H_

#include <tvm/runtime/c_runtime_api.h>

#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "minrpc_interfaces.h"
#include "rpc_reference.h"

namespace tvm {
namespace runtime {

#define PRINT_BYTES false

/*!
 * \brief Generates a user readeable log on the console
 */
class Logger {
 public:
  Logger() {}

  /*!
   * \brief this function logs a string
   *
   * \param s the string to be logged.
   */
  void Log(const char* s) { os_ << s; }
  void Log(std::string s) { os_ << s; }

  /*!
   * \brief this function logs a numerical value
   *
   * \param desc adds any necessary description before the value.
   * \param val is the value to be logged.
   */
  template <typename T>
  void LogValue(const char* desc, T val) {
    os_ << desc << val;
  }

  /*!
   * \brief this function logs the properties of a DLDevice
   *
   * \param desc adds any necessary description before the DLDevice.
   * \param dev is the pointer to the DLDevice to be logged.
   */
  void LogDLDevice(const char* desc, DLDevice* dev) {
    os_ << desc << "(" << dev->device_type << "," << dev->device_id << ")";
  }

  /*!
   * \brief this function logs the properties of a DLDataType
   *
   * \param desc adds any necessary description before the DLDataType.
   * \param data is the pointer to the DLDataType to be logged.
   */
  void LogDLData(const char* desc, DLDataType* data) {
    os_ << desc << "(" << (uint16_t)data->code << "," << (uint16_t)data->bits << "," << data->lanes
        << ")";
  }

  /*!
   * \brief this function logs a handle name.
   *
   * \param name is the name to be logged.
   */
  void LogHandleName(std::string name) {
    if (name.length() > 0) {
      os_ << " <" << name.c_str() << ">";
    }
  }

  /*!
   * \brief this function logs a TVMValue based on its type.
   *
   * \param tcode the type_code of the value stored in TVMValue.
   * \param value is the TVMValue to be logged.
   */
  void LogTVMValue(int tcode, TVMValue value);

  /*!
   * \brief this function output the log to the console.
   */
  void OutputLog();

 private:
  std::stringstream os_;
};

/*!
 * \brief A wrapper for a MinRPCReturns object, that also logs the responses.
 *
 * \param next underlying MinRPCReturns that generates the responses.
 */
class MinRPCReturnsWithLog : public MinRPCReturnInterface {
 public:
  /*!
   * \brief Constructor.
   * \param io The IO handler.
   */
  MinRPCReturnsWithLog(MinRPCReturnInterface* next, Logger* logger)
      : next_(next), logger_(logger) {}

  ~MinRPCReturnsWithLog() {}

  void ReturnVoid();

  void ReturnHandle(void* handle);

  void ReturnException(const char* msg);

  void ReturnPackedSeq(const TVMValue* arg_values, const int* type_codes, int num_args);

  void ReturnCopyFromRemote(uint8_t* data_ptr, uint64_t num_bytes);

  void ReturnLastTVMError();

  void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone);

  /*!
   * \brief this function logs a list of TVMValues, and registers handle_name when needed.
   *
   * \param values is the list of TVMValues.
   * \param tcodes is the list type_code of the TVMValues.
   * \param num_args is the number of items in the list.
   */
  void ProcessValues(const TVMValue* values, const int* tcodes, int num_args);

  /*!
   * \brief this function is called when a new command is executed.
   * It clears the handle_name_ and records the command code.
   *
   * \param code the RPC command code.
   */
  void ResetHandleName(RPCCode code);

  /*!
   * \brief appends name to the handle_name_.
   *
   * \param name handle name.
   */
  void UpdateHandleName(const char* name);

  /*!
   * \brief get the stored handle description.
   *
   * \param handle the handle to get the description for.
   */
  void GetHandleName(void* handle);

  /*!
   * \brief remove the handle description from handle_descriptions_.
   *
   * \param handle the handle to remove the description for.
   */
  void ReleaseHandleName(void* handle);

 private:
  /*!
   * \brief add the handle description to handle_descriptions_.
   *
   * \param handle the handle to add the description for.
   */
  void RegisterHandleName(void* handle);

  MinRPCReturnInterface* next_;
  std::string handle_name_;
  std::unordered_map<void*, std::string> handle_descriptions_;
  RPCCode code_;
  Logger* logger_;
};

/*!
 * \brief A wrapper for a MinRPCExecute object, that also logs the responses.
 *
 * \param next: underlying MinRPCExecute that processes the packets.
 */
class MinRPCExecuteWithLog : public MinRPCExecInterface {
 public:
  MinRPCExecuteWithLog(MinRPCExecInterface* next, Logger* logger) : next_(next), logger_(logger) {
    ret_handler_ = reinterpret_cast<MinRPCReturnsWithLog*>(next_->GetReturnInterface());
  }

  ~MinRPCExecuteWithLog() {}

  void InitServer(int num_args);

  void NormalCallFunc(uint64_t call_handle, TVMValue* values, int* tcodes, int num_args);

  void CopyFromRemote(DLTensor* arr, uint64_t num_bytes, uint8_t* temp_data);

  int CopyToRemote(DLTensor* arr, uint64_t _num_bytes, uint8_t* _data_ptr);

  void SysCallFunc(RPCCode code, TVMValue* values, int* tcodes, int num_args);

  void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone);

  MinRPCReturnInterface* GetReturnInterface() { return next_->GetReturnInterface(); }

 private:
  /*!
   * \brief this function logs a list of TVMValues, and updates handle_name when needed.
   *
   * \param values is the list of TVMValues.
   * \param tcodes is the list type_code of the TVMValues.
   * \param num_args is the number of items in the list.
   */
  void ProcessValues(TVMValue* values, int* tcodes, int num_args);

  /*!
   * \brief this function is called when a new command is executed.
   *
   * \param code the RPC command code.
   */
  void SetRPCCode(RPCCode code);

  MinRPCExecInterface* next_;
  MinRPCReturnsWithLog* ret_handler_;
  Logger* logger_;
};

/*!
 * \brief A No-operation MinRPCReturns used within the MinRPCSniffer
 *
 * \tparam TIOHandler* IO provider to provide io handling.
 */
template <typename TIOHandler>
class MinRPCReturnsNoOp : public MinRPCReturnInterface {
 public:
  /*!
   * \brief Constructor.
   * \param io The IO handler.
   */
  explicit MinRPCReturnsNoOp(TIOHandler* io) : io_(io) {}
  ~MinRPCReturnsNoOp() {}
  void ReturnVoid() {}
  void ReturnHandle(void* handle) {}
  void ReturnException(const char* msg) {}
  void ReturnPackedSeq(const TVMValue* arg_values, const int* type_codes, int num_args) {}
  void ReturnCopyFromRemote(uint8_t* data_ptr, uint64_t num_bytes) {}
  void ReturnLastTVMError() {}
  void ThrowError(RPCServerStatus code, RPCCode info) {}

 private:
  TIOHandler* io_;
};

/*!
 * \brief A No-operation MinRPCExecute used within the MinRPCSniffer
 *
 * \tparam ReturnInterface* ReturnInterface pointer to generate and send the responses.

 */
class MinRPCExecuteNoOp : public MinRPCExecInterface {
 public:
  explicit MinRPCExecuteNoOp(MinRPCReturnInterface* ret_handler) : ret_handler_(ret_handler) {}
  ~MinRPCExecuteNoOp() {}
  void InitServer(int _num_args) {}
  void NormalCallFunc(uint64_t call_handle, TVMValue* values, int* tcodes, int num_args) {}
  void CopyFromRemote(DLTensor* arr, uint64_t num_bytes, uint8_t* temp_data) {}
  int CopyToRemote(DLTensor* arr, uint64_t num_bytes, uint8_t* data_ptr) { return 1; }
  void SysCallFunc(RPCCode code, TVMValue* values, int* tcodes, int num_args) {}
  void ThrowError(RPCServerStatus code, RPCCode info) {}
  MinRPCReturnInterface* GetReturnInterface() { return ret_handler_; }

 private:
  MinRPCReturnInterface* ret_handler_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MINRPC_MINRPC_LOGGER_H_"
