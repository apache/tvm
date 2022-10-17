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

#include "minrpc_logger.h"

#include <string.h>
#include <time.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "minrpc_interfaces.h"
#include "rpc_reference.h"

namespace tvm {
namespace runtime {

void Logger::LogTVMValue(int tcode, TVMValue value) {
  switch (tcode) {
    case kDLInt: {
      LogValue<int64_t>("(int64)", value.v_int64);
      break;
    }
    case kDLUInt: {
      LogValue<uint64_t>("(uint64)", value.v_int64);
      break;
    }
    case kDLFloat: {
      LogValue<float>("(float)", value.v_float64);
      break;
    }
    case kTVMDataType: {
      LogDLData("DLDataType(code,bits,lane)", &value.v_type);
      break;
    }
    case kDLDevice: {
      LogDLDevice("DLDevice(type,id)", &value.v_device);
      break;
    }
    case kTVMPackedFuncHandle: {
      LogValue<void*>("(PackedFuncHandle)", value.v_handle);
      break;
    }
    case kTVMModuleHandle: {
      LogValue<void*>("(ModuleHandle)", value.v_handle);
      break;
    }
    case kTVMOpaqueHandle: {
      LogValue<void*>("(OpaqueHandle)", value.v_handle);
      break;
    }
    case kTVMDLTensorHandle: {
      LogValue<void*>("(TensorHandle)", value.v_handle);
      break;
    }
    case kTVMNDArrayHandle: {
      LogValue<void*>("kTVMNDArrayHandle", value.v_handle);
      break;
    }
    case kTVMNullptr: {
      Log("Nullptr");
      break;
    }
    case kTVMStr: {
      Log("\"");
      Log(value.v_str);
      Log("\"");
      break;
    }
    case kTVMBytes: {
      TVMByteArray* bytes = static_cast<TVMByteArray*>(value.v_handle);
      int len = bytes->size;
      LogValue<int64_t>("(Bytes) [size]: ", len);
      if (PRINT_BYTES) {
        Log(", [Values]:");
        Log(" { ");
        if (len > 0) {
          LogValue<uint64_t>("", (uint8_t)bytes->data[0]);
        }
        for (int j = 1; j < len; j++) LogValue<uint64_t>(" - ", (uint8_t)bytes->data[j]);
        Log(" } ");
      }
      break;
    }
    default: {
      Log("ERROR-kUnknownTypeCode)");
      break;
    }
  }
  Log("; ");
}

void Logger::OutputLog() {
  LOG(INFO) << os_.str();
  os_.str(std::string());
}

void MinRPCReturnsWithLog::ReturnVoid() {
  next_->ReturnVoid();
  logger_->Log("-> ReturnVoid");
  logger_->OutputLog();
}

void MinRPCReturnsWithLog::ReturnHandle(void* handle) {
  next_->ReturnHandle(handle);
  if (code_ == RPCCode::kGetGlobalFunc) {
    RegisterHandleName(handle);
  }
  logger_->LogValue<void*>("-> ReturnHandle: ", handle);
  logger_->OutputLog();
}

void MinRPCReturnsWithLog::ReturnException(const char* msg) {
  next_->ReturnException(msg);
  logger_->Log("-> Exception: ");
  logger_->Log(msg);
  logger_->OutputLog();
}

void MinRPCReturnsWithLog::ReturnPackedSeq(const TVMValue* arg_values, const int* type_codes,
                                           int num_args) {
  next_->ReturnPackedSeq(arg_values, type_codes, num_args);
  ProcessValues(arg_values, type_codes, num_args);
  logger_->OutputLog();
}

void MinRPCReturnsWithLog::ReturnCopyFromRemote(uint8_t* data_ptr, uint64_t num_bytes) {
  next_->ReturnCopyFromRemote(data_ptr, num_bytes);
  logger_->LogValue<uint64_t>("-> CopyFromRemote: ", num_bytes);
  logger_->LogValue<void*>(", ", static_cast<void*>(data_ptr));
  logger_->OutputLog();
}

void MinRPCReturnsWithLog::ReturnLastTVMError() {
  const char* err = TVMGetLastError();
  ReturnException(err);
}

void MinRPCReturnsWithLog::ThrowError(RPCServerStatus code, RPCCode info) {
  next_->ThrowError(code, info);
  logger_->Log("-> ERROR: ");
  logger_->Log(RPCServerStatusToString(code));
  logger_->OutputLog();
}

void MinRPCReturnsWithLog::ProcessValues(const TVMValue* values, const int* tcodes, int num_args) {
  if (tcodes != nullptr) {
    logger_->Log("-> [");
    for (int i = 0; i < num_args; ++i) {
      logger_->LogTVMValue(tcodes[i], values[i]);

      if (tcodes[i] == kTVMOpaqueHandle) {
        RegisterHandleName(values[i].v_handle);
      }
    }
    logger_->Log("]");
  }
}

void MinRPCReturnsWithLog::ResetHandleName(RPCCode code) {
  code_ = code;
  handle_name_.clear();
}

void MinRPCReturnsWithLog::UpdateHandleName(const char* name) {
  if (handle_name_.length() != 0) {
    handle_name_.append("::");
  }
  handle_name_.append(name);
}

void MinRPCReturnsWithLog::GetHandleName(void* handle) {
  if (handle_descriptions_.find(handle) != handle_descriptions_.end()) {
    handle_name_.append(handle_descriptions_[handle]);
    logger_->LogHandleName(handle_name_);
  }
}

void MinRPCReturnsWithLog::ReleaseHandleName(void* handle) {
  if (handle_descriptions_.find(handle) != handle_descriptions_.end()) {
    logger_->LogHandleName(handle_descriptions_[handle]);
    handle_descriptions_.erase(handle);
  }
}

void MinRPCReturnsWithLog::RegisterHandleName(void* handle) {
  handle_descriptions_[handle] = handle_name_;
}

void MinRPCExecuteWithLog::InitServer(int num_args) {
  SetRPCCode(RPCCode::kInitServer);
  logger_->Log("Init Server");
  next_->InitServer(num_args);
}

void MinRPCExecuteWithLog::NormalCallFunc(uint64_t call_handle, TVMValue* values, int* tcodes,
                                          int num_args) {
  SetRPCCode(RPCCode::kCallFunc);
  logger_->LogValue<void*>("call_handle: ", reinterpret_cast<void*>(call_handle));
  ret_handler_->GetHandleName(reinterpret_cast<void*>(call_handle));
  if (num_args > 0) {
    logger_->Log(", ");
  }
  ProcessValues(values, tcodes, num_args);
  next_->NormalCallFunc(call_handle, values, tcodes, num_args);
}

void MinRPCExecuteWithLog::CopyFromRemote(DLTensor* arr, uint64_t num_bytes, uint8_t* temp_data) {
  SetRPCCode(RPCCode::kCopyFromRemote);
  logger_->LogValue<void*>("data_handle: ", static_cast<void*>(arr->data));
  logger_->LogDLDevice(", DLDevice(type,id):", &(arr->device));
  logger_->LogValue<int64_t>(", ndim: ", arr->ndim);
  logger_->LogDLData(", DLDataType(code,bits,lane): ", &(arr->dtype));
  logger_->LogValue<uint64_t>(", num_bytes:", num_bytes);
  next_->CopyFromRemote(arr, num_bytes, temp_data);
}

int MinRPCExecuteWithLog::CopyToRemote(DLTensor* arr, uint64_t num_bytes, uint8_t* data_ptr) {
  SetRPCCode(RPCCode::kCopyToRemote);
  logger_->LogValue<void*>("data_handle: ", static_cast<void*>(arr->data));
  logger_->LogDLDevice(", DLDevice(type,id):", &(arr->device));
  logger_->LogValue<int64_t>(", ndim: ", arr->ndim);
  logger_->LogDLData(", DLDataType(code,bits,lane): ", &(arr->dtype));
  logger_->LogValue<uint64_t>(", byte_offset: ", arr->byte_offset);
  return next_->CopyToRemote(arr, num_bytes, data_ptr);
}

void MinRPCExecuteWithLog::SysCallFunc(RPCCode code, TVMValue* values, int* tcodes, int num_args) {
  SetRPCCode(code);
  if ((code) == RPCCode::kFreeHandle) {
    if ((num_args == 2) && (tcodes[0] == kTVMOpaqueHandle) && (tcodes[1] == kDLInt)) {
      logger_->LogValue<void*>("handle: ", static_cast<void*>(values[0].v_handle));
      if (values[1].v_int64 == kTVMModuleHandle || values[1].v_int64 == kTVMPackedFuncHandle) {
        ret_handler_->ReleaseHandleName(static_cast<void*>(values[0].v_handle));
      }
    }
  } else {
    ProcessValues(values, tcodes, num_args);
  }
  next_->SysCallFunc(code, values, tcodes, num_args);
}

void MinRPCExecuteWithLog::ThrowError(RPCServerStatus code, RPCCode info) {
  logger_->Log("-> Error\n");
  next_->ThrowError(code, info);
}

void MinRPCExecuteWithLog::ProcessValues(TVMValue* values, int* tcodes, int num_args) {
  if (tcodes != nullptr) {
    logger_->Log("[");
    for (int i = 0; i < num_args; ++i) {
      logger_->LogTVMValue(tcodes[i], values[i]);

      if (tcodes[i] == kTVMStr) {
        if (strlen(values[i].v_str) > 0) {
          ret_handler_->UpdateHandleName(values[i].v_str);
        }
      }
    }
    logger_->Log("]");
  }
}

void MinRPCExecuteWithLog::SetRPCCode(RPCCode code) {
  logger_->Log(RPCCodeToString(code));
  logger_->Log(", ");
  ret_handler_->ResetHandleName(code);
}

}  // namespace runtime
}  // namespace tvm
