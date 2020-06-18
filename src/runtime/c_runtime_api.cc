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
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <sstream>
#include <array>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <cctype>
#include "runtime_base.h"
#include "object_internal.h"

namespace tvm {
namespace runtime {

std::string GetCustomTypeName(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("runtime._datatype_get_type_name");
  CHECK(f) << "Function runtime._datatype_get_type_name not found";
  return (*f)(type_code).operator std::string();
}

uint8_t GetCustomTypeCode(const std::string& type_name) {
  auto f = tvm::runtime::Registry::Get("runtime._datatype_get_type_code");
  CHECK(f) << "Function runtime._datatype_get_type_code not found";
  return (*f)(type_name).operator int();
}

bool GetCustomTypeRegistered(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("runtime._datatype_get_type_registered");
  CHECK(f) << "Function runtime._datatype_get_type_registered not found";
  return (*f)(type_code).operator bool();
}

uint8_t ParseCustomDatatype(const std::string& s, const char** scan) {
  CHECK(s.substr(0, 6) == "custom") << "Not a valid custom datatype string";

  auto tmp = s.c_str();

  CHECK(s.c_str() == tmp);
  *scan = s.c_str() + 6;
  CHECK(s.c_str() == tmp);
  if (**scan != '[') LOG(FATAL) << "expected opening brace after 'custom' type in" << s;
  CHECK(s.c_str() == tmp);
  *scan += 1;
  CHECK(s.c_str() == tmp);
  size_t custom_name_len = 0;
  CHECK(s.c_str() == tmp);
  while (*scan + custom_name_len <= s.c_str() + s.length() && *(*scan + custom_name_len) != ']')
    ++custom_name_len;
  CHECK(s.c_str() == tmp);
  if (*(*scan + custom_name_len) != ']')
    LOG(FATAL) << "expected closing brace after 'custom' type in" << s;
  CHECK(s.c_str() == tmp);
  *scan += custom_name_len + 1;
  CHECK(s.c_str() == tmp);

  auto type_name = s.substr(7, custom_name_len);
  CHECK(s.c_str() == tmp);
  return GetCustomTypeCode(type_name);
}

class DeviceAPIManager {
 public:
  static const int kMaxDeviceAPI = 32;
  // Get API
  static DeviceAPI* Get(const TVMContext& ctx) {
    return Get(ctx.device_type);
  }
  static DeviceAPI* Get(int dev_type, bool allow_missing = false) {
    return Global()->GetAPI(dev_type, allow_missing);
  }

 private:
  std::array<DeviceAPI*, kMaxDeviceAPI> api_;
  DeviceAPI* rpc_api_{nullptr};
  std::mutex mutex_;
  // constructor
  DeviceAPIManager() {
    std::fill(api_.begin(), api_.end(), nullptr);
  }
  // Global static variable.
  static DeviceAPIManager* Global() {
    static DeviceAPIManager inst;
    return &inst;
  }
  // Get or initialize API.
  DeviceAPI* GetAPI(int type, bool allow_missing) {
    if (type < kRPCSessMask) {
      if (api_[type] != nullptr) return api_[type];
      std::lock_guard<std::mutex> lock(mutex_);
      if (api_[type] != nullptr) return api_[type];
      api_[type] = GetAPI(DeviceName(type), allow_missing);
      return api_[type];
    } else {
      if (rpc_api_ != nullptr) return rpc_api_;
      std::lock_guard<std::mutex> lock(mutex_);
      if (rpc_api_ != nullptr) return rpc_api_;
      rpc_api_ = GetAPI("rpc", allow_missing);
      return rpc_api_;
    }
  }
  DeviceAPI* GetAPI(const std::string name, bool allow_missing) {
    std::string factory = "device_api." + name;
    auto* f = Registry::Get(factory);
    if (f == nullptr) {
      CHECK(allow_missing)
          << "Device API " << name << " is not enabled.";
      return nullptr;
    }
    void* ptr = (*f)();
    return static_cast<DeviceAPI*>(ptr);
  }
};

DeviceAPI* DeviceAPI::Get(TVMContext ctx, bool allow_missing) {
  return DeviceAPIManager::Get(
      static_cast<int>(ctx.device_type), allow_missing);
}

void* DeviceAPI::AllocWorkspace(TVMContext ctx,
                                size_t size,
                                DLDataType type_hint) {
  return AllocDataSpace(ctx, size, kTempAllocaAlignment, type_hint);
}

void DeviceAPI::FreeWorkspace(TVMContext ctx, void* ptr) {
  FreeDataSpace(ctx, ptr);
}

TVMStreamHandle DeviceAPI::CreateStream(TVMContext ctx) {
  LOG(FATAL) << "Device does not support stream api.";
  return 0;
}

void DeviceAPI::FreeStream(TVMContext ctx, TVMStreamHandle stream) {
  LOG(FATAL) << "Device does not support stream api.";
}

void DeviceAPI::SyncStreamFromTo(TVMContext ctx,
                                 TVMStreamHandle event_src,
                                 TVMStreamHandle event_dst) {
  LOG(FATAL) << "Device does not support stream api.";
}

//--------------------------------------------------------
// Error handling mechanism
// -------------------------------------------------------
// Standard error message format, {} means optional
//--------------------------------------------------------
// {error_type:} {message0}
// {message1}
// {message2}
// {Stack trace:}    // stack traces follow by this line
//   {trace 0}       // two spaces in the begining.
//   {trace 1}
//   {trace 2}
//--------------------------------------------------------
/*!
 * \brief Normalize error message
 *
 *  Parse them header generated by by LOG(FATAL) and CHECK
 *  and reformat the message into the standard format.
 *
 *  This function will also merge all the stack traces into
 *  one trace and trim them.
 *
 * \param err_msg The error message.
 * \return normalized message.
 */
std::string NormalizeError(std::string err_msg) {
  // ------------------------------------------------------------------------
  // log with header, {} indicates optional
  //-------------------------------------------------------------------------
  // [timestamp] file_name:line_number: {check_msg:} {error_type:} {message0}
  // {message1}
  // Stack trace:
  //   {stack trace 0}
  //   {stack trace 1}
  //-------------------------------------------------------------------------
  // Normalzied version
  //-------------------------------------------------------------------------
  // error_type: check_msg message0
  // {message1}
  // Stack trace:
  //   File file_name, line lineno
  //   {stack trace 0}
  //   {stack trace 1}
  //-------------------------------------------------------------------------
  int line_number = 0;
  std::istringstream is(err_msg);
  std::string line, file_name, error_type, check_msg;

  // Parse log header and set the fields,
  // Return true if it the log is in correct format,
  // return false if something is wrong.
  auto parse_log_header = [&]() {
    // skip timestamp
    if (is.peek() != '[') {
      getline(is, line);
      return true;
    }
    if (!(is >> line)) return false;
    // get filename
    while (is.peek() == ' ') is.get();
#ifdef _MSC_VER  // handle volume separator ":" in Windows path
    std::string drive;
    if (!getline(is, drive, ':')) return false;
    if (!getline(is, file_name, ':')) return false;
    file_name = drive + ":" + file_name;
#else
    if (!getline(is, file_name, ':')) return false;
#endif
    // get line number
    if (!(is >> line_number)) return false;
    // get rest of the message.
    while (is.peek() == ' ' || is.peek() == ':') is.get();
    if (!getline(is, line)) return false;
    // detect check message, rewrite to remote extra :
    if (line.compare(0, 13, "Check failed:") == 0) {
      size_t end_pos = line.find(':', 13);
      if (end_pos == std::string::npos) return false;
      check_msg = line.substr(0, end_pos + 1) + ' ';
      line = line.substr(end_pos + 1);
    }
    return true;
  };
  // if not in correct format, do not do any rewrite.
  if (!parse_log_header()) return err_msg;
  // Parse error type.
  {
    size_t start_pos = 0, end_pos;
    for (; start_pos < line.length() && line[start_pos] == ' '; ++start_pos) {}
    for (end_pos = start_pos; end_pos < line.length(); ++end_pos) {
      char ch = line[end_pos];
      if (ch == ':') {
        error_type = line.substr(start_pos, end_pos - start_pos);
        break;
      }
      // [A-Z0-9a-z_.]
      if (!std::isalpha(ch) && !std::isdigit(ch) && ch != '_' && ch != '.') break;
    }
    if (error_type.length() != 0) {
      // if we successfully detected error_type: trim the following space.
      for (start_pos = end_pos + 1;
           start_pos < line.length() && line[start_pos] == ' '; ++start_pos) {}
      line = line.substr(start_pos);
    } else {
      // did not detect error_type, use default value.
      line = line.substr(start_pos);
      error_type = "TVMError";
    }
  }
  // Seperate out stack trace.
  std::ostringstream os;
  os << error_type << ": " << check_msg << line << '\n';

  bool trace_mode = true;
  std::vector<std::string> stack_trace;
  while (getline(is, line)) {
    if (trace_mode) {
      if (line.compare(0, 2, "  ") == 0) {
        stack_trace.push_back(line);
      } else {
        trace_mode = false;
        // remove EOL trailing stacktrace.
        if (line.length() == 0) continue;
      }
    }
    if (!trace_mode) {
      if (line.compare(0, 11, "Stack trace") == 0) {
        trace_mode = true;
      } else {
        os << line << '\n';
      }
    }
  }
  if (stack_trace.size() != 0 || file_name.length() != 0) {
    os << "Stack trace:\n";
    if (file_name.length() != 0) {
      os << "  File \"" << file_name << "\", line " << line_number << "\n";
    }
    // Print out stack traces, optionally trim the c++ traces
    // about the frontends (as they will be provided by the frontends).
    bool ffi_boundary = false;
    for (const auto& line : stack_trace) {
      // Heuristic to detect python ffi.
      if (line.find("libffi.so") != std::string::npos ||
          line.find("core.cpython") != std::string::npos) {
        ffi_boundary = true;
      }
      // If the backtrace is not c++ backtrace with the prefix "  [bt]",
      // then we can stop trimming.
      if (ffi_boundary && line.compare(0, 6, "  [bt]") != 0) {
        ffi_boundary = false;
      }
      if (!ffi_boundary) {
        os << line << '\n';
      }
      // The line after TVMFuncCall cound be in FFI.
      if (line.find("(TVMFuncCall") != std::string::npos) {
        ffi_boundary = true;
      }
    }
  }
  return os.str();
}

}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

struct TVMRuntimeEntry {
  std::string ret_str;
  std::string last_error;
  TVMByteArray ret_bytes;
};

typedef dmlc::ThreadLocalStore<TVMRuntimeEntry> TVMAPIRuntimeStore;

const char *TVMGetLastError() {
  return TVMAPIRuntimeStore::Get()->last_error.c_str();
}

int TVMAPIHandleException(const std::runtime_error &e) {
  TVMAPISetLastError(NormalizeError(e.what()).c_str());
  return -1;
}

void TVMAPISetLastError(const char* msg) {
  TVMAPIRuntimeStore::Get()->last_error = msg;
}

int TVMModLoadFromFile(const char* file_name,
                       const char* format,
                       TVMModuleHandle* out) {
  API_BEGIN();
  TVMRetValue ret;
  ret = Module::LoadFromFile(file_name, format);
  TVMValue val;
  int type_code;
  ret.MoveToCHost(&val, &type_code);
  *out = val.v_handle;
  API_END();
}

int TVMModImport(TVMModuleHandle mod,
                 TVMModuleHandle dep) {
  API_BEGIN();
  ObjectInternal::GetModuleNode(mod)->Import(
      GetRef<Module>(ObjectInternal::GetModuleNode(dep)));
  API_END();
}

int TVMModGetFunction(TVMModuleHandle mod,
                      const char* func_name,
                      int query_imports,
                      TVMFunctionHandle *func) {
  API_BEGIN();
  PackedFunc pf = ObjectInternal::GetModuleNode(mod)->GetFunction(
      func_name, query_imports != 0);
  if (pf != nullptr) {
    *func = new PackedFunc(pf);
  } else {
    *func = nullptr;
  }
  API_END();
}

int TVMModFree(TVMModuleHandle mod) {
  return TVMObjectFree(mod);
}

int TVMBackendGetFuncFromEnv(void* mod_node,
                             const char* func_name,
                             TVMFunctionHandle *func) {
  API_BEGIN();
  *func = (TVMFunctionHandle)(
      static_cast<ModuleNode*>(mod_node)->GetFuncFromEnv(func_name));
  API_END();
}

void* TVMBackendAllocWorkspace(int device_type,
                               int device_id,
                               uint64_t size,
                               int dtype_code_hint,
                               int dtype_bits_hint) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;

  DLDataType type_hint;
  type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  type_hint.lanes = 1;

  return DeviceAPIManager::Get(ctx)->AllocWorkspace(ctx,
                                                    static_cast<size_t>(size),
                                                    type_hint);
}

int TVMBackendFreeWorkspace(int device_type,
                            int device_id,
                            void* ptr) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->FreeWorkspace(ctx, ptr);
  return 0;
}

int TVMBackendRunOnce(void** handle,
                      int (*f)(void*),
                      void* cdata,
                      int nbytes) {
  if (*handle == nullptr) {
    *handle = reinterpret_cast<void*>(1);
    return (*f)(cdata);
  }
  return 0;
}

int TVMFuncFree(TVMFunctionHandle func) {
  API_BEGIN();
  delete static_cast<PackedFunc*>(func);
  API_END();
}

int TVMFuncCall(TVMFunctionHandle func,
                TVMValue* args,
                int* arg_type_codes,
                int num_args,
                TVMValue* ret_val,
                int* ret_type_code) {
  API_BEGIN();
  TVMRetValue rv;
  (*static_cast<const PackedFunc*>(func)).CallPacked(
      TVMArgs(args, arg_type_codes, num_args), &rv);
  // handle return string.
  if (rv.type_code() == kTVMStr ||
      rv.type_code() == kTVMDataType ||
      rv.type_code() == kTVMBytes) {
    TVMRuntimeEntry* e = TVMAPIRuntimeStore::Get();
    if (rv.type_code() != kTVMDataType) {
      e->ret_str = *rv.ptr<std::string>();
    } else {
      e->ret_str = rv.operator std::string();
    }
    if (rv.type_code() == kTVMBytes) {
      e->ret_bytes.data = e->ret_str.c_str();
      e->ret_bytes.size = e->ret_str.length();
      *ret_type_code = kTVMBytes;
      ret_val->v_handle = &(e->ret_bytes);
    } else {
      *ret_type_code = kTVMStr;
      ret_val->v_str = e->ret_str.c_str();
    }
  } else {
    rv.MoveToCHost(ret_val, ret_type_code);
  }
  API_END();
}

int TVMCFuncSetReturn(TVMRetValueHandle ret,
                      TVMValue* value,
                      int* type_code,
                      int num_ret) {
  API_BEGIN();
  CHECK_EQ(num_ret, 1);
  TVMRetValue* rv = static_cast<TVMRetValue*>(ret);
  *rv = TVMArgValue(value[0], type_code[0]);
  API_END();
}

int TVMFuncCreateFromCFunc(TVMPackedCFunc func,
                           void* resource_handle,
                           TVMPackedCFuncFinalizer fin,
                           TVMFunctionHandle *out) {
  API_BEGIN();
  if (fin == nullptr) {
    *out = new PackedFunc(
        [func, resource_handle](TVMArgs args, TVMRetValue* rv) {
          int ret = func((TVMValue*)args.values, (int*)args.type_codes, // NOLINT(*)
                         args.num_args, rv, resource_handle);
          if (ret != 0) {
            throw dmlc::Error(TVMGetLastError() + ::dmlc::StackTrace());
          }
        });
  } else {
    // wrap it in a shared_ptr, with fin as deleter.
    // so fin will be called when the lambda went out of scope.
    std::shared_ptr<void> rpack(resource_handle, fin);
    *out = new PackedFunc(
        [func, rpack](TVMArgs args, TVMRetValue* rv) {
          int ret = func((TVMValue*)args.values, (int*)args.type_codes, // NOLINT(*)
                         args.num_args, rv, rpack.get());
          if (ret != 0) {
            throw dmlc::Error(TVMGetLastError() + ::dmlc::StackTrace());
          }
      });
  }
  API_END();
}

int TVMStreamCreate(int device_type, int device_id, TVMStreamHandle* out) {
  API_BEGIN();
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  *out = DeviceAPIManager::Get(ctx)->CreateStream(ctx);
  API_END();
}

int TVMStreamFree(int device_type, int device_id, TVMStreamHandle stream) {
  API_BEGIN();
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->FreeStream(ctx, stream);
  API_END();
}

int TVMSetStream(int device_type, int device_id, TVMStreamHandle stream) {
  API_BEGIN();
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->SetStream(ctx, stream);
  API_END();
}

int TVMSynchronize(int device_type, int device_id, TVMStreamHandle stream) {
  API_BEGIN();
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->StreamSync(ctx, stream);
  API_END();
}

int TVMStreamStreamSynchronize(int device_type,
                               int device_id,
                               TVMStreamHandle src,
                               TVMStreamHandle dst) {
  API_BEGIN();
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->SyncStreamFromTo(ctx, src, dst);
  API_END();
}

int TVMCbArgToReturn(TVMValue* value, int* code) {
  API_BEGIN();
  tvm::runtime::TVMRetValue rv;
  rv = tvm::runtime::TVMMovableArgValue_(*value, *code);
  rv.MoveToCHost(value, code);
  API_END();
}

// set device api
TVM_REGISTER_GLOBAL(tvm::runtime::symbol::tvm_set_device)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(args[0].operator int());
    ctx.device_id = args[1];
    DeviceAPIManager::Get(ctx)->SetDevice(ctx);
  });

// set device api
TVM_REGISTER_GLOBAL("runtime.GetDeviceAttr")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    TVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(args[0].operator int());
    ctx.device_id = args[1];

    DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[2].operator int());
    if (kind == kExist) {
      DeviceAPI* api = DeviceAPIManager::Get(ctx.device_type, true);
      if (api != nullptr) {
        api->GetAttr(ctx, kind, ret);
      } else {
        *ret = 0;
      }
    } else {
      DeviceAPIManager::Get(ctx)->GetAttr(ctx, kind, ret);
    }
  });


TVM_REGISTER_GLOBAL("runtime.TVMSetStream")
.set_body_typed(TVMSetStream);
