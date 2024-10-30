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
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>

#include "object_internal.h"
#include "runtime_base.h"

namespace tvm {
namespace runtime {

std::string GetCustomTypeName(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("runtime._datatype_get_type_name");
  ICHECK(f) << "Function runtime._datatype_get_type_name not found";
  return (*f)(type_code).operator std::string();
}

uint8_t GetCustomTypeCode(const std::string& type_name) {
  auto f = tvm::runtime::Registry::Get("runtime._datatype_get_type_code");
  ICHECK(f) << "Function runtime._datatype_get_type_code not found";
  return (*f)(type_name).operator int();
}

bool GetCustomTypeRegistered(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("runtime._datatype_get_type_registered");
  ICHECK(f) << "Function runtime._datatype_get_type_registered not found";
  return (*f)(type_code).operator bool();
}

uint8_t ParseCustomDatatype(const std::string& s, const char** scan) {
  ICHECK(s.substr(0, 6) == "custom") << "Not a valid custom datatype string";

  auto tmp = s.c_str();

  ICHECK(s.c_str() == tmp);
  *scan = s.c_str() + 6;
  ICHECK(s.c_str() == tmp);
  if (**scan != '[') LOG(FATAL) << "expected opening brace after 'custom' type in" << s;
  ICHECK(s.c_str() == tmp);
  *scan += 1;
  ICHECK(s.c_str() == tmp);
  size_t custom_name_len = 0;
  ICHECK(s.c_str() == tmp);
  while (*scan + custom_name_len <= s.c_str() + s.length() && *(*scan + custom_name_len) != ']')
    ++custom_name_len;
  ICHECK(s.c_str() == tmp);
  if (*(*scan + custom_name_len) != ']')
    LOG(FATAL) << "expected closing brace after 'custom' type in" << s;
  ICHECK(s.c_str() == tmp);
  *scan += custom_name_len + 1;
  ICHECK(s.c_str() == tmp);

  auto type_name = s.substr(7, custom_name_len);
  ICHECK(s.c_str() == tmp);
  return GetCustomTypeCode(type_name);
}

class DeviceAPIManager {
 public:
  static const int kMaxDeviceAPI = TVMDeviceExtType_End;
  // Get API
  static DeviceAPI* Get(const Device& dev) { return Get(dev.device_type); }
  static DeviceAPI* Get(int dev_type, bool allow_missing = false) {
    return Global()->GetAPI(dev_type, allow_missing);
  }

 private:
  std::array<DeviceAPI*, kMaxDeviceAPI> api_;
  DeviceAPI* rpc_api_{nullptr};
  std::mutex mutex_;
  // constructor
  DeviceAPIManager() { std::fill(api_.begin(), api_.end(), nullptr); }
  // Global static variable.
  static DeviceAPIManager* Global() {
    static DeviceAPIManager* inst = new DeviceAPIManager();
    return inst;
  }
  // Get or initialize API.
  DeviceAPI* GetAPI(int type, bool allow_missing) {
    if (type < kRPCSessMask) {
      if (api_[type] != nullptr) return api_[type];
      std::lock_guard<std::mutex> lock(mutex_);
      if (api_[type] != nullptr) return api_[type];
      api_[type] = GetAPI(DLDeviceType2Str(type), allow_missing);
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
      ICHECK(allow_missing) << "Device API " << name << " is not enabled.";
      return nullptr;
    }
    void* ptr = (*f)();
    return static_cast<DeviceAPI*>(ptr);
  }
};

DeviceAPI* DeviceAPI::Get(Device dev, bool allow_missing) {
  return DeviceAPIManager::Get(static_cast<int>(dev.device_type), allow_missing);
}

void* DeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return AllocDataSpace(dev, size, kTempAllocaAlignment, type_hint);
}

static size_t GetDataAlignment(const DLDataType dtype) {
  size_t align = (dtype.bits / 8) * dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

size_t DeviceAPI::GetDataSize(const DLTensor& arr, Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value().empty() || mem_scope.value() == "global") {
    size_t size = 1;
    for (tvm_index_t i = 0; i < arr.ndim; ++i) {
      size *= static_cast<size_t>(arr.shape[i]);
    }
    size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
    return size;
  }
  LOG(FATAL) << "Device does not support physical mem computation with "
             << "specified memory scope: " << mem_scope.value();
  return 0;
}

void* DeviceAPI::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value() == "" || mem_scope.value() == "global") {
    // by default, we can always redirect to the flat memory allocations
    DLTensor temp;
    temp.data = nullptr;
    temp.device = dev;
    temp.ndim = ndim;
    temp.dtype = dtype;
    temp.shape = const_cast<int64_t*>(shape);
    temp.strides = nullptr;
    temp.byte_offset = 0;
    size_t size = GetDataSize(temp);
    size_t alignment = GetDataAlignment(temp.dtype);
    return AllocDataSpace(dev, size, alignment, dtype);
  }
  LOG(FATAL) << "Device does not support allocate data space with "
             << "specified memory scope: " << mem_scope.value();
  return nullptr;
}

void DeviceAPI::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  // by default, we can always redirect to the flat memory copy operation.
  size_t nbytes = GetDataSize(*from);
  ICHECK_EQ(nbytes, GetDataSize(*to));

  ICHECK(IsContiguous(*from) && IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";
  CopyDataFromTo(from->data, from->byte_offset, to->data, to->byte_offset, nbytes, from->device,
                 to->device, from->dtype, stream);
}

void DeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                               size_t num_bytes, Device dev_from, Device dev_to,
                               DLDataType type_hint, TVMStreamHandle stream) {
  LOG(FATAL) << "Device does not support CopyDataFromTo.";
}

void DeviceAPI::FreeWorkspace(Device dev, void* ptr) { FreeDataSpace(dev, ptr); }

TVMStreamHandle DeviceAPI::CreateStream(Device dev) { return nullptr; }

void DeviceAPI::FreeStream(Device dev, TVMStreamHandle stream) {}

TVMStreamHandle DeviceAPI::GetCurrentStream(Device dev) { return nullptr; }

void DeviceAPI::SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
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
//   {trace 0}       // two spaces in the beginning.
//   {trace 1}
//   {trace 2}
//--------------------------------------------------------
/*!
 * \brief Normalize error message
 *
 *  Parse them header generated by LOG(FATAL) and ICHECK
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
      std::string ending = ": ";
      size_t end_pos = line.find(ending, 13);
      if (end_pos == std::string::npos) return false;
      check_msg = line.substr(0, end_pos + ending.size());
      line = line.substr(end_pos + ending.size());
    }
    return true;
  };
  // if not in correct format, do not do any rewrite.
  if (!parse_log_header()) return err_msg;
  // Parse error type.
  {
    size_t start_pos = 0, end_pos;
    for (; start_pos < line.length() && line[start_pos] == ' '; ++start_pos) {
    }
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
      for (start_pos = end_pos + 1; start_pos < line.length() && line[start_pos] == ' ';
           ++start_pos) {
      }
      line = line.substr(start_pos);
    } else {
      // did not detect error_type, use default value.
      line = line.substr(start_pos);
      error_type = "TVMError";
    }
  }
  // Separate out stack trace.
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

struct WrappedPythonError : Error {
  WrappedPythonError() : Error("") {}
  explicit WrappedPythonError(WrappedPythonObject obj)
      : Error(""), obj(std::move(obj)), cpp_backtrace(tvm::runtime::Backtrace()) {}

  WrappedPythonObject obj;
  std::string cpp_backtrace;
};

struct TVMRuntimeEntry {
  std::string ret_str;
  TVMByteArray ret_bytes;

  std::variant<WrappedPythonError, InternalError, std::string> last_error;
  std::string last_error_formatted;
};

typedef dmlc::ThreadLocalStore<TVMRuntimeEntry> TVMAPIRuntimeStore;

const char* TVMGetLastError() {
  auto* store = TVMAPIRuntimeStore::Get();
  const auto& last_error = store->last_error;
  if (const auto* message = std::get_if<std::string>(&last_error)) {
    return message->c_str();
  } else if (const auto* internal = std::get_if<InternalError>(&last_error)) {
    // Use last_error_formatted to store the formatted error message, to avoid
    // dangling pointer.
    store->last_error_formatted = NormalizeError(internal->full_message());
    return store->last_error_formatted.c_str();
  } else {
    return nullptr;
  }
}

void* TVMGetLastPythonError() {
  auto& last_error = TVMAPIRuntimeStore::Get()->last_error;
  if (auto* wrapped = std::get_if<WrappedPythonError>(&last_error)) {
    return wrapped->obj.raw_pointer();
  } else {
    return nullptr;
  }
}

const char* TVMGetLastBacktrace() {
  const auto& last_error = TVMAPIRuntimeStore::Get()->last_error;
  if (const auto* wrapped = std::get_if<WrappedPythonError>(&last_error)) {
    return wrapped->cpp_backtrace.data();
  } else if (const auto* wrapped = std::get_if<InternalError>(&last_error)) {
    return wrapped->backtrace().data();
  } else {
    return nullptr;
  }
}

void TVMDropLastPythonError() {
  auto& last_error = TVMAPIRuntimeStore::Get()->last_error;
  if (std::get_if<WrappedPythonError>(&last_error)) {
    last_error = "";
  }
}

int TVMAPIHandleException(const std::exception& e) {
  auto& last_error = TVMAPIRuntimeStore::Get()->last_error;

  if (const auto* wrapped = dynamic_cast<const WrappedPythonError*>(&e)) {
    last_error = *wrapped;
  } else if (const auto* internal = dynamic_cast<const InternalError*>(&e)) {
    last_error = *internal;
  } else {
    last_error = NormalizeError(e.what());
  }
  return -1;
}

void TVMAPISetLastPythonError(void* obj) {
  auto& last_error = TVMAPIRuntimeStore::Get()->last_error;
  last_error = WrappedPythonError(WrappedPythonObject(obj));
}

void TVMThrowLastError() {
  auto& last_error = TVMAPIRuntimeStore::Get()->last_error;
  if (auto* wrapped = std::get_if<WrappedPythonError>(&last_error)) {
    WrappedPythonError wrapped_err;
    std::swap(wrapped_err, *wrapped);
    throw wrapped_err;
  } else if (auto* internal = std::get_if<InternalError>(&last_error)) {
    throw *internal;
  } else if (auto* message = std::get_if<std::string>(&last_error)) {
    throw tvm::Error(NormalizeError(*message) + tvm::runtime::Backtrace());
  }
}

void TVMAPISetLastError(const char* msg) {
  auto& last_error = TVMAPIRuntimeStore::Get()->last_error;
  last_error = msg;
}

int TVMModLoadFromFile(const char* file_name, const char* format, TVMModuleHandle* out) {
  API_BEGIN();
  TVMRetValue ret;
  ret = Module::LoadFromFile(file_name, format);
  TVMValue val;
  int type_code;
  ret.MoveToCHost(&val, &type_code);
  *out = val.v_handle;
  API_END();
}

int TVMModImport(TVMModuleHandle mod, TVMModuleHandle dep) {
  API_BEGIN();
  ObjectInternal::GetModuleNode(mod)->Import(GetRef<Module>(ObjectInternal::GetModuleNode(dep)));
  API_END();
}

int TVMModGetFunction(TVMModuleHandle mod, const char* func_name, int query_imports,
                      TVMFunctionHandle* func) {
  API_BEGIN();
  PackedFunc pf = ObjectInternal::GetModuleNode(mod)->GetFunction(func_name, query_imports != 0);
  if (pf != nullptr) {
    tvm::runtime::TVMRetValue ret;
    ret = pf;
    TVMValue val;
    int type_code;
    ret.MoveToCHost(&val, &type_code);
    *func = val.v_handle;
  } else {
    *func = nullptr;
  }
  API_END();
}

int TVMModFree(TVMModuleHandle mod) { return TVMObjectFree(mod); }

int TVMBackendGetFuncFromEnv(void* mod_node, const char* func_name, TVMFunctionHandle* func) {
  API_BEGIN();
  *func = (TVMFunctionHandle)(static_cast<ModuleNode*>(mod_node)->GetFuncFromEnv(func_name))->get();
  API_END();
}

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size, int dtype_code_hint,
                               int dtype_bits_hint) {
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;

  DLDataType type_hint;
  type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  type_hint.lanes = 1;

  return DeviceAPIManager::Get(dev)->AllocWorkspace(dev, static_cast<size_t>(size), type_hint);
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  DeviceAPIManager::Get(dev)->FreeWorkspace(dev, ptr);
  return 0;
}

int TVMBackendRunOnce(void** handle, int (*f)(void*), void* cdata, int nbytes) {
  if (*handle == nullptr) {
    *handle = reinterpret_cast<void*>(1);
    return (*f)(cdata);
  }
  return 0;
}

int TVMFuncFree(TVMFunctionHandle func) { return TVMObjectFree(func); }

int TVMByteArrayFree(TVMByteArray* arr) {
  if (arr == &TVMAPIRuntimeStore::Get()->ret_bytes) {
    return 0;  // Thread-local storage does not need explicit deleting.
  }

  delete arr;
  return 0;
}

int TVMFuncCall(TVMFunctionHandle func, TVMValue* args, int* arg_type_codes, int num_args,
                TVMValue* ret_val, int* ret_type_code) {
  API_BEGIN();
  TVMRetValue rv;
  (static_cast<const PackedFuncObj*>(func))
      ->CallPacked(TVMArgs(args, arg_type_codes, num_args), &rv);
  // handle return string.
  if (rv.type_code() == kTVMStr || rv.type_code() == kTVMDataType || rv.type_code() == kTVMBytes) {
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

int TVMCFuncSetReturn(TVMRetValueHandle ret, TVMValue* value, int* type_code, int num_ret) {
  API_BEGIN();
  ICHECK_EQ(num_ret, 1);
  TVMRetValue* rv = static_cast<TVMRetValue*>(ret);
  *rv = TVMArgValue(value[0], type_code[0]);
  API_END();
}

int TVMFuncCreateFromCFunc(TVMPackedCFunc func, void* resource_handle, TVMPackedCFuncFinalizer fin,
                           TVMFunctionHandle* out) {
  API_BEGIN();
  if (fin == nullptr) {
    tvm::runtime::TVMRetValue ret;
    ret = PackedFunc([func, resource_handle](TVMArgs args, TVMRetValue* rv) {
      int ret = func(const_cast<TVMValue*>(args.values), const_cast<int*>(args.type_codes),
                     args.num_args, rv, resource_handle);
      if (ret != 0) {
        TVMThrowLastError();
      }
    });
    TVMValue val;
    int type_code;
    ret.MoveToCHost(&val, &type_code);
    *out = val.v_handle;
  } else {
    // wrap it in a shared_ptr, with fin as deleter.
    // so fin will be called when the lambda went out of scope.
    std::shared_ptr<void> rpack(resource_handle, fin);
    tvm::runtime::TVMRetValue ret;
    ret = PackedFunc([func, rpack](TVMArgs args, TVMRetValue* rv) {
      int ret = func(const_cast<TVMValue*>(args.values), const_cast<int*>(args.type_codes),
                     args.num_args, rv, rpack.get());
      if (ret != 0) {
        TVMThrowLastError();
      }
    });
    TVMValue val;
    int type_code;
    ret.MoveToCHost(&val, &type_code);
    *out = val.v_handle;
  }
  API_END();
}

int TVMStreamCreate(int device_type, int device_id, TVMStreamHandle* out) {
  API_BEGIN();
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  *out = DeviceAPIManager::Get(dev)->CreateStream(dev);
  API_END();
}

int TVMStreamFree(int device_type, int device_id, TVMStreamHandle stream) {
  API_BEGIN();
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  DeviceAPIManager::Get(dev)->FreeStream(dev, stream);
  API_END();
}

int TVMSetStream(int device_type, int device_id, TVMStreamHandle stream) {
  API_BEGIN();
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  DeviceAPIManager::Get(dev)->SetStream(dev, stream);
  API_END();
}

int TVMSynchronize(int device_type, int device_id, TVMStreamHandle stream) {
  API_BEGIN();
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  DeviceAPIManager::Get(dev)->StreamSync(dev, stream);
  API_END();
}

int TVMStreamStreamSynchronize(int device_type, int device_id, TVMStreamHandle src,
                               TVMStreamHandle dst) {
  API_BEGIN();
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  DeviceAPIManager::Get(dev)->SyncStreamFromTo(dev, src, dst);
  API_END();
}

int TVMCbArgToReturn(TVMValue* value, int* code) {
  API_BEGIN();
  tvm::runtime::TVMRetValue rv;
  rv = tvm::runtime::TVMMovableArgValue_(*value, *code);
  rv.MoveToCHost(value, code);
  API_END();
}

int TVMDeviceAllocDataSpace(DLDevice dev, size_t nbytes, size_t alignment, DLDataType type_hint,
                            void** out_data) {
  API_BEGIN();
  out_data[0] = DeviceAPIManager::Get(dev)->AllocDataSpace(dev, nbytes, alignment, type_hint);
  API_END();
}

int TVMDeviceAllocDataSpaceWithScope(DLDevice dev, int ndim, const int64_t* shape, DLDataType dtype,
                                     const char* mem_scope, void** out_data) {
  API_BEGIN();
  Optional<String> scope;
  if (mem_scope != nullptr) {
    scope = String(std::string(mem_scope));
  }
  out_data[0] = DeviceAPIManager::Get(dev)->AllocDataSpace(dev, ndim, shape, dtype, scope);
  API_END();
}

int TVMDeviceFreeDataSpace(DLDevice dev, void* ptr) {
  API_BEGIN();
  DeviceAPIManager::Get(dev)->FreeDataSpace(dev, ptr);
  API_END();
}

int TVMDeviceCopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  API_BEGIN();
  DLDevice dev_from = from->device;
  DLDevice dev_to = to->device;
  DLDevice dev = dev_from.device_type != kDLCPU ? dev_from : dev_to;
  DeviceAPIManager::Get(dev)->CopyDataFromTo(from, to, stream);
  API_END();
}

// set device api
TVM_REGISTER_GLOBAL(tvm::runtime::symbol::tvm_set_device)
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLDevice dev;
      dev.device_type = static_cast<DLDeviceType>(args[0].operator int());
      dev.device_id = args[1];
      DeviceAPIManager::Get(dev)->SetDevice(dev);
    });

// set device api
TVM_REGISTER_GLOBAL("runtime.GetDeviceAttr").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(args[0].operator int());
  dev.device_id = args[1];

  DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[2].operator int());
  if (kind == kExist) {
    DeviceAPI* api = DeviceAPIManager::Get(dev.device_type, true);
    if (api != nullptr) {
      api->GetAttr(dev, kind, ret);
    } else {
      *ret = 0;
    }
  } else {
    DeviceAPIManager::Get(dev)->GetAttr(dev, kind, ret);
  }
});

TVM_REGISTER_GLOBAL("runtime.TVMSetStream").set_body_typed(TVMSetStream);
