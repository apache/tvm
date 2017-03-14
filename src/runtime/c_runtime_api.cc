/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <dmlc/timer.h>
#include <array>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <thread>
#include <mutex>
#include "./runtime_base.h"
#include "./device_api.h"

namespace tvm {
namespace runtime {

class DeviceAPIManager {
 public:
  static const int kMaxDeviceAPI = 16;
  // Get API
  static DeviceAPI* Get(TVMContext ctx) {
    return Global()->GetAPI(ctx.device_type);
  }

 private:
  std::array<DeviceAPI*, kMaxDeviceAPI> api_;
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
  DeviceAPI* GetAPI(DLDeviceType type) {
    if (api_[type] != nullptr) return api_[type];
    std::lock_guard<std::mutex> lock(mutex_);
    if (api_[type] != nullptr) return api_[type];
    std::string factory = "_device_api_" + DeviceName(type);
    auto* f = Registry::Get(factory);
    CHECK(f != nullptr)
        << "Device API " << DeviceName(type) << " is not enabled.";
    void* ptr = (*f)();
    api_[type] = static_cast<DeviceAPI*>(ptr);
    return api_[type];
  }
};


inline TVMArray* TVMArrayCreate_() {
  TVMArray* arr = new TVMArray();
  arr->shape = nullptr;
  arr->strides = nullptr;
  arr->ndim = 0;
  arr->data = nullptr;
  return arr;
}

inline void TVMArrayFree_(TVMArray* arr) {
  if (arr != nullptr) {
    // ok to delete nullptr
    delete[] arr->shape;
    delete[] arr->strides;
    if (arr->data != nullptr) {
      DeviceAPIManager::Get(arr->ctx)->FreeDataSpace(
          arr->ctx, arr->data);
    }
  }
  delete arr;
}

inline void VerifyType(TVMType dtype) {
  CHECK_GE(dtype.lanes, 1U);
  if (dtype.code == kFloat) {
    CHECK_EQ(dtype.bits % 32U, 0U);
  } else {
    CHECK_EQ(dtype.bits % 8U, 0U);
  }
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataSize(TVMArray* arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr->ndim; ++i) {
    size *= arr->shape[i];
  }
  size *= (arr->dtype.bits / 8) * arr->dtype.lanes;
  return size;
}

inline size_t GetDataAlignment(TVMArray* arr) {
  size_t align = (arr->dtype.bits / 8) * arr->dtype.lanes;
  if (align < 8) return 8;
  return align;
}

}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

struct TVMRuntimeEntry {
  std::string ret_str;
  std::string last_error;
  // threads used in parallel for
  std::vector<std::thread> par_threads;
  // errors created in parallel for.
  std::vector<std::string> par_errors;
  // number of parallel threads
  int num_par_threads{1};

  TVMRuntimeEntry() {
    const char *val = getenv("TVM_NUM_THREADS");
    if (val == nullptr) {
      val = getenv("OMP_NUM_THREADS");
    }
    if (val != nullptr) {
      num_par_threads = atoi(val);
    } else {
      num_par_threads = std::thread::hardware_concurrency() / 2;
    }
  }
};

typedef dmlc::ThreadLocalStore<TVMRuntimeEntry> TVMAPIRuntimeStore;

const char *TVMGetLastError() {
  return TVMAPIRuntimeStore::Get()->last_error.c_str();
}

void TVMAPISetLastError(const char* msg) {
  TVMAPIRuntimeStore::Get()->last_error = msg;
}

int TVMModLoadFromFile(const char* file_name,
                       const char* format,
                       TVMModuleHandle* out) {
  API_BEGIN();
  Module m = Module::LoadFromFile(file_name, format);
  *out = new Module(m);
  API_END();
}

int TVMModImport(TVMModuleHandle mod,
                 TVMModuleHandle dep) {
  API_BEGIN();
  static_cast<Module*>(mod)->Import(
      *static_cast<Module*>(dep));
  API_END();
}

int TVMModGetFunction(TVMModuleHandle mod,
                      const char* func_name,
                      int query_imports,
                      TVMFunctionHandle *func) {
  API_BEGIN();
  PackedFunc pf = static_cast<Module*>(mod)->GetFunction(
      func_name, query_imports != 0);
  if (pf != nullptr) {
    *func = new PackedFunc(pf);
  } else {
    *func = nullptr;
  }
  API_END();
}

int TVMModPreCompile(TVMModuleHandle mod,
                     const char* func_name,
                     TVMContext ctx) {
  API_BEGIN();
  (*static_cast<Module*>(mod))->PreCompile(func_name, ctx);
  API_END();
}

int TVMModFree(TVMModuleHandle mod) {
  API_BEGIN();
  delete static_cast<Module*>(mod);
  API_END();
}

int TVMBackendGetFuncFromEnv(void* mod_node,
                             const char* func_name,
                             TVMFunctionHandle *func) {
  API_BEGIN();
  *func = (TVMFunctionHandle)(
      static_cast<ModuleNode*>(mod_node)->GetFuncFromEnv(func_name));
  API_END();
}

int TVMBackendParallelFor(
    int64_t begin,
    int64_t end,
    int (*lambda)(int64_t begin, int64_t end, void* env),
    void* env) {
  TVMRuntimeEntry* rt = TVMAPIRuntimeStore::Get();
  int nthread = rt->num_par_threads;
  rt->par_threads.resize(nthread);
  rt->par_errors.clear();
  rt->par_errors.resize(nthread);
  int64_t step = (end - begin + nthread - 1) / nthread;
  auto fexec = [lambda, env, begin, end, step, rt](int i) {
    int64_t ibegin = std::min(end, begin + step * i);
    int64_t iend = std::min(end, begin + step * (i + 1));
    int rv = (*lambda)(ibegin, iend, env);
    if (rv != 0) {
      std::ostringstream os;
      os << "Thread " << i << " error:" << TVMGetLastError();
      rt->par_errors[i] = os.str();
    }
  };
  for (int i = 0; i < nthread; ++i) {
    rt->par_threads[i] = std::thread(fexec, i);
  }
  int ret = 0;
  for (int i = 0; i < nthread; ++i) {
    rt->par_threads[i].join();
    if (rt->par_errors[i].length() != 0) ret = -1;
  }
  if (ret == 0) return ret;
  std::ostringstream os;
  for (int i = 0; i < nthread; ++i) {
    if (rt->par_errors[i].length() != 0) {
      os << rt->par_errors[i] << '\n';
    }
  }
  rt->last_error = os.str();
  return -1;
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
  if (rv.type_code() == kStr ||
      rv.type_code() == kTVMType) {
    TVMRuntimeEntry* e = TVMAPIRuntimeStore::Get();
    e->ret_str = rv.operator std::string();
    *ret_type_code = kStr;
    ret_val->v_str = e->ret_str.c_str();
  } else {
    rv.MoveToCHost(ret_val, ret_type_code);
  }
  API_END();
}

int TVMCFuncSetReturn(TVMRetValueHandle ret,
                      TVMValue value,
                      int type_code) {
  API_BEGIN();
  TVMRetValue* rv = static_cast<TVMRetValue*>(ret);
  *rv = TVMArgValue(value, type_code);
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
          func((TVMValue*)args.values, (int*)args.type_codes, // NOLINT(*)
               args.num_args, rv, resource_handle);
        });
  } else {
    // wrap it in a shared_ptr, with fin as deleter.
    // so fin will be called when the lambda went out of scope.
    std::shared_ptr<void> rpack(resource_handle, fin);
    *out = new PackedFunc(
        [func, rpack](TVMArgs args, TVMRetValue* rv) {
          func((TVMValue*)args.values, (int*)args.type_codes, // NOLINT(*)
               args.num_args, rv, rpack.get());
      });
  }
  API_END();
}

int TVMArrayAlloc(const tvm_index_t* shape,
                  tvm_index_t ndim,
                  TVMType dtype,
                  TVMContext ctx,
                  TVMArrayHandle* out) {
  TVMArray* arr = nullptr;
  API_BEGIN();
  // shape
  arr = TVMArrayCreate_();
  // ndim
  arr->ndim = ndim;
  // dtype
  VerifyType(dtype);
  arr->dtype = dtype;
  tvm_index_t* shape_copy = new tvm_index_t[ndim];
  std::copy(shape, shape + ndim, shape_copy);
  arr->shape = shape_copy;
  // ctx
  arr->ctx = ctx;
  size_t size = GetDataSize(arr);
  size_t alignment = GetDataAlignment(arr);
  arr->data = DeviceAPIManager::Get(ctx)->AllocDataSpace(
      ctx, size, alignment);
  *out = arr;
  API_END_HANDLE_ERROR(TVMArrayFree_(arr));
}

int TVMArrayFree(TVMArrayHandle handle) {
  API_BEGIN();
  TVMArray* arr = handle;
  TVMArrayFree_(arr);
  API_END();
}

int TVMArrayCopyFromTo(TVMArrayHandle from,
                       TVMArrayHandle to,
                       TVMStreamHandle stream) {
  API_BEGIN();
  size_t from_size = GetDataSize(from);
  size_t to_size = GetDataSize(to);
  CHECK_EQ(from_size, to_size)
      << "TVMArrayCopyFromTo: The size must exactly match";
  TVMContext ctx = from->ctx;
  if (ctx.device_type == kCPU) {
    ctx = to->ctx;
  } else {
    CHECK(to->ctx.device_type == kCPU ||
          to->ctx.device_type == from->ctx.device_type)
        << "Can not copy across different ctx types directly";
  }
  DeviceAPIManager::Get(ctx)->CopyDataFromTo(
      from->data, to->data, from_size,
      from->ctx, to->ctx, stream);
  API_END();
}

int TVMSynchronize(TVMContext ctx, TVMStreamHandle stream) {
  API_BEGIN();
  DeviceAPIManager::Get(ctx)->StreamSync(ctx, stream);
  API_END();
}
