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

/*
 * \file tvmjs_support.cc
 * \brief Support functions to be linked with wasm_runtime to provide
 *        PackedFunc callbacks in tvmjs.
 *        We do not need to link this file in standalone wasm.
 */

// configurations for the dmlc log.
#define DMLC_LOG_CUSTOMIZE 0
#define DMLC_LOG_STACK_TRACE 0
#define DMLC_LOG_DEBUG 0
#define DMLC_LOG_NODATE 1
#define DMLC_LOG_FATAL_THROW 0


#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/device_api.h>

extern "C" {
// --- Additional C API for the Wasm runtime ---
/*!
 * \brief Allocate space aligned to 64 bit.
 * \param size The size of the space.
 * \return The allocated space.
 */
TVM_DLL void* TVMWasmAllocSpace(int size);

/*!
 * \brief Free the space allocated by TVMWasmAllocSpace.
 * \param data The data pointer.
 */
TVM_DLL void TVMWasmFreeSpace(void* data);

/*!
 * \brief Create PackedFunc from a resource handle.
 * \param resource_handle The handle to the resource.
 * \param out The output PackedFunc.
 * \sa TVMWasmPackedCFunc, TVMWasmPackedCFuncFinalizer
3A * \return 0 if success.
 */
TVM_DLL int TVMWasmFuncCreateFromCFunc(void* resource_handle,
                                       TVMFunctionHandle *out);

// --- APIs to be implemented by the frontend. ---
/*!
 * \brief Wasm frontend packed function caller.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param ret The return value handle.
 * \param resource_handle The handle additional resouce handle from fron-end.
 * \return 0 if success, -1 if failure happens, set error via TVMAPISetLastError.
 */
extern int TVMWasmPackedCFunc(TVMValue* args,
                              int* type_codes,
                              int num_args,
                              TVMRetValueHandle ret,
                              void* resource_handle);

/*!
 * \brief Wasm frontend resource finalizer.
 * \param resource_handle The pointer to the external resource.
 */
extern void TVMWasmPackedCFuncFinalizer(void* resource_handle);
}  // extern "C"


void* TVMWasmAllocSpace(int size) {
  int num_count = (size + 7) / 8;
  return new int64_t[num_count];
}

void TVMWasmFreeSpace(void* arr) {
  delete[] static_cast<int64_t*>(arr);
}

int TVMWasmFuncCreateFromCFunc(void* resource_handle,
                               TVMFunctionHandle *out) {
  return TVMFuncCreateFromCFunc(
    TVMWasmPackedCFunc, resource_handle,
    TVMWasmPackedCFuncFinalizer, out);
}


namespace tvm {
namespace runtime {

// chrono in the WASI does not provide very accurate time support
// and also have problems in the i64 support in browser.
// We redirect the timer to a JS side time using performance.now
PackedFunc WrapWasmTimeEvaluator(PackedFunc pf,
                                 TVMContext ctx,
                                 int number,
                                 int repeat,
                                 int min_repeat_ms) {
  auto ftimer = [pf, ctx, number, repeat, min_repeat_ms](
      TVMArgs args, TVMRetValue *rv) {

    TVMRetValue temp;
    auto finvoke = [&](int n) {
      // start timing
      for (int i = 0; i < n; ++i) {
        pf.CallPacked(args, &temp);
      }
      DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
    };

    auto* get_timer = runtime::Registry::Get("wasm.GetTimer");
    CHECK(get_timer != nullptr) << "Cannot find wasm.GetTimer in the global function";
    TypedPackedFunc<double(int number)> timer_ms = (*get_timer)(
        TypedPackedFunc<void(int)>(finvoke));

    std::ostringstream os;
    finvoke(1);

    int setup_number = number;

    for (int i = 0; i < repeat; ++i) {
      double duration_ms = 0.0;

      do {
        if (duration_ms > 0.0) {
          setup_number = static_cast<int>(
              std::max((min_repeat_ms / (duration_ms / number) + 1),
                       number * 1.618));   // 1.618 is chosen by random
        }
        duration_ms = timer_ms(setup_number);
      } while (duration_ms < min_repeat_ms);

      double speed = duration_ms / setup_number / 1000;
      os.write(reinterpret_cast<char*>(&speed), sizeof(speed));
    }

    std::string blob = os.str();
    TVMByteArray arr;
    arr.size = blob.length();
    arr.data = blob.data();
    // return the time.
    *rv = arr;
  };
  return PackedFunc(ftimer);
}

TVM_REGISTER_GLOBAL("wasm.RPCTimeEvaluator")
.set_body_typed([](Optional<Module> opt_mod,
                   std::string name,
                   int device_type,
                   int device_id,
                   int number,
                   int repeat,
                   int min_repeat_ms) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;

  if (opt_mod.defined()) {
    Module m = opt_mod.value();
    std::string tkey = m->type_key();
    return WrapWasmTimeEvaluator(
        m.GetFunction(name, false), ctx, number, repeat, min_repeat_ms);
  } else {
    auto* pf = runtime::Registry::Get(name);
    CHECK(pf != nullptr) << "Cannot find " << name << " in the global function";
    return WrapWasmTimeEvaluator(
        *pf, ctx, number, repeat, min_repeat_ms);
  }
});

}  // namespace runtime
}  // namespace tvm
