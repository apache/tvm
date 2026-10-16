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
 * \file tvm/runtime/c_backend_api.h
 * \brief TVM runtime backend API.
 *
 *  The functions defined in this header are intended to be
 *  used by compiled tvm operators, usually user do not need to use these
 *  function directly.
 */
#ifndef TVM_RUNTIME_C_BACKEND_API_H_
#define TVM_RUNTIME_C_BACKEND_API_H_

#include <tvm/runtime/base.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Select the runtime device for the calling host thread.
 *
 * Dispatches to DeviceAPI::SetDevice with the given local DLPack device type
 * and runtime device index (DLDevice::device_id). This does not save or restore
 * the previous selection. Backends without device-selection state may treat
 * selection as a no-op. Valid indices and index validation are backend-specific.
 *
 * \param device_type Local DLPack device type, not an RPC session-encoded type.
 * \param device_index Nonnegative runtime device index, not a device type or
 * target configuration index.
 * \return 0 on success, -1 on invalid arguments, unavailable backend, or backend
 * failure. Failure records the raised TVM FFI error on the calling thread; no
 * C++ exception crosses this C entry point.
 */
TVM_RUNTIME_DLL int TVMBackendSetDevice(int device_type, int device_index);

/*!
 * \brief Query the selected runtime device index on the calling host thread.
 *
 * Dispatches to DeviceAPI::GetDeviceIndex without changing device selection.
 * Getter support is optional and independent of setter support: the default
 * implementation reports NotImplementedError, including for backends without
 * device-selection state. RPC devices are not supported.
 *
 * \param device_type Local DLPack device type, not an RPC session-encoded type.
 * \param out_device_index Non-null pointer receiving the runtime device index
 * (DLDevice::device_id), not a device type or target configuration index. Written
 * only on success; its previous value is unchanged on failure.
 * \return 0 on success, -1 on invalid arguments, unavailable backend, unsupported
 * query, or backend failure. Failure records the raised TVM FFI error on the
 * calling thread; no C++ exception crosses this C entry point.
 */
TVM_RUNTIME_DLL int TVMBackendGetDeviceIndex(int device_type, int* out_device_index);

/*!
 * \brief Backend function to allocate temporal workspace.
 *
 * \note The result allocated space is ensured to be aligned to kTempAllocaAlignment.
 *
 * \param nbytes The size of the space requested.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \param dtype_code_hint The type code of the array elements. Only used in
 * certain backends such as OpenGL.
 * \param dtype_bits_hint The type bits of the array elements. Only used in
 * certain backends such as OpenGL.
 * \return A valid pointer on success, nullptr on failure. Failure records the
 * raised TVM FFI error on the calling thread; no C++ exception crosses this C
 * entry point.
 */
TVM_RUNTIME_DLL void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                                               int dtype_code_hint, int dtype_bits_hint);

/*!
 * \brief Backend function to free temporal workspace.
 *
 * \param ptr The result allocated space pointer.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return 0 on success, -1 on failure. Failure records the raised TVM FFI error
 * on the calling thread; no C++ exception crosses this C entry point.
 *
 * \sa TVMBackendAllocWorkspace
 */
TVM_RUNTIME_DLL int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

/*!
 * \brief Environment for TVM parallel task.
 */
typedef struct {
  /*!
   * \brief Auxiliary used for synchronization
   */
  void* sync_handle;
  /*! \brief total amount of task */
  int32_t num_task;
} TVMParallelGroupEnv;

/*!
 * \brief The callback function to execute a parallel lambda
 * \param task_id the task id of the function.
 * \param penv The parallel environment backs the execution.
 * \param cdata The supporting closure data.
 */
typedef int (*FTVMParallelLambda)(int task_id, TVMParallelGroupEnv* penv, void* cdata);

/*!
 * \brief Backend function for running parallel jobs.
 *
 * \param flambda The parallel function to be launched.
 * \param cdata The closure data.
 * \param num_task Number of tasks to launch, can be 0, means launch
 *           with all available threads.
 *
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_RUNTIME_DLL int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task);

/*!
 * \brief BSP barrier between parallel threads
 * \param task_id the task id of the function.
 * \param penv The parallel environment backs the execution.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_RUNTIME_DLL int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv);

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_RUNTIME_C_BACKEND_API_H_
