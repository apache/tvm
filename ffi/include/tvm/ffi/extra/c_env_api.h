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
 * \file tvm/ffi/extra/c_env_api.h
 * \brief Extra environment API.
 */
#ifndef TVM_FFI_EXTRA_C_ENV_API_H_
#define TVM_FFI_EXTRA_C_ENV_API_H_

#include <tvm/ffi/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Stream context
// Focusing on minimalistic thread-local context recording stream being used.
// We explicitly not handle allocation/de-allocation of stream here.
// ----------------------------------------------------------------------------
typedef void* TVMFFIStreamHandle;

/*!
 * \brief FFI function to set the current stream for a device
 *
 * \param device_type The type of the device.
 * \param device_id The id of the device.
 * \param stream The stream to set.
 * \param opt_out_original_stream Output original stream if the address is not nullptr.
 * \note The stream is a weak reference that is cached/owned by the module.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvSetStream(int32_t device_type, int32_t device_id,
                                   TVMFFIStreamHandle stream,
                                   TVMFFIStreamHandle* opt_out_original_stream);

/*!
 * \brief FFI function to get the current stream for a device
 *
 * \param device_type The type of the device.
 * \param device_id The id of the device.
 * \return The current stream of the device.
 */
TVM_FFI_DLL TVMFFIStreamHandle TVMFFIEnvGetCurrentStream(int32_t device_type, int32_t device_id);

/*!
 * \brief Check if there are any signals raised in the surrounding env.
 * \return 0 when success, nonzero when failure happens
 * \note Under python this function redirects to PyErr_CheckSignals
 */
TVM_FFI_DLL int TVMFFIEnvCheckSignals();

/*!
 * \brief Register a symbol into the from the surrounding env such as python
 * \param name The name of the symbol.
 * \param symbol The symbol to register.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvRegisterCAPI(const char* name, void* symbol);

// ----------------------------------------------------------------------------
// Module symbol management in callee side
// ----------------------------------------------------------------------------
/*!
 * \brief FFI function to lookup a function from a module's imports.
 *
 * This is a helper function that is used by generated code.
 *
 * \param library_ctx The library context module handle.
 * \param func_name The name of the function.
 * \param out The result function.
 * \note The returned function is a weak reference that is cached/owned by the module.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvModLookupFromImports(TVMFFIObjectHandle library_ctx, const char* func_name,
                                              TVMFFIObjectHandle* out);

/*
 * \brief Register a symbol value that will be initialized when a library with the symbol is loaded.
 *
 * This function can be used to make context functions to be available in the library
 * module that wants to avoid an explicit link dependency
 *
 * \param name The name of the symbol.
 * \param symbol The symbol to register.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvModRegisterContextSymbol(const char* name, void* symbol);

/*!
 * \brief Register a symbol that will be initialized when a system library is loaded.
 *
 * \param name The name of the symbol.
 * \param symbol The symbol to register.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvModRegisterSystemLibSymbol(const char* name, void* symbol);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // TVM_FFI_EXTRA_C_ENV_API_H_
