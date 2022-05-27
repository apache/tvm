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
 * \file include/tvm/runtime/crt/func_registry.h
 * \brief Defines generic string-based function lookup structs
 */
#ifndef TVM_RUNTIME_CRT_FUNC_REGISTRY_H_
#define TVM_RUNTIME_CRT_FUNC_REGISTRY_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/crt/error_codes.h>

typedef uint16_t tvm_function_index_t;

typedef uint16_t tvm_module_index_t;

/*!
 * \brief A data structure that facilitates function lookup by C-string name.
 */
typedef struct TVMFuncRegistry {
  /*! \brief Names of registered functions, concatenated together and separated by \0.
   * An additional \0 is present at the end of the concatenated blob to mark the end.
   *
   * Byte 0 and 1 are the number of functions in `funcs`.
   */
  const char* names;

  /*! \brief Function pointers, in the same order as their names in `names`. */
  const TVMBackendPackedCFunc* funcs;
} TVMFuncRegistry;

/*!
 * \brief Get the of the number of functions from registry.
 *
 * \param reg TVMFunctionRegistry instance that contains the function.
 * \return The number of functions from registry.
 */
uint16_t TVMFuncRegistry_GetNumFuncs(const TVMFuncRegistry* reg);

/*!
 * \brief Set the number of functions to registry.
 *
 * \param reg TVMFunctionRegistry instance that contains the function.
 * \param num_funcs The number of functions
 * \return 0 when successful.
 */
int TVMFuncRegistry_SetNumFuncs(const TVMFuncRegistry* reg, const uint16_t num_funcs);

/*!
 * \brief Get the address of 0th function from registry.
 *
 * \param reg TVMFunctionRegistry instance that contains the function.
 * \return the address of 0th function from registry
 */
const char* TVMFuncRegistry_Get0thFunctionName(const TVMFuncRegistry* reg);

/*!
 * \brief Get packed function from registry by name.
 *
 * \param reg TVMFunctionRegistry instance that contains the function.
, * \param name The function name
 * \param function_index Pointer to receive the 0-based index of the function in the registry, if it
 *     was found. Unmodified otherwise.
 * \return kTvmErrorNoError when successful. kTvmErrorFunctionNameNotFound when no function matched
`name`.
 */
tvm_crt_error_t TVMFuncRegistry_Lookup(const TVMFuncRegistry* reg, const char* name,
                                       tvm_function_index_t* function_index);

/*!
 * \brief Fetch TVMBackendPackedCFunc given a function index
 *
 * \param reg TVMFunctionRegistry instance that contains the function.
 * \param index Index of the function.
 * \param out_func Pointer which receives the function pointer at `index`, if a valid
 *      index was given. Unmodified otherwise.
 * \return kTvmErrorNoError when successful. kTvmErrorFunctionIndexInvalid when index was out of
 * range.
 */
tvm_crt_error_t TVMFuncRegistry_GetByIndex(const TVMFuncRegistry* reg, tvm_function_index_t index,
                                           TVMBackendPackedCFunc* out_func);

/*!
 * \brief A TVMFuncRegistry that supports adding and changing the functions.
 */
typedef struct TVMMutableFuncRegistry {
  TVMFuncRegistry registry;

  /*! \brief maximum number of functions in this registry. */
  size_t max_functions;
} TVMMutableFuncRegistry;

// Defined to work around compiler limitations.
#define TVM_AVERAGE_FUNCTION_NAME_STRLEN_BYTES 10

/*!
 * \brief Size of an average function name in a TVMMutableFuncRegistry, in bytes.
 *
 * This is just an assumption made by the runtime for ease of use.
 */
static const size_t kTvmAverageFunctionNameStrlenBytes = TVM_AVERAGE_FUNCTION_NAME_STRLEN_BYTES;

/*!
 * \brief Size of an average entry in a TVMMutableFuncRegistry, in bytes.
 *
 * Assumes a constant average function name length.
 */
static const size_t kTvmAverageFuncEntrySizeBytes =
    TVM_AVERAGE_FUNCTION_NAME_STRLEN_BYTES + 1 + sizeof(void*);

/*!
 * \brief Create a new mutable function registry from a block of memory.
 *
 * \param reg TVMMutableFuncRegistry to create.
 * \param buffer Backing memory available for this function registry.
 * \param buffer_size_bytes Number of bytes available in buffer.
 * \return kTvmErrorNoError when successful. kTvmErrorBufferTooSmall when buffer_size_bytes is so
 *      small that a single function cannot be registered.
 */
tvm_crt_error_t TVMMutableFuncRegistry_Create(TVMMutableFuncRegistry* reg, uint8_t* buffer,
                                              size_t buffer_size_bytes);

/*!
 * \brief Add or set a function in the registry.
 *
 * \param reg The mutable function registry to affect.
 * \param name Name of the function.
 * \param func The function pointer.
 * \param override non-zero if an existing entry should be overridden.
 * \return kTvmErrorNoError when successful. kTvmErrorRegistryFull when `reg` already contains
 *     `max_functions` entries. kTvmErrorFunctionAlreadyDefined when a function named `name` is
 * already present in the registry, and `override` == 0.
 */
tvm_crt_error_t TVMMutableFuncRegistry_Set(TVMMutableFuncRegistry* reg, const char* name,
                                           TVMBackendPackedCFunc func, int override);

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_CRT_FUNC_REGISTRY_H_
