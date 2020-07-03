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

// LINT_C_FILE

/*!
 * \file tvm/runtime/crt/func_registry.c
 * \brief Defines implementations of generic string-based function lookup structs
 */

#include <stdio.h>
#include <string.h>
#include <tvm/runtime/crt/func_registry.h>

/*!
 * \brief strcmp against the next string in the registry, and return the end.
 *
 * Regardless of return value, after calling this function, cursor's value will be modified to
 * point at the \0 at the end of the string it currently points to.
 *
 * \param cursor Pointer to cursor to first string to compare.
 * \param name Pointer to reference string.
 * \return 0 if the string pointed to by cursor == name; non-zero otherwise.
 */
int strcmp_cursor(const char** cursor, const char* name) {
  int return_value = 0;
  while (return_value == 0) {
    char c = **cursor;
    char n = *name;
    return_value = ((int)c) - ((int)n);

    if (n == 0 || c == 0) {
      break;
    }

    name++;
    (*cursor)++;
  }

  while (**cursor != 0) {
    (*cursor)++;
  }

  return return_value;
}

tvm_crt_error_t TVMFuncRegistry_Lookup(const TVMFuncRegistry* reg, const char* name,
                                       tvm_function_index_t* function_index) {
  tvm_function_index_t idx;
  const char* reg_name_ptr;

  idx = 0;
  // NOTE: reg_name_ptr starts at index 1 to skip num_funcs.
  for (reg_name_ptr = reg->names + 1; *reg_name_ptr != '\0'; reg_name_ptr++) {
    if (!strcmp_cursor(&reg_name_ptr, name)) {
      *function_index = idx;
      return kTvmErrorNoError;
    }

    idx++;
  }

  return kTvmErrorFunctionNameNotFound;
}

tvm_crt_error_t TVMFuncRegistry_GetByIndex(const TVMFuncRegistry* reg,
                                           tvm_function_index_t function_index,
                                           TVMBackendPackedCFunc* out_func) {
  uint8_t num_funcs;

  num_funcs = reg->names[0];
  if (function_index >= num_funcs) {
    return kTvmErrorFunctionIndexInvalid;
  }

  *out_func = reg->funcs[function_index];
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMMutableFuncRegistry_Create(TVMMutableFuncRegistry* reg, uint8_t* buffer,
                                              size_t buffer_size_bytes) {
  if (buffer_size_bytes < kTvmAverageFuncEntrySizeBytes) {
    return kTvmErrorBufferTooSmall;
  }

  memset(reg, 0, sizeof(*reg));
  reg->registry.names = (const char*)buffer;
  buffer[0] = 0;  // number of functions present in buffer.
  buffer[1] = 0;  // end of names list marker.

  // compute a guess of the average size of one entry:
  //  - assume average function name is around ~10 bytes
  //  - 1 byte for \0
  //  - size of 1 function pointer
  reg->max_functions = buffer_size_bytes / kTvmAverageFuncEntrySizeBytes;
  reg->registry.funcs =
      (TVMBackendPackedCFunc*)(buffer + buffer_size_bytes - reg->max_functions * sizeof(void*));

  return kTvmErrorNoError;
}

tvm_crt_error_t TVMMutableFuncRegistry_Set(TVMMutableFuncRegistry* reg, const char* name,
                                           TVMBackendPackedCFunc func, int override) {
  size_t idx;
  char* reg_name_ptr;

  idx = 0;
  // NOTE: safe to discard const qualifier here, since reg->registry.names was set from
  // TVMMutableFuncRegistry_Create above.
  // NOTE: reg_name_ptr starts at index 1 to skip num_funcs.
  for (reg_name_ptr = (char*)reg->registry.names + 1; *reg_name_ptr != 0; reg_name_ptr++) {
    if (!strcmp_cursor((const char**)&reg_name_ptr, name)) {
      if (override == 0) {
        return kTvmErrorFunctionAlreadyDefined;
      }
      ((TVMBackendPackedCFunc*)reg->registry.funcs)[idx] = func;
      return kTvmErrorNoError;
    }

    idx++;
  }

  size_t name_len = strlen(name);
  ssize_t names_bytes_remaining = ((const char*)reg->registry.funcs) - reg_name_ptr;
  if (idx >= reg->max_functions || name_len + 1 > names_bytes_remaining) {
    return kTvmErrorFunctionRegistryFull;
  }

  memcpy(reg_name_ptr, name, name_len + 1);
  reg_name_ptr += name_len + 1;
  *reg_name_ptr = 0;
  ((TVMBackendPackedCFunc*)reg->registry.funcs)[idx] = func;
  ((char*)reg->registry.names)[0]++;  // increment num_funcs.

  return kTvmErrorNoError;
}
