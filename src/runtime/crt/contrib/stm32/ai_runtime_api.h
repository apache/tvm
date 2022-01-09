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
 * \file ai_runtime_api.h
 * \brief The runtime API for the TVM generated C code.
 */

#ifndef TVM_RUNTIME_CRT_CONTRIB_STM32_AI_RUNTIME_API_H_
#define TVM_RUNTIME_CRT_CONTRIB_STM32_AI_RUNTIME_API_H_

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#include "dlpack/dlpack.h"              // From TVM
#include "tvm/runtime/c_runtime_api.h"  // From TVM

//
// This describes current ai_runtime version
//
#define AI_PLATFORM_RUNTIME_MAJOR 1
#define AI_PLATFORM_RUNTIME_MINOR 0
#define AI_PLATFORM_RUNTIME_MICRO 0

#define AI_STATIC static

#if defined(_MSC_VER)
#define AI_INLINE __inline
#define AI_API_ENTRY __declspec(dllexport)
#define AI_ALIGNED(x) /* AI_ALIGNED(x) */
#elif defined(__ICCARM__) || defined(__IAR_SYSTEMS_ICC__)
#define AI_INLINE inline
#define AI_API_ENTRY /* AI_API_ENTRY */
#define AI_ALIGNED(x) AI_CONCAT(AI_ALIGNED_, x)
#elif defined(__CC_ARM)
#define AI_INLINE __inline
#define AI_API_ENTRY __attribute__((visibility("default")))
#define AI_ALIGNED(x) __attribute__((aligned(x)))
/* Keil disallows anonymous union initialization by default */
#pragma anon_unions
#elif defined(__GNUC__)
#define AI_INLINE __inline
#define AI_API_ENTRY __attribute__((visibility("default")))
#define AI_ALIGNED(x) __attribute__((aligned(x)))
#else
/* Dynamic libraries are not supported by the compiler */
#define AI_API_ENTRY  /* AI_API_ENTRY */
#define AI_ALIGNED(x) /* AI_ALIGNED(x) */
#endif

/*********************************************************/

typedef void* ai_handle;

#define AI_HANDLE_PTR(ptr_) ((ai_handle)(ptr_))
#define AI_HANDLE_NULL AI_HANDLE_PTR(NULL)

typedef uint8_t* ai_ptr;

typedef enum { AI_STATUS_OK = 0, AI_STATUS_ERROR = 1, AI_STATUS_DELEGATE_ERROR = 2 } ai_status;

// =======================================================
//                  ai_quantization_info
//
//   Parameters for asymmetric quantization across a dimension (i.e
//   per output channel quantization).
//   quantized_dimension specifies which dimension the scales and
//   zero_points correspond to.
//   For a particular value in quantized_dimension, quantized values
//   can be converted back to float using:
//     real_value = scale * (quantized_value - zero_point)
// =======================================================

typedef struct {
  /*!
   * \brief The quantization info, if quantized
   */
  float* scale;
  int32_t* zero_point;
  int32_t dim;
} ai_quantization_info;

// =======================================================
//                       ai_tensor
// =======================================================

typedef struct {
  /*!
   * \brief The TVM tensor.
   */
  DLTensor dltensor;
  /*!
   * \brief The quantization info, if quantized
   */
  ai_quantization_info* quant;
} ai_tensor;

// =======================================================
//   get_dltensor
// =======================================================
AI_STATIC AI_INLINE DLTensor* get_dltensor(ai_tensor* tensor) { return &tensor->dltensor; }

// =======================================================
//   get_tensor_elts
// =======================================================
AI_STATIC AI_INLINE uint32_t get_tensor_elts(const ai_tensor* tensor) {
  const DLTensor* t = &tensor->dltensor;
  uint32_t elts = 1;
  for (int i = 0; i < t->ndim; ++i) {
    elts *= t->shape[i];
  }
  return elts;
}

// =======================================================
//   get_tensor_size
// =======================================================
AI_STATIC AI_INLINE uint32_t get_tensor_size(const ai_tensor* tensor) {
  const DLTensor* t = &tensor->dltensor;
  uint32_t size = 1;
  for (int i = 0; i < t->ndim; ++i) {
    size *= t->shape[i];
  }
  size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
  return size;
}

// =======================================================
//                    ai_network_info
// =======================================================

typedef struct {
  const char* name;
  const char* datetime;
  const char* revision;
  const char* tool_version;
  const char* api_version;
  uint16_t n_nodes;
  uint8_t n_inputs;
  uint8_t n_outputs;
  uint32_t activations_size;
  uint32_t params_size;
  ai_ptr activations;
  ai_tensor** inputs;
  ai_tensor** outputs;
  const ai_ptr (*ai_get_params)(void);
  ai_status (*ai_create)(const ai_ptr weights, const ai_ptr activations);
  ai_status (*ai_destroy)();
  ai_status (*ai_run)(ai_tensor* input[], ai_tensor* output[]);
} ai_model_info;

#define AI_MODEL_name(x) (x->name)
#define AI_MODEL_datetime(x) (x->datetime)
#define AI_MODEL_revision(x) (x->revision)
#define AI_MODEL_tool_version(x) (x->tool_version)
#define AI_MODEL_api_version(x) (x->api_version)
#define AI_MODEL_n_nodes(x) (x->n_nodes)
#define AI_MODEL_n_inputs(x) (x->n_inputs)
#define AI_MODEL_n_outputs(x) (x->n_outputs)
#define AI_MODEL_activations_size(x) (x->activations_size)
#define AI_MODEL_params_size(x) (x->params_size)
#define AI_MODEL_inputs(x) (x->inputs)
#define AI_MODEL_outputs(x) (x->outputs)
#define AI_MODEL_activations(x) (x->activations)

// =======================================================
//                         Iterator
//
//   Usage:
//
//     for (ai_models_iterator it = ai_models_iterator_begin();
//          it != ai_models_iterator_end();
//          it = ai_models_iterator_next(it)) {
//       const char * name = ai_models_iterator_value(it);
//     }
//
// =======================================================

typedef uint32_t ai_model_iterator;

ai_model_iterator ai_model_iterator_begin();
ai_model_iterator ai_model_iterator_next(ai_model_iterator it);
ai_model_iterator ai_model_iterator_end();
ai_model_info* ai_model_iterator_value(ai_model_iterator it);

// =======================================================
//                    External Interface
// =======================================================

ai_status ai_create(ai_model_info* nn, ai_ptr activations, ai_handle* handle);

ai_status ai_destroy(ai_handle handle);

const char* ai_get_error(ai_handle handle);

int32_t ai_get_input_size(ai_handle handle);

int32_t ai_get_output_size(ai_handle handle);

ai_tensor* ai_get_input(ai_handle handle, int32_t index);

ai_tensor* ai_get_output(ai_handle handle, int32_t index);

ai_status ai_run(ai_handle handle);

//
// Additional methods
//
const char* ai_get_name(ai_handle handle);
const char* ai_get_datetime(ai_handle handle);
const char* ai_get_revision(ai_handle handle);
const char* ai_get_tool_version(ai_handle handle);
const char* ai_get_api_version(ai_handle handle);

uint32_t ai_get_node_size(ai_handle handle);
uint32_t ai_get_activations_size(ai_handle handle);
uint32_t ai_get_params_size(ai_handle handle);

ai_ptr ai_get_activations(ai_handle handle);
const ai_ptr ai_get_params(ai_handle handle);

//
// Quantization
//
const ai_quantization_info* ai_get_quantization(ai_tensor* tensor);

#endif  // TVM_RUNTIME_CRT_CONTRIB_STM32_AI_RUNTIME_API_H_
