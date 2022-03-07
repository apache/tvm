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
 * \file ai_runtime_api.c
 * \brief The runtime API for the TVM generated C code.
 */

// LINT_C_FILE

#include "ai_runtime_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =======================================================
//                    ai_network_t
// =======================================================

typedef struct {
  ai_model_info* info;
  ai_tensor** inputs;
  ai_tensor** outputs;
  ai_ptr activations;
  const char* error;
} ai_network_t;

//
// .nn_models_info section
//
extern uintptr_t __models_section_start__;
extern uintptr_t __models_section_end__;

uint32_t _modelsSection_start = (uint32_t)(&__models_section_start__);
uint32_t _modelsSection_end = (uint32_t)(&__models_section_end__);

// =======================================================
//                       Iterator
// =======================================================
ai_model_iterator ai_model_iterator_begin() {
  return _modelsSection_start;  // begin()
}

ai_model_iterator ai_model_iterator_end() { return _modelsSection_end; }

ai_model_iterator ai_model_iterator_next(ai_model_iterator idx) {
  return (idx + sizeof(ai_model_info));
}

ai_model_info* ai_model_iterator_value(ai_model_iterator idx) { return (ai_model_info*)idx; }

// =======================================================
//   ai_create
// =======================================================
AI_API_ENTRY ai_status ai_create(ai_model_info* nn, ai_ptr activations, ai_handle* handle) {
  uint32_t n_inputs = AI_MODEL_n_inputs(nn);
  uint32_t n_outputs = AI_MODEL_n_outputs(nn);

  ai_status status = AI_STATUS_OK;

  //
  // Create internal network representation
  //
  ai_network_t* network = (ai_network_t*)malloc(sizeof(ai_network_t));

  network->info = nn;

  for (int i = 0; i < n_inputs; i++) {
    network->inputs = AI_MODEL_inputs(nn);
  }
  for (int i = 0; i < n_outputs; i++) {
    network->outputs = AI_MODEL_outputs(nn);
  }

  network->activations = activations;

  network->error = NULL;

  const ai_ptr params = nn->ai_get_params();
  status = nn->ai_create(params, activations);
  if (status != AI_STATUS_OK) {
    network->error = TVMGetLastError();
  }

  //
  // Setup weights and activations
  //
  *handle = network;

  return status;
}

// =======================================================
//   ai_destroy
// =======================================================
AI_API_ENTRY ai_status ai_destroy(ai_handle handle) {
  if (handle == NULL) {
    return AI_STATUS_ERROR;
  }

  ai_network_t* network = (ai_network_t*)handle;

  free(network);

  return AI_STATUS_OK;
}

// =======================================================
//   ai_get_error
// =======================================================
AI_API_ENTRY
const char* ai_get_error(ai_handle handle) {
  if (handle == NULL) {
    return "Network handle is NULL";
  }
  ai_network_t* network = (ai_network_t*)handle;
  if (network->error == NULL) {
    return "";
  }
  return network->error;
}

// =======================================================
//   ai_get_input_size
// =======================================================
AI_API_ENTRY int32_t ai_get_input_size(ai_handle handle) {
  if (handle == NULL) {
    return 0;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_n_inputs(network->info);
}

// =======================================================
//   ai_get_output_size
// =======================================================
AI_API_ENTRY int32_t ai_get_output_size(ai_handle handle) {
  if (handle == NULL) {
    return 0;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_n_outputs(network->info);
}

// =======================================================
//   ai_get_input
// =======================================================
AI_API_ENTRY ai_tensor* ai_get_input(ai_handle handle, int32_t index) {
  if (handle == NULL) {
    return NULL;
  }
  ai_network_t* network = (ai_network_t*)handle;
  if (index >= AI_MODEL_n_inputs(network->info)) {
    network->error = "Input index out of range";
    return NULL;
  }
  return (network->inputs)[index];
}

// =======================================================
//   ai_get_output
// =======================================================
AI_API_ENTRY ai_tensor* ai_get_output(ai_handle handle, int32_t index) {
  if (handle == NULL) {
    return NULL;
  }
  ai_network_t* network = (ai_network_t*)handle;
  if (index >= AI_MODEL_n_outputs(network->info)) {
    network->error = "Output index out of range";
    return NULL;
  }
  return (network->outputs)[index];
}

// =======================================================
//   ai_run
// =======================================================
AI_API_ENTRY ai_status ai_run(ai_handle handle) {
  if (handle == NULL) {
    return AI_STATUS_ERROR;
  }
  ai_network_t* network = (ai_network_t*)handle;

  ai_model_info* nn = network->info;

  uint32_t n_inputs = AI_MODEL_n_inputs(nn);
  uint32_t n_outputs = AI_MODEL_n_outputs(nn);
  ai_status status = AI_STATUS_OK;

  //
  // Check that input tensors have been specified
  //
  uint32_t i;
  for (i = 0; i < n_inputs; i++) {
    ai_tensor* input_tensor = network->inputs[i];
    DLTensor* input = &input_tensor->dltensor;
    if (input->data == NULL) {
      network->error = "Network input NULL";
      return AI_STATUS_ERROR;
    }
  }
  for (i = 0; i < n_outputs; i++) {
    ai_tensor* output_tensor = network->outputs[i];
    DLTensor* output = &output_tensor->dltensor;
    if (output->data == NULL) {
      network->error = "Network output NULL";
      return AI_STATUS_ERROR;
    }
  }

  status = nn->ai_run(network->inputs, network->outputs);

  if (status != AI_STATUS_OK) {
    const char* err = TVMGetLastError();
    network->error = err;
  }

  return status;
}

// =======================================================
//   ai_get_name
// =======================================================
const char* ai_get_name(ai_handle handle) {
  if (handle == NULL) {
    return NULL;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_name(network->info);
}

// =======================================================
//   ai_get_datetime
// =======================================================
const char* ai_get_datetime(ai_handle handle) {
  if (handle == NULL) {
    return NULL;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_datetime(network->info);
}

// =======================================================
//   ai_get_revision
// =======================================================
const char* ai_get_revision(ai_handle handle) {
  if (handle == NULL) {
    return NULL;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_revision(network->info);
}

// =======================================================
//   ai_get_tool_version
// =======================================================
const char* ai_get_tool_version(ai_handle handle) {
  if (handle == NULL) {
    return NULL;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_tool_version(network->info);
}

// =======================================================
//   ai_get_api_version
// =======================================================
const char* ai_get_api_version(ai_handle handle) {
  if (handle == NULL) {
    return NULL;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_api_version(network->info);
}

// =======================================================
//   ai_get_node_size
// =======================================================
uint32_t ai_get_node_size(ai_handle handle) {
  if (handle == NULL) {
    return 0;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_n_nodes(network->info);
}

// =======================================================
//   ai_get_activations_size
// =======================================================
uint32_t ai_get_activations_size(ai_handle handle) {
  if (handle == NULL) {
    return 0;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_activations_size(network->info);
}

// =======================================================
//   ai_get_params_size
// =======================================================
uint32_t ai_get_params_size(ai_handle handle) {
  if (handle == NULL) {
    return 0;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return AI_MODEL_params_size(network->info);
}

// =======================================================
//   ai_get_activations
// =======================================================
ai_ptr ai_get_activations(ai_handle handle) {
  if (handle == NULL) {
    return 0;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return network->activations;
}

// =======================================================
//   ai_get_params
// =======================================================
const ai_ptr ai_get_params(ai_handle handle) {
  if (handle == NULL) {
    return NULL;
  }
  ai_network_t* network = (ai_network_t*)handle;
  return network->info->ai_get_params();
}

// =======================================================
//   ai_get_quantization
// =======================================================
const ai_quantization_info* ai_get_quantization(ai_tensor* tensor) {
  if (tensor == NULL) {
    return NULL;
  }
  return tensor->quant;
}
