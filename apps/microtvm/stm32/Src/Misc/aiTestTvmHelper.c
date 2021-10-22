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
 * \file aiTestTvmHelper.c
 * \brief STM32 TVM Helper functions for STM32 AI test application
 */

/*
 * Description:
 *
 *
 * History:
 *  - v1.0 - Initial version
 */

#include <stdio.h>
#include <string.h>

#include <aiTestTvmHelper.h>

// ==================================================
//   aiPlatformVersion
// ==================================================
void aiPlatformVersion (void)
{
  printf("\r\nAI platform (API %d.%d.%d - RUNTIME %d.%d.%d)\r\n",
      AI_PLATFORM_API_MAJOR,
      AI_PLATFORM_API_MINOR,
      AI_PLATFORM_API_MICRO,
      AI_PLATFORM_RUNTIME_MAJOR,
      AI_PLATFORM_RUNTIME_MINOR,
      AI_PLATFORM_RUNTIME_MICRO);
}

// ==================================================
//   aiLogErr
// ==================================================
void aiLogErr(const char *fct, const char * err)
{
  if (fct) {
    printf("E: AI error: %s - %s\r\n", fct, err);
  }
  else {
    printf("E: AI error - %s\r\n", err);
  }
}

// ==================================================
//   aiPrintLayoutBuffer
// ==================================================

const char* _tvm_to_str(DLDataType dtype)
{
    if (dtype.code == kDLBfloat)
        return "kDLBfloat";
    if (dtype.code == kDLFloat)
        return "kDLFloat";
    else if (dtype.code == kDLUInt)
        return "kDLUInt";
    else if (dtype.code == kDLInt)
        return "kDLInt";
    else
        return "??";
}


void aiPrintLayoutBuffer (const char *msg, int idx, ai_tensor *tensor)
{
  DLTensor * dltensor = &tensor->dltensor;
  DLDataType dtype = dltensor->dtype;

  printf("%s[%d] ", msg, idx);

  /* Data type/format */
  printf(" %s/%ubits(%u)", _tvm_to_str(dtype), dtype.bits, dtype.lanes);

  /* meta data */
  if (tensor->quant != NULL) {
    const ai_quantization_info* info = tensor->quant;
    printf(", scale=%f zp=%d (%d)", *info->scale, *(int *)info->zero_point, (int)info->dim);
  }

  /* shape/size and @ */
  int32_t size = get_tensor_size (tensor);
  printf(", %ld bytes, shape=(", size);
  for (int i = 0; i < dltensor->ndim; ++i) {
    if (i != 0)
      printf(", ");
    printf("%d", (int)dltensor->shape[i]);
  }
  printf("), (0x%08x)\r\n", (unsigned int)dltensor->data);
}

// ==================================================
//   aiPrintNetworkInfo
// ==================================================
void aiPrintNetworkInfo(ai_model_info *nn, ai_handle hdl)
{
  const char *name = nn->name;
  const char *datetime = nn->datetime;
  const char *revision = nn->revision;
  const char *tool_version = nn->tool_version;

  uint32_t n_nodes = nn->n_nodes;
  uint32_t n_inputs = nn->n_inputs;
  uint32_t n_outputs = nn->n_outputs;

  uint32_t act_size = nn->activations_size;
  uint32_t params_size = nn->params_size;

  ai_ptr params_addr = 0;
  ai_ptr act_addr = 0;
  
  if (hdl) {
    params_addr = ai_get_params(hdl);
    act_addr = ai_get_activations(hdl);
  }

  printf("\r\nNetwork configuration\r\n");
  printf(" Model name          : %s\r\n", name);
  printf(" Compile datetime    : %s\r\n", datetime);
  printf(" Tool version       : %s (%s)\r\n", tool_version, revision);
  
  printf("\r\nNetwork info\r\n");
  printf("  nodes              : %ld\r\n", n_nodes);
  printf("  activation         : %ld bytes", act_size);
  if (act_addr)
    printf(" (0x%08x)\r\n", (int)act_addr);
  else
    printf(" - not initialized\r\n");
  printf("  params             : %ld bytes", params_size);
  if (params_addr)
    printf(" (0x%08x)\r\n", (int)params_addr);
  else
    printf(" - not initialized\r\n");
  printf("  inputs/outputs     : %lu/%lu\r\n", n_inputs, n_outputs);

  if (hdl == NULL)
    return;

  for (int i = 0; i < n_inputs; i++) {
    ai_tensor *input = ai_get_input(hdl, i);
    aiPrintLayoutBuffer("   I", i, input);
  }

  for (int i = 0; i < n_outputs; i++) {
    ai_tensor *output = ai_get_output(hdl, i);
    aiPrintLayoutBuffer("   O", i, output);
  }
}


#if defined(NO_X_CUBE_AI_RUNTIME) && NO_X_CUBE_AI_RUNTIME == 1

#include "aiPbMgr.h"
#include <stdlib.h>

static ai_buffer_format set_ai_buffer_format(DLDataType dtype)
{
  if ((dtype.code == kDLFloat) && (dtype.bits == 32))
    return AI_BUFFER_FORMAT_FLOAT;
  if ((dtype.code == kDLUInt) && (dtype.bits == 8))
    return AI_BUFFER_FORMAT_U8;
  if ((dtype.code == kDLInt) && (dtype.bits == 8))
    return AI_BUFFER_FORMAT_S8;
  return AI_BUFFER_FORMAT_NONE;
}

static void set_ai_buffer(ai_tensor *tensor, ai_buffer* to_)
{
  DLTensor * dltensor = &tensor->dltensor;
  DLDataType dtype = dltensor->dtype;

  struct ai_buffer_ext *eto_ = (struct ai_buffer_ext *)to_;

  to_->n_batches = dltensor->shape[0];

  if (dltensor->ndim == 2) {
      to_->height = 1;
      to_->width = 1;
      to_->channels = dltensor->shape[1];
  }
  else if (dltensor->ndim == 3) {
      to_->height = dltensor->shape[1];
      to_->width = 1;
      to_->channels = dltensor->shape[2];
  } 
  else if (dltensor->ndim == 4) {
      to_->height = dltensor->shape[1];
      to_->width = dltensor->shape[2];
      to_->channels = dltensor->shape[3];
  }
 
  to_->data = (ai_handle)dltensor->data;
  to_->format = set_ai_buffer_format(dtype);
  to_->meta_info = NULL;

  if (tensor->quant != NULL) {
    const ai_quantization_info* info = tensor->quant;
    eto_->buffer.meta_info = (ai_buffer_meta_info *)&eto_->extra;
    eto_->extra.scale = *info->scale;
    eto_->extra.zero_point = *info->zero_point;
  }

  return;
}

void aiTvmToReport(ai_model_info *nn, ai_handle hdl, ai_network_report *report)
{
  const char *name = nn->name;
  const char *datetime = nn->datetime;
  const char *revision = nn->revision;
  //const char *tool_version = nn->tool_version;

  uint32_t n_nodes = nn->n_nodes;
  uint32_t n_inputs = nn->n_inputs;
  uint32_t n_outputs = nn->n_outputs;

  uint32_t act_size = nn->activations_size;
  uint32_t params_size = nn->params_size;

  size_t size;

  const char *_null = "NULL";

  ai_platform_version _version_api = { AI_PLATFORM_API_MAJOR,
                                       AI_PLATFORM_API_MINOR,
                                       AI_PLATFORM_API_MICRO, 0 };
  ai_platform_version _version_rt = { AI_PLATFORM_RUNTIME_MAJOR,
                                      AI_PLATFORM_RUNTIME_MINOR,
                                      AI_PLATFORM_RUNTIME_MICRO, 0};
  ai_platform_version _version_null = {0, 0, 0, 0 };

  const ai_buffer _def_buffer = { AI_BUFFER_FORMAT_U8, 1, 1, 2, 1, NULL, NULL };

  report->model_name = name;
  report->model_signature = _null;
  report->model_datetime = datetime;
  report->compile_datetime = __DATE__ " " __TIME__;
  report->runtime_revision = revision;
  report->tool_revision = "TVM";

  report->runtime_version = _version_rt;
  report->tool_version = _version_api;

  report->tool_api_version = _version_null;
  report->api_version = _version_null;
  report->interface_api_version = _version_null;

  report->n_macc = 1;

  report->n_inputs = n_inputs;
  size = report->n_inputs * sizeof(struct ai_buffer_ext);
  report->inputs = (ai_buffer *)malloc(size);
  for (int idx=0; idx<report->n_inputs; idx++) {
    ai_tensor *input = ai_get_input(hdl, idx);
    set_ai_buffer(input, &report->inputs[idx]);
  }

  report->n_outputs = n_outputs;
  size = report->n_outputs * sizeof(struct ai_buffer_ext);
  report->outputs = (ai_buffer *)malloc(size);
  for (int idx=0; idx<report->n_outputs; idx++) {
    ai_tensor *output = ai_get_output(hdl, idx);
    set_ai_buffer(output, &report->outputs[idx]);
  }

  report->n_nodes = n_nodes;
  report->signature = (ai_signature)0;

  report->activations = _def_buffer;
  report->activations.channels = (int)act_size;
  report->params = _def_buffer;
  report->params.channels = (int)params_size;
}

#endif
