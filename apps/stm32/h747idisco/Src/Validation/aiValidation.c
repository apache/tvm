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
 * \file aiValidation.c
 * \brief AI Validation application (entry points)
 *
 */

 /* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

/* Specific AI test application  */
#include <aiValidation.h>

#include <aiTestUtility.h>
#include <aiTestTvmHelper.h>
#include <aiPbMgr.h>

/* AI header files */
#include "ai_runtime_api.h"


/* -----------------------------------------------------------------------------
 * TEST-related definitions
 * -----------------------------------------------------------------------------
 */

#define _APP_VERSION_MAJOR_     (0x01)
#define _APP_VERSION_MINOR_     (0x00)
#define _APP_VERSION_           ((_APP_VERSION_MAJOR_ << 8) | _APP_VERSION_MINOR_)
#define _APP_NAME_              "STM32 - TVM validation application"


#define _MAX_AI_MNETWORK_NUMBER (4)
#define _MAX_AI_INPUTS_COUNT    (2)
// For now TVM generator only supports single output models.
#define _MAX_AI_OUTPUTS_COUNT   (1)

struct network_exec_ctx {
  ai_handle hdl;
  ai_ptr activations;
  ai_ptr inputs[_MAX_AI_INPUTS_COUNT];
  ai_ptr outputs[_MAX_AI_OUTPUTS_COUNT];
  ai_network_report report;
} net_exec_ctx[_MAX_AI_MNETWORK_NUMBER];


/* -----------------------------------------------------------------------------
 * AI-related functions
 * -----------------------------------------------------------------------------
 */

static struct network_exec_ctx *aiGetCtxFromName(const char *nn_name)
{
  struct network_exec_ctx *cur = NULL;

  if (!nn_name)
    return NULL;

  for (int idx=0; idx < _MAX_AI_MNETWORK_NUMBER; idx++) {
    cur = &net_exec_ctx[idx];
    if (cur->hdl &&
        (strlen(cur->report.model_name) == strlen(nn_name)) &&
        (strncmp(cur->report.model_name, nn_name,
            strlen(cur->report.model_name)) == 0)) {
      break;
    }
    cur = NULL;
  }
  return cur;
}

static void aiDone(struct network_exec_ctx *ctx)
{
  printf("Releasing the instance...\r\n");

  if (ctx->hdl != AI_HANDLE_NULL) {
    //
    // Free the activations if the model does not include built-in
    // activation storage.
    //
    if (ctx->activations != NULL) {
      free(ctx->activations);
    }
    free(ctx->report.inputs);
    free(ctx->report.outputs);
    if (ai_destroy(ctx->hdl) != AI_STATUS_OK) {
	    const char * err = ai_get_error(ctx->hdl);
	    aiLogErr("ai_destroy", err);
    }
    //
    // Free the input/outputs buffers if the model does not include built-in
    // input/output storage.
    //
    for (int i=0; i<_MAX_AI_INPUTS_COUNT;i++)
      if (ctx->inputs[i] != NULL)
        free(ctx->inputs[i]);
    for (int i=0; i<_MAX_AI_OUTPUTS_COUNT;i++)
      if (ctx->outputs[i] != NULL)
        free(ctx->outputs[i]);
    memset(ctx, 0, sizeof(struct network_exec_ctx));
  }
}

static int aiBootstrap(ai_model_info* nn, struct network_exec_ctx *ctx)
{
  const char * nn_name = AI_MODEL_name(nn);

  ai_status err = AI_STATUS_OK;

  /* Creating the instance */
  printf("Creating the instance for the model \"%s\"..\r\n", nn_name);

  //
  // Allocate the activations if the model does not include built-in
  // activation storage.
  //
  if (AI_MODEL_activations(nn) != NULL) {
    ctx->activations = NULL;
    err = ai_create(nn, AI_MODEL_activations(nn), &ctx->hdl);
  }
  else {
    ctx->activations = (ai_ptr)malloc(AI_MODEL_activations_size(nn));
    err = ai_create(nn, ctx->activations, &ctx->hdl);
  }

  if (err != AI_STATUS_OK) {
    const char * msg = ai_get_error(ctx->hdl);
    aiLogErr("ai_create", msg);
    return -1;
  }

  uint16_t n_inputs = ai_get_input_size(ctx->hdl);
  uint16_t n_outputs = ai_get_output_size(ctx->hdl);

  if ((n_inputs > _MAX_AI_INPUTS_COUNT) || (n_outputs > _MAX_AI_OUTPUTS_COUNT))
  {
    printf("E: Exceed supported number of inputs/outputs..\r\n");
    aiDone(ctx);
    return -1;
  }

  //
  // Allocate the input/outputs buffers if necessary
  //
  for (int i = 0; i < n_inputs; i++) {
    ai_tensor *input_tensor = ai_get_input(ctx->hdl, i);
    DLTensor *tensor = get_dltensor (input_tensor);
    if (tensor->data == NULL) {
      uint32_t bytes = get_tensor_size (input_tensor);
      ctx->inputs[i] = (ai_ptr)malloc(bytes);
      tensor->data = ctx->inputs[i];
    }
  }

  for (int i = 0; i < n_outputs; i++) {
    ai_tensor *output_tensor = ai_get_output(ctx->hdl, i);
    DLTensor *tensor = get_dltensor (output_tensor);
    if (tensor->data == NULL) {
      uint32_t bytes = get_tensor_size (output_tensor);
      ctx->outputs[i] = (ai_ptr)malloc(bytes);
      tensor->data = ctx->outputs[i];
    }
  }


  /* Display network instance info */
  aiPrintNetworkInfo(nn, ctx->hdl);

  aiTvmToReport(nn, ctx->hdl, &ctx->report);

  return 0;
}

static void aiDeInit(void)
{
  for (int idx=0; idx < _MAX_AI_MNETWORK_NUMBER; idx++) {
    aiDone(&net_exec_ctx[idx]);
  }
}

static int aiInit(void)
{
  int res;
  int idx;

  aiPlatformVersion();

  /* clean APP network instance context */
  for (idx=0; idx < _MAX_AI_MNETWORK_NUMBER; idx++) {
    memset(&net_exec_ctx[idx], 0, sizeof(struct network_exec_ctx));
  }

  /* discover and create the instances */
  idx = 0;
  for (ai_model_iterator it = ai_model_iterator_begin();
      it != ai_model_iterator_end();
      it = ai_model_iterator_next(it)) {

    ai_model_info *nn = ai_model_iterator_value(it);

    printf("\r\nFound the model \"%s\" (idx=%d)\r\n", nn->name, idx);

    res = aiBootstrap(nn, &net_exec_ctx[idx]);
    if (res) {
      aiDeInit();
      return res;
    }
    idx++;
  }

  return 0;
}


/* -----------------------------------------------------------------------------
 * Specific test APP commands
 * -----------------------------------------------------------------------------
 */

 void aiPbCmdNNInfo(const reqMsg *req, respMsg *resp, void *param)
{
  UNUSED(param);

  struct network_exec_ctx *ctx = NULL;
  if ((req->param >= 0) && (req->param <_MAX_AI_MNETWORK_NUMBER))
    ctx = &net_exec_ctx[req->param];

  if (ctx && ctx->hdl)
    aiPbMgrSendNNInfo(req, resp, EnumState_S_IDLE,
        &ctx->report);
  else
    aiPbMgrSendAck(req, resp, EnumState_S_ERROR,
        EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
}

void aiPbCmdNNRun(const reqMsg *req, respMsg *resp, void *param)
{
  ai_status tvm_status = AI_STATUS_OK;
  uint32_t tend;
  bool res;
  UNUSED(param);

  /* 0 - Check if requested c-name model is available -------------- */
  struct network_exec_ctx *ctx = aiGetCtxFromName(req->name);

  if (!ctx) {
    aiPbMgrSendAck(req, resp, EnumState_S_ERROR,
        EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
    return;
  }

  /* 1 - Send a ACK (ready to receive a tensor) -------------------- */
  aiPbMgrSendAck(req, resp, EnumState_S_WAITING,
      aiPbAiBufferSize(&ctx->report.inputs[0]), EnumError_E_NONE);

  /* 2 - Receive all input tensors --------------------------------- */
  for (int i = 0; i < ctx->report.n_inputs; i++) {
    /* upload a buffer */
    EnumState state = EnumState_S_WAITING;
    if ((i + 1) == ctx->report.n_inputs)
      state = EnumState_S_PROCESSING;
    res = aiPbMgrReceiveAiBuffer3(req, resp, state, &ctx->report.inputs[i]);
    if (res != true)
      return;
  }

  /* 3 - Processing ------------------------------------------------ */

  cyclesCounterStart();
  tvm_status = ai_run(ctx->hdl);
  tend = cyclesCounterEnd();

  if (tvm_status != AI_STATUS_OK) {
    printf("E: TVM ai_run() fails\r\n");
    aiPbMgrSendAck(req, resp, EnumState_S_ERROR,
        EnumError_E_GENERIC, EnumError_E_GENERIC);
    return;
  }

  /* 4 - Send all output tensors ----------------------------------- */
  for (int i = 0; i < ctx->report.n_outputs; i++) {
    EnumState state = EnumState_S_PROCESSING;
    if ((i + 1) == ctx->report.n_outputs)
      state = EnumState_S_DONE;
    aiPbMgrSendAiBuffer4(req, resp, state,
        EnumLayerType_LAYER_TYPE_OUTPUT << 16 | 0,
        0, dwtCyclesToFloatMs(tend),
        &ctx->report.outputs[i], 0.0f, 0);
  }
}


static aiPbCmdFunc pbCmdFuncTab[] = {
    AI_PB_CMD_SYNC(NULL),
    AI_PB_CMD_SYS_INFO(NULL),
    { EnumCmd_CMD_NETWORK_INFO, &aiPbCmdNNInfo, NULL },
    { EnumCmd_CMD_NETWORK_RUN, &aiPbCmdNNRun, NULL },
#if defined(AI_PB_TEST) && AI_PB_TEST == 1
    AI_PB_CMD_TEST(NULL),
#endif
    AI_PB_CMD_END,
};


/* -----------------------------------------------------------------------------
 * Exported/Public functions
 * -----------------------------------------------------------------------------
 */

int aiValidationInit(void)
{
  printf("\r\n#\r\n");
  printf("# %s %d.%d\r\n", _APP_NAME_ , _APP_VERSION_MAJOR_, _APP_VERSION_MINOR_);
  printf("#\r\n");

  systemSettingLog();
  cyclesCounterInit();

  return 0;
}

int aiValidationProcess(void)
{
  int r;

  r = aiInit();
  if (r) {
    printf("\r\nE:  aiInit() r=%d\r\n", r);
    HAL_Delay(2000);
    return r;
  } else {
    printf("\r\n");
    printf("-------------------------------------------\r\n");
    printf("| READY to receive a CMD from the HOST... |\r\n");
    printf("-------------------------------------------\r\n");
    printf("\r\n");
    printf("# Note: At this point, default ASCII-base terminal should be closed\r\n");
    printf("# Stm32com-base interface should be used\r\n");
    printf("# (i.e. Python stm32com module). Protocol version = %d.%d\r\n",
        EnumVersion_P_VERSION_MAJOR,
        EnumVersion_P_VERSION_MINOR);
  }

  aiPbMgrInit(pbCmdFuncTab);

  do {
    r = aiPbMgrWaitAndProcess();
  } while (r==0);

  return r;
}

void aiValidationDeInit(void)
{
  printf("\r\n");
  aiDeInit();
  printf("bye bye ...\r\n");
}

