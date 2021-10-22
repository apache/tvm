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
 * \file aiSystemPerformance.c
 * \brief Entry points for AI system performance application.
 *        Simple STM32 application to measure and report the system
 *        performance of a generated model.
 */
 
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

/* Specific AI test application  */
#include <aiSystemPerformance.h>

#include <aiTestUtility.h>
#include <aiTestTvmHelper.h>

/* AI header files */
#include "ai_runtime_api.h"

//
// MIN_STACK_SIZE
//
extern uint32_t _estack;
extern uint32_t _sstack;

uint32_t estack_addr = (uint32_t) (&_estack);
uint32_t sstack_addr = (uint32_t) (&_sstack);

/* -----------------------------------------------------------------------------
 * TEST-related definitions
 * -----------------------------------------------------------------------------
 */

#define _APP_VERSION_MAJOR_     (0x01)
#define _APP_VERSION_MINOR_     (0x00)
#define _APP_VERSION_           ((_APP_VERSION_MAJOR_ << 8) | _APP_VERSION_MINOR_)
#define _APP_NAME_              "STM32 - TVM System performance application"

#define _APP_ITER_              10  /* number of iteration for perf. test */

static bool profiling_mode = false;
static int  profiling_factor = 5;

#define _MAX_AI_MNETWORK_NUMBER (4)
#define _MAX_AI_INPUTS_COUNT    (2)
// For now TVM generator only supports single output models.
#define _MAX_AI_OUTPUTS_COUNT   (1)

struct network_exec_ctx {
  ai_handle hdl;
  ai_ptr activations;
  ai_ptr inputs[_MAX_AI_INPUTS_COUNT];
  ai_ptr outputs[_MAX_AI_OUTPUTS_COUNT];
} net_exec_ctx[_MAX_AI_MNETWORK_NUMBER];


/* -----------------------------------------------------------------------------
 * AI-related functions
 * -----------------------------------------------------------------------------
 */

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
    if (ai_destroy(ctx->hdl) != AI_STATUS_OK) {
      const char * err = ai_get_error(ctx->hdl);
      aiLogErr("ai_destroy", err);
    }
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

  /* Display network instance info */
  aiPrintNetworkInfo(nn, ctx->hdl);

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
 * Specific APP/test functions
 * -----------------------------------------------------------------------------
 */

static int aiTestPerformance(int idx)
{
  int iter;
  int niter;
  struct dwtTime t;
  uint64_t tcumul;
  uint64_t tend;
  uint64_t tmin;
  uint64_t tmax;

  ai_handle handle = net_exec_ctx[idx].hdl;
  ai_status status = AI_STATUS_OK;

  if (handle == AI_HANDLE_NULL) {
    printf("E: network handle is NULL (idx=%d)\r\n", idx);
    return -1;
  }

  MON_STACK_INIT(estack_addr-sstack_addr);

  if (profiling_mode)
    niter = _APP_ITER_ * profiling_factor;
  else
    niter = _APP_ITER_;

  printf("\r\nRunning PerfTest on \"%s\" with random inputs (%d iterations)...\r\n", ai_get_name(handle), niter);

  MON_STACK_CHECK0();

  /* reset/init cpu clock counters */
  tcumul = 0ULL;
  tmin = UINT64_MAX;
  tmax = 0UL;

  MON_STACK_MARK();

  uint16_t n_inputs = ai_get_input_size(handle);
  uint16_t n_outputs = ai_get_output_size(handle);

  //
  // Allocate input/output tensors
  //
  for (int i = 0; i < n_inputs; i++) {
    ai_tensor *input_tensor = ai_get_input(handle, i);
    DLTensor *tensor = get_dltensor (input_tensor);
    if (tensor->data == NULL) {
      uint32_t bytes = get_tensor_size (input_tensor);
      net_exec_ctx[idx].inputs[i] = (ai_ptr)malloc(bytes);
      tensor->data = net_exec_ctx[idx].inputs[i];
    }
  }

  for (int i = 0; i < n_outputs; i++) {
    ai_tensor *output_tensor = ai_get_output(handle, i);
    DLTensor * tensor = get_dltensor (output_tensor);
    if (tensor->data == NULL) {
      uint32_t bytes = get_tensor_size (output_tensor);
      net_exec_ctx[idx].outputs[i] = (ai_ptr)malloc(bytes);
      tensor->data = net_exec_ctx[idx].outputs[i];
    }
  }
  
  if (profiling_mode) {
    printf("Profiling mode (%d)...\r\n", profiling_factor);
    fflush(stdout);
  }

  MON_ALLOC_RESET();

  for (iter = 0; iter < niter; iter++) {

    //
    // Fill input vectors
    //

    for (int ii = 0; ii < n_inputs; ii++) {
      ai_tensor * input_tensor = ai_get_input(handle, ii);
      DLTensor * input = &input_tensor->dltensor;
      if (input == NULL) {
        printf("E: corrupted inputs ...\r\n");
        HAL_Delay(100);
        return -1;
      }

      DLDataType dtype = input->dtype;

      if (dtype.lanes > 1) {
        printf("E: vector inputs are not supported ...\r\n");
        HAL_Delay(100);
        return -1;
      }

      if (dtype.code == kDLBfloat) {
        printf("E: Double float inputs are not supported ...\r\n");
        HAL_Delay(100);
        return -1;
      }

      // Compute input tensor size - elements
      uint32_t size = get_tensor_elts (input_tensor);
      int8_t * in_data = (int8_t*)input->data;

      for (int i = 0; i < size; ++i) {
        //
        // uniform distribution between -1.0 and 1.0
        //
        const float v = 2.0f * (float) rand() / (float) RAND_MAX - 1.0f;
        if (dtype.code == kDLFloat) {
          *(float *)(in_data + i * 4) = v;
        }
        else {
          in_data[i] = (int8_t)(v * 127);
        }
      }
    } // n_inputs

    MON_ALLOC_ENABLE();

    cyclesCounterStart();
    status = ai_run(handle);

    if (status != AI_STATUS_OK) {
      const char * err = ai_get_error(handle);
      aiLogErr("ai_run", err);
      break;
    }
    tend = cyclesCounterEnd();

    MON_ALLOC_DISABLE();

#if 0
    //
    // Check outputs
    //
    for (int ii = 0; ii < n_outputs; ii++) {
      ai_tensor * output_tensor = ai_get_output(handle, ii);
      DLTensor * output = &output_tensor->dltensor;
      if (output == NULL) {
        printf("E: corrupted outputs ...\r\n");
        HAL_Delay(100);
        return -1;
      }

      DLDataType dtype = output->dtype;
      uint32_t size = get_tensor_elts (output_tensor);
      int8_t * out_data = (int8_t*)output->data;
      printf (" == output[%d]: [", ii);
      for (int i = 0; i < size; ++i) {
	if (dtype.code == kDLFloat) {
          printf ("%g ", *(float *)(out_data + i * 4));
        }
        else {
          printf ("%d ", out_data[i]);
        }
      }
      printf ("]\r\n");

    } // outputs
#endif // 9

    if (tend < tmin)
      tmin = tend;
    if (tend > tmax)
      tmax = tend;
    tcumul += tend;
    dwtCyclesToTime(tend, &t);

#if _APP_DEBUG_ == 1
    printf(" #%02d %8d.%03dms (%lu cycles)\r\n", iter, t.ms, t.us, tend);
#else

    if (!profiling_mode) {
      if (t.s > 10)
        niter = iter;
      printf(".");
      fflush(stdout);
    }
#endif
  }

#if _APP_DEBUG_ != 1
  printf("\r\n");
#endif

  //
  // Free input/output storage
  //
  for (int i = 0; i < n_inputs; i++) {
    if (net_exec_ctx[idx].inputs[i] != NULL) {
      ai_tensor *input_tensor = ai_get_input(handle, i);
      DLTensor *tensor = get_dltensor (input_tensor);
      free (net_exec_ctx[idx].inputs[i]);
      net_exec_ctx[idx].inputs[i] = NULL;
      tensor->data = NULL;
    }
  }
  for (int i = 0; i < n_outputs; i++) {
    if (net_exec_ctx[idx].outputs[i] != NULL) {
      ai_tensor *output_tensor = ai_get_output(handle, i);
      DLTensor *tensor = get_dltensor (output_tensor);
      free (net_exec_ctx[idx].outputs[i]);
      net_exec_ctx[idx].outputs[i] = NULL;
      tensor->data = NULL;
    }
  }

  MON_STACK_EVALUATE();

  printf("\r\n");

  tcumul /= (uint64_t)iter;
  dwtCyclesToTime(tcumul, &t);

  printf("Results for \"%s\", %d inference(s) @%ldMHz/%ldMHz\r\n",
      ai_get_name(handle), iter,
      HAL_RCC_GetSysClockFreq() / 1000000,
      HAL_RCC_GetHCLKFreq() / 1000000
  );

  printf(" duration     : %d.%03d ms (average)\r\n", t.s * 1000 + t.ms, t.us);
  printf(" CPU cycles   : %lu -%lu/+%lu (average,-/+)\r\n",
      (uint32_t)(tcumul), (uint32_t)(tcumul - tmin),
      (uint32_t)(tmax - tcumul));
  printf(" CPU Workload : %d%c\r\n", (int)((tcumul * 100) / t.fcpu), '%');

  MON_STACK_REPORT();
  MON_ALLOC_REPORT();

  return 0;
}

/* -----------------------------------------------------------------------------
 * Exported/Public functions
 * -----------------------------------------------------------------------------
 */

#define CONS_EVT_TIMEOUT    (0)
#define CONS_EVT_QUIT       (1)
#define CONS_EVT_RESTART    (2)
#define CONS_EVT_HELP       (3)
#define CONS_EVT_PAUSE      (4)
#define CONS_EVT_PROF       (5)
#define CONS_EVT_HIDE       (6)

#define CONS_EVT_UNDEFINED  (100)


static int aiTestConsole(void)
{
  uint8_t c = 0;

  if (ioRawGetUint8(&c, 5000) == -1) /* Timeout */
    return CONS_EVT_TIMEOUT;

  if ((c == 'q') || (c == 'Q'))
    return CONS_EVT_QUIT;

  if ((c == 'r') || (c == 'R'))
    return CONS_EVT_RESTART;

  if ((c == 'h') || (c == 'H') || (c == '?'))
    return CONS_EVT_HELP;

  if ((c == 'p') || (c == 'P'))
    return CONS_EVT_PAUSE;

  if ((c == 'x') || (c == 'X'))
    return CONS_EVT_PROF;

  return CONS_EVT_UNDEFINED;
}

int aiSystemPerformanceInit(void)
{
  printf("\r\n#\r\n");
  printf("# %s %d.%d\r\n", _APP_NAME_ , _APP_VERSION_MAJOR_, _APP_VERSION_MINOR_ );
  printf("#\r\n");

  systemSettingLog();
  cyclesCounterInit();
  aiInit();
  srand(3); /* deterministic outcome */
  dwtReset();
  return 0;
}

int aiSystemPerformanceProcess(void)
{
  int r;
  int idx = 0;

  do {
    r = aiTestPerformance(idx);
    idx = (idx+1) % _MAX_AI_MNETWORK_NUMBER;
    if (!net_exec_ctx[idx].hdl)
      idx = 0;

    if (!r) {
      r = aiTestConsole();
      if (r == CONS_EVT_UNDEFINED) {
        r = 0;
      }
      else if (r == CONS_EVT_HELP) {
        printf("\r\n");
        printf("Possible key for the interactive console:\r\n");
        printf("  [q,Q]      quit the application\r\n");
        printf("  [r,R]      re-start (NN de-init and re-init)\r\n");
        printf("  [p,P]      pause\r\n");
        printf("  [h,H,?]    this information\r\n");
        printf("   xx        continue immediately\r\n");
        printf("\r\n");
        printf("Press any key to continue..\r\n");

        while ((r = aiTestConsole()) == CONS_EVT_TIMEOUT) {
          HAL_Delay(1000);
        }

        if (r == CONS_EVT_UNDEFINED)
          r = 0;
      }

      if (r == CONS_EVT_PROF) {
        profiling_mode = true;
        profiling_factor *= 2;
        r = 0;
      }

      if (r == CONS_EVT_RESTART) {
        profiling_mode = false;
        profiling_factor = 5;
        printf("\r\n");
        aiDeInit();
        aiSystemPerformanceInit();
        r = 0;
      }

      if (r == CONS_EVT_QUIT) {
        profiling_mode = false;
        printf("\r\n");
        disableInts();
        aiDeInit();
        printf("\r\n");
        printf("Board should be reseted...\r\n");
        while (1) {
          HAL_Delay(1000);
        }
      }

      if (r == CONS_EVT_PAUSE) {
        printf("\r\n");
        printf("Press any key to continue..\r\n");
        while ((r = aiTestConsole()) == CONS_EVT_TIMEOUT) {
          HAL_Delay(1000);
        }
        r = 0;
      }
    }
  } while (r==0);

  return r;
}

void aiValidationDeInit(void)
{
  printf("\r\n");
  aiDeInit();
  printf("bye bye ...\r\n");
}

