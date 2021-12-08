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

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ai_runtime_api.h"
#include "network.h"
#include "network_data.h"

//
// Network that we are testing
//
extern ai_model_info network_network;

//
// Dummy: for the runtime
//
uint32_t __models_section_start__ = (uint32_t)&network_network;
uint32_t __models_section_end__ = (uint32_t)&network_network + sizeof(ai_model_info);

static ai_model_info* _model_p = &network_network;

//
// Global handle to reference the instantiated NN
//
static ai_handle _network = AI_HANDLE_NULL;

static uint8_t LoadInputImg(const char* filename, ai_tensor* input);
static int32_t quantize_val(float val, ai_quantization_info* quant);
static float dequantize_val(int32_t val, ai_quantization_info* quant);

// =================================================================
//   Convert_Fixed_To_Float
// =================================================================
static float Convert_Fixed_To_Float(uint8_t data, int8_t fl) {
  uint8_t val = data;
  float x;
  if (fl >= 0) {
    x = ((float)val) / (float)(1 << fl);  // NOLINT
  } else {
    x = ((float)val) / (1 / (float)(1 << fl));  // NOLINT
  }
  return x;
}

// =======================================================
//    error
// =======================================================
static void error(const char* fmt, ...) {
  va_list vp;
  char emsg[512];
  int32_t loc = 0;

  //
  // Prepare main error message:
  //
  va_start(vp, fmt);
  loc += vsprintf(&emsg[loc], fmt, vp);
  va_end(vp);

  // fputs (emsg, stderr);
  // fflush (stderr);

  fprintf(stderr, " #### Error: %s.\n", emsg);

  exit(-1);
}

// ==================================================
//   aiLogErr
// ==================================================
static void aiLogErr(const char* fct, const char* msg) {
  if (fct) {
    printf("E: AI error: %s - %s\r\n", fct, msg);
  } else {
    printf("E: AI error - %s\r\n", msg);
  }
}

// ==================================================
//   aiPrintLayoutBuffer
// ==================================================
static void aiPrintLayoutBuffer(const char* msg, int idx, ai_tensor* tensor) {
  DLTensor* dltensor = get_dltensor(tensor);
  DLDataType dtype = dltensor->dtype;

  printf("%s[%d] ", msg, idx);
  printf(" (%u, %u, %u)", dtype.code, dtype.bits, dtype.lanes);
  //
  // Quantization info exists for input/output tensors
  //
  const ai_quantization_info* quant = ai_get_quantization(tensor);
  if (quant != NULL) {
    printf(" -- TODO: quantization info \n");
  }

  int32_t size = get_tensor_size(tensor);
  printf(" %d bytes, shape=(", size);
  for (int i = 0; i < dltensor->ndim; ++i) {
    printf("%d,", (int32_t)dltensor->shape[i]);
  }
  printf("), address = 0x%08x\r\n", (unsigned int)dltensor->data);
}

// ==================================================
//   aiPrintNetworkInfo
// ==================================================
static void aiPrintNetworkInfo(ai_handle network) {
  const char* name = ai_get_name(network);
  const char* datetime = ai_get_datetime(network);
  const char* revision = ai_get_revision(network);
  const char* tool_version = ai_get_tool_version(network);
  const char* api_version = ai_get_api_version(network);

  uint32_t n_nodes = ai_get_node_size(network);
  uint32_t n_inputs = ai_get_input_size(network);
  uint32_t n_outputs = ai_get_output_size(network);

  uint32_t activations_size = ai_get_activations_size(network);
  uint32_t params_size = ai_get_params_size(network);

  printf("Network configuration...\r\n");
  printf(" Model name         : %s\r\n", name);
  printf(" Compile datetime   : %s\r\n", datetime);
  printf(" Tool revision      : %s (%s)\r\n", revision, tool_version);
  printf(" API version        : %s\r\n", api_version);
  printf("Network info...\r\n");
  printf("  nodes             : %d\r\n", n_nodes);
  printf("  activation        : %d bytes\r\n", activations_size);
  printf("  params            : %d bytes\r\n", params_size);
  printf("  inputs/outputs    : %u/%u\r\n", n_inputs, n_outputs);
}

// ======================================================
//   aiInit
// ======================================================
static int aiInit(void) {
  ai_status err = AI_STATUS_OK;

  const char* nn_name = AI_MODEL_name(_model_p);
  ai_ptr built_in_activations = AI_MODEL_activations(_model_p);

  //
  // Creating the network
  //
  printf("Creating the network \"%s\"..\r\n", nn_name);

  err = ai_create(_model_p, built_in_activations, &_network);
  if (err != AI_STATUS_OK) {
    const char* msg = ai_get_error(_network);
    aiLogErr("ai_create", msg);
    return -1;
  }

  //
  // Query the created network to get relevant info from it
  //
  aiPrintNetworkInfo(_network);

  uint32_t n_inputs = ai_get_input_size(_network);
  uint32_t n_outputs = ai_get_output_size(_network);
  uint32_t activations_size = ai_get_activations_size(_network);
  uint32_t params_size = ai_get_params_size(_network);

  const ai_ptr params = ai_get_params(_network);
  ai_ptr activations = ai_get_activations(_network);

  printf("Weights buffer     : 0x%08x %d bytes)\r\n", (unsigned int)params,
         (unsigned int)params_size);
  printf("Activation buffer  : 0x%08x (%d bytes) %s\r\n", (unsigned int)activations,
         (unsigned int)activations_size,
         ((uint32_t)activations & (uint32_t)0xFF000000) ? "internal" : "external");

  printf("Inputs:\r\n");
  for (int i = 0; i < n_inputs; i++) {
    ai_tensor* input = ai_get_input(_network, i);
    aiPrintLayoutBuffer("   I", i, input);
  }

  printf("Outputs:\r\n");
  for (int i = 0; i < n_outputs; i++) {
    ai_tensor* output = ai_get_output(_network, i);
    aiPrintLayoutBuffer("   O", i, output);
  }

  return 0;
}

// ======================================================
//   aiDeInit
// ======================================================
static void aiDeInit(void) {
  ai_status err = AI_STATUS_OK;

  printf("Releasing the network(s)...\r\n");

  if (ai_destroy(_network) != AI_STATUS_OK) {
    const char* err = ai_get_error(_network);
    aiLogErr("ai_destroy", err);
  }
  _network = AI_HANDLE_NULL;
  return;
}

// =================================================================
//   argmax
//
//   Description  : return argument of table maximum value
//   Argument     : Vector_db *vec: table
//   Return Value : int: index of max value
// =================================================================
static uint8_t argmax(int8_t* vec, uint32_t num) {
  uint32_t i;
  uint8_t arg = 0;
  int8_t imax = vec[0];
  for (i = 1; i < num; i++) {
    imax = (imax > vec[i]) ? imax : vec[i];
    if (imax == vec[i]) {
      arg = i;
    }
  }
  return (arg);
}

// ======================================================
//   aiRun
// ======================================================
static int aiRun(void) {
  ai_status err = AI_STATUS_OK;

  //
  // Inputs
  //
  ai_tensor* input = ai_get_input(_network, 0);
  if (input == NULL) {
    const char* err = ai_get_error(_network);
    aiLogErr("ai_run", err);
    return -1;
  }

  //
  // Outputs
  //
  ai_tensor* output = ai_get_output(_network, 0);
  if (output == NULL) {
    const char* err = ai_get_error(_network);
    aiLogErr("ai_run", err);
    return -1;
  }

  DLDataType out_dtype = output->dltensor.dtype;
  if (out_dtype.lanes > 1) {
    printf("E: vector outputs are not supported ...\r\n");
    return -1;
  }

  uint32_t elts = get_tensor_elts(output);

  char outfile_name[128];
  sprintf(outfile_name, "%s/tvm_results.txt", BUILD_PATH);  // NOLINT
  FILE* outfile = fopen(outfile_name, "w");

  for (int i = 0; i <= 9; i++) {
    char image[128];

    sprintf(image, "%s/0%d.raw", IMAGE_PATH, i);  // NOLINT
    printf("Loading input image %s ... \n", image);
    if (LoadInputImg(image, input) != 0) {
      error("Loading image %s\n", image);
    }

    //
    // Run the inference
    //
    printf("Running the network\r\n");

    if (ai_run(_network) != AI_STATUS_OK) {
      const char* err = ai_get_error(_network);
      aiLogErr("ai_run", err);
      return -1;
    }

    const ai_quantization_info* output_quant = ai_get_quantization(output);
    if (output_quant == NULL) {
      //
      // Floating point model
      //
      float* probabilities = (float*)output->dltensor.data;  // NOLINT
      for (int i = 0; i < elts; i++) {
        float val = probabilities[i];
        // printf (" -- probability[%d] = %g \n", i, val);
        fprintf(outfile, "%g ", val);
      }

    } else {
      //
      // Quantized model
      //
      if (out_dtype.code == kDLInt) {
        int8_t* probabilities = (int8_t*)output->dltensor.data;  // NOLINT
        for (int i = 0; i < elts; i++) {
          int8_t qval = probabilities[i];
          // printf (" -- probability[%d] = %d \n", i, qval);
          float val = dequantize_val(qval, output_quant);
          fprintf(outfile, "%g ", val);
        }
      } else {
        uint8_t* probabilities = (uint8_t*)output->dltensor.data;  // NOLINT
        for (int i = 0; i < elts; i++) {
          uint8_t qval = probabilities[i];
          // printf (" -- probability[%d] = %d \n", i, qval);
          float val = dequantize_val(qval, output_quant);
          fprintf(outfile, "%g ", val);
        }
      }
    }
    fprintf(outfile, "\n");
  }
  fclose(outfile);

  return 0;
}

// =================================================================
//   quantize_val
// =================================================================
static int32_t quantize_val(float val, ai_quantization_info* quant) {
  float new_val;
  float input_scale = quant->scale[0];
  int32_t input_zero_point = quant->zero_point[0];
  new_val = val / input_scale + input_zero_point;
  return (int32_t)new_val;
}

// =================================================================
//   dequantize_val
// =================================================================
static float dequantize_val(int32_t val, ai_quantization_info* quant) {
  float new_val;
  float output_scale = quant->scale[0];
  int32_t output_zero_point = quant->zero_point[0];
  new_val = (val - output_zero_point) * output_scale;
  return new_val;
}

// =================================================================
//   LoadInputImg
// =================================================================
uint8_t LoadInputImg(const char* filename, ai_tensor* input) {
  DLDataType dtype = input->dltensor.dtype;

  const ai_quantization_info* input_quant = ai_get_quantization(input);

  if (dtype.lanes > 1) {
    printf("E: vector inputs are not supported ...\r\n");
    return -1;
  }

  if (dtype.code == kDLBfloat) {
    printf("E: Double float inputs are not supported ...\r\n");
    return -1;
  }

  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    printf("== File %s not found\n", filename);
    return (-1);
  }

  //
  // Find file size
  //
  fseek(file, 0L, SEEK_END);
  size_t img_size = ftell(file);
  (void)fseek(file, 0L, SEEK_SET);

  // printf ("== Image size = %d\n", img_size);

  uint8_t* image = (uint8_t*)malloc(img_size);  // NOLINT
  size_t size = fread(image, 1, img_size, file);
  if (size != img_size) {
    perror("fread");
    printf("== Problem reading %s\n", filename);
    return (-1);
  }

  fclose(file);

  uint32_t x;
  uint8_t* p = image;
  uint8_t* pg = (uint8_t*)input->dltensor.data;  // NOLINT

  for (x = 0; x < img_size; x++) {
    uint8_t val = p[x];
    //
    // Input image needs to be normalized into [0..1] interval
    //
    float nval = ((float)val) / 255.0;  // NOLINT
    if (input_quant != NULL) {
      if (dtype.code == kDLInt) {
        int8_t qval = quantize_val(nval, input_quant);
        *pg = qval;
        pg += sizeof(int8_t);
      } else {
        uint8_t qval = quantize_val(nval, input_quant);
        *pg = qval;
        pg += sizeof(uint8_t);
      }
    } else {
      *(float*)pg = nval;  // NOLINT
      pg += sizeof(float);
    }
  }

  free(image);

  return 0;
}

// ======================================================
//   main
// ======================================================
int main(int argc, char* argv[]) {
  int status;

  status = aiInit();
  if (status != 0) {
    printf("Error initializing.\n");
  }

  status = aiRun();
  if (status != 0) {
    printf("Error running.\n");
  }

  aiDeInit();

  return (0);
}
