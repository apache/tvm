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

#include <stdio.h>

#include "def.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "mfcc.h"
#include "mfcc_preprocessor.h"
#include "mic_reader.h"
#include "scope_timer.h"

// TVM stuff
#include <tvm_runtime.h>
#include <tvmgen_kws.h>

#include "inputs.h"
#include "outputs.h"

// Interprep inference result
bool check_result(float* output, size_t output_len, int64_t tm_from) {
  const char* chances[] = {"_silence_", "_unknown_", "yes", "no",  "up",   "down",
                           "left",      "right",     "on",  "off", "stop", "go"};

  size_t f = 0;
  for (size_t i = 0; i < output_len; i++) {
    if (output[i] > output[f]) f = i;
  }
#ifdef A_MODE_DEBUG
  for (size_t i = 0; i < output_len; i++) {
    DEBUG_PRINTF("  %f - %s\n", output[i],
                 ((sizeof(chances) / sizeof(*chances)) > i) ? chances[i] : "_unknown_");
  }
#endif  // A_MODE_DEBUG
  if (output[f] >= A_INFERENCE_POINT) {
    printf("***** word is '%s' (%f, %lld us) *****\n",
           ((sizeof(chances) / sizeof(*chances)) > f) ? chances[f] : "_unknown_", output[f],
           esp_timer_get_time() - tm_from);
    return true;
  }
  return false;
}

extern "C" void app_main(void) {
  printf("inmp441 %dbit sample capture\n", A_HW_BITS_PER_SAMPLE_DATA);

  int64_t t1, t2;
  int64_t tm_inference_from;

  // Model input is MFCC of input signal
  float* mfccBuffer = (float*)input;

  LOG_TIME(MicReader mic, "MicReader");
  LOG_TIME(MfccPreprocessor pp, "MfccPreprocessor");
  LOG_TIME(mic.Setup(), "Setup");

  if (pp.Errors() || mic.Errors()) {
    printf("init errors. stop\n");
    while (1) {
      vTaskDelay(1);
    }
    return;
  }

  printf("ready. waiting...\n");
  while (1) {
    // Find first non-silent fragment
    if (mic.Prepare()) {
      printf("\b\rprocessing...\n\b\r");
      tm_inference_from = esp_timer_get_time();
      while (1) {
        // Collect miningful signal
        if (!mic.Collect()) {
          // Unknown!
          printf("***** unknown word (%lld us) *****\n", esp_timer_get_time() - tm_inference_from);
          break;
        }
        // Find MFCC
        {
          DEBUG_SCOPE_TIMER("MFCC");
          pp.Apply(mic.AudioBuffer(), mic.KwsDurationSize(), mfccBuffer);
        }
        // Setup model's endpoints
        struct tvmgen_kws_outputs kws_outputs = {
            .Identity = output,
        };
        struct tvmgen_kws_inputs kws_inputs = {
            .input = input,
        };
        // Run inference
        {
          DEBUG_SCOPE_TIMER("TVM");
          tvmgen_kws_run(&kws_inputs, &kws_outputs);
        }
        // Find category
        if (check_result(reinterpret_cast<float*>(output), output_len, tm_inference_from)) break;
        vTaskDelay(1);
      }
      printf("ready. waiting...\n");
    }
    vTaskDelay(1);
  }
}
