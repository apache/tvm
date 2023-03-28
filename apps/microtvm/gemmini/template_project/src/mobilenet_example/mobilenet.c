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

#include "assert.h"
#include "stddef.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#ifndef BAREMETAL
#include "sys/mman.h"
#endif
#include "input.h"
#include "model/tvmgen_default.h"
#include "output.h"

uint8_t output_pred[1001];

int argmax(uint8_t* vec) {
  int idx = 0;
  uint8_t max_value = 0;
  for (int i = 0; i < 1001; i++) {
    if (vec[i] > max_value) {
      idx = i;
      max_value = vec[i];
    }
  }
  return idx;
}

void get_top_5_labels(int* top_5, uint8_t* predicted_output) {
  uint8_t prev_max_value = (uint8_t)255;
  uint8_t current_max_value = 0;
  int idx = 0;
  for (int i = 0; i < 5; i++) {
    current_max_value = 0;
    idx = 0;
    for (int j = 0; j < 1001; j++) {
      if ((predicted_output[j] > current_max_value) && (predicted_output[j] < prev_max_value)) {
        current_max_value = predicted_output[j];
        idx = j;
      }
    }
    top_5[i] = idx;
    prev_max_value = current_max_value;
  }
}

int main() {
  printf("Starting MobileNet test...\r\n");
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  int top_5_labels[5];

  struct tvmgen_default_inputs inputs;
  inputs.input = input;
  struct tvmgen_default_outputs outputs;
  outputs.MobilenetV2_Predictions_Reshape = output_pred;
  int error_counter = 0;

  tvmgen_default_run(&inputs, &outputs);

  // Look for errors!
  /*for(int i = 0; i < output_len; i++)
  {
          if(output_pred[i] != output[i])
{
error_counter += 1;
printf("ERROR IN MOBILENET EXAMPLE! output_pred[%d] (%d) != output[%d]
(%d)\r\n",i,(int)output_pred[i],i,(int)output[i]);
//exit(1);
}
  }*/

  get_top_5_labels(top_5_labels, output_pred);

  printf("Real Top-5 output labels: [ ");
  for (int i = 0; i < 5; i++) printf("%d ", (int)top_5_labels[i]);
  printf("]\r\n");

  printf("Expected Top-5 output labels: [ ");
  for (int i = 0; i < 5; i++) printf("%d ", (int)output[i]);
  printf("]\r\n");

  /*for(int i = 0; i < 5; i++)
        {
                if(top_5_labels[i] != output[i])
    {
      error_counter += 1;
      printf("ERROR IN MOBILENET EXAMPLE! top_5_labels[%d] (%d) != output[%d]
    (%d)\r\n",i,(int)top_5_labels[i],i,(int)output[i]);
      //exit(1);
    }
        }*/

  // printf("SUCCESS!\r\n");
  exit(0);

  // Take the argmax to get the predicted label, and the expected label
  /*int predicted_label = argmax(output_pred);
  int expected_label = argmax(output);
  printf("Expected label = %d\r\n",expected_label);
  printf("Predicted label = %d\r\n",predicted_label);
  if(expected_label == predicted_label) printf("SUCCESS!\r\n");
  else printf("FAILED!\r\n");
  exit(0);*/
}
