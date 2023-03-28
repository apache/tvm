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
#include "input_1.h"
#include "input_2.h"
#include "model/tvmgen_default.h"
#include "output.h"

int8_t output_add[OUTPUT_LEN];

int main() {
  printf("Starting add test...\r\n");
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  struct tvmgen_default_inputs inputs;
  inputs.serving_default_x_0 = input_1;
  inputs.serving_default_y_0 = input_2;
  struct tvmgen_default_outputs outputs;
  outputs.PartitionedCall_0 = output_add;
  int error_counter = 0;

  tvmgen_default_run(&inputs, &outputs);

  // Look for errors!
  for (int i = 0; i < OUTPUT_LEN; i++) {
    if (output_add[i] != output[i]) {
      error_counter += 1;
      printf("ERROR IN ADD EXAMPLE! output_add[%d] (%d) != output[%d] (%d)\r\n", i, output_add[i],
             i, output[i]);
      // exit(1);
    }
  }

  // We allow for a very small percentage of errors, this could be related to rounding errors
  float error_perc = ((float)(error_counter / OUTPUT_LEN) * 100);
  if (error_perc < 1)
    printf("SUCCESS! (error_counter = %d)\r\n", error_counter);
  else
    printf("FAIL! (error_counter = %d)\r\n", error_counter);
  exit(0);
}
