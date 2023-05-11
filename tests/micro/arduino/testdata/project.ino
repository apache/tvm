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

#include "src/platform.h"
#include "src/data/yes.c"
#include "src/data/no.c"
#include "src/data/unknown.c"
#include "src/data/silence.c"
#include "src/standalone_crt/include/tvm/runtime/crt/platform.h"

void performInference(int8_t input_data[1960], char *data_name) {
  int8_t output_data[4];
  unsigned long start_time = micros();
  TVMExecute(input_data, output_data);
  unsigned long end_time = micros();

  Serial.print(data_name);
  Serial.print(",");
  Serial.print(end_time - start_time);
  Serial.print(",");
  for (int i = 0; i < 4; i++) {
    Serial.print(output_data[i]);
    Serial.print(",");
  }
  Serial.println();
}

void setup() {
  TVMPlatformInitialize();
  Serial.begin(115200);
}

void loop() {
  Serial.println();
  Serial.println("category,runtime,yes,no,silence,unknown");
  performInference((int8_t*) input_yes, "yes");
  performInference((int8_t*) input_no, "no");
  performInference((int8_t*) input_silence, "silence");
  performInference((int8_t*) input_unknown, "unknown");
}
