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

#include <fstream>
#include <iostream>
#include <sstream>

// The max loop/function id used among all lwp_handler calls. Since
// the id is used to index into the lwp_counter buffer, the size of the
// buffer must be equal or greater than the max possible id.
#define LWP_COUNTER_SIZE 5000

// LWP_BUFFER_SIZE needs to be at most 100 * LWP_COUNTER_SIZE since 100 is
// the max number of entries recorded for each instrumented location.
#define LWP_BUFFER_SIZE (LWP_COUNTER_SIZE * 100)

uint32_t lwp_counter[LWP_COUNTER_SIZE] = {0};
uint32_t lwp_buffer[LWP_BUFFER_SIZE];
uint32_t* __lwp_counter = lwp_counter;
uint32_t* __lwp_buffer_ptr = lwp_buffer;
uint32_t __lwp_buffer_size = LWP_BUFFER_SIZE;
uint32_t __lwp_enable_flag = 1;
uint32_t __lwp_buffer_count = 0;

bool WriteLWPOutput(const std::string& out_json) {
  std::ostringstream s;
  s << "{\n";
  s << "\t\"entries\":[\n";
  for (size_t i = 0; i < __lwp_buffer_count; i += 4) {
    s << "\t{\n";
    s << "\t\t\"ret\":" << std::dec << lwp_buffer[i] << ",\n";
    s << "\t\t\"id\":" << std::dec << lwp_buffer[i + 1] << ",\n";
    uint64_t pcycles = (static_cast<uint64_t>(lwp_buffer[i + 3]) << 32) + lwp_buffer[i + 2];
    s << "\t\t\"cyc\":" << std::dec << pcycles << "\n";
    s << "\t}";
    if (i < __lwp_buffer_count - 4) {
      s << ",\n";
    }
  }
  s << "\t],\n\n";
  s << "\t\"loop_counts\":[\n";
  for (size_t i = 0; i < LWP_COUNTER_SIZE; i++) {
    s << "\t\t" << lwp_counter[i] / 2;
    if (i < LWP_COUNTER_SIZE - 1)
      s << ",\n";
    else
      s << "\n";
  }
  s << "\t]\n}\n";
  std::ofstream ofc(out_json);
  if (!ofc.is_open()) {
    return false;
  }

  ofc << s.str() << "\n";

  if (!ofc) {
    return false;
  }
  ofc.close();
  return true;
}
