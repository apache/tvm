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

#include <gtest/gtest.h>
#include <tvm/runtime/profiling.h>

#include "../src/runtime/opencl/opencl_common.h"

using namespace tvm::runtime;
using namespace tvm::runtime::cl;

#define BUFF_SIZE 1024
#define NUM_REPEAT 10

TEST(OpenCLTimerNode, nested_timers) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();
  OpenCLThreadEntry* thr = workspace->GetThreadEntry();

  int err;
  cl_int* tmp_buf = new cl_int[BUFF_SIZE];
  int64_t nested_time_sum = 0;

  auto did = workspace->GetCLDeviceID(thr->device.device_id);
  auto platform = workspace->device_to_platform[did];
  Timer init_timer = Timer::Start(thr->device);
  for (int i = 0; i < NUM_REPEAT; ++i) {
    Timer nested_timer = Timer::Start(thr->device);
    // create some events
    cl_event ev = clCreateUserEvent(workspace->contexts[platform], &err);
    OPENCL_CHECK_ERROR(err);
    cl_mem cl_buf = clCreateBuffer(workspace->contexts[platform], CL_MEM_READ_ONLY,
                                   BUFF_SIZE * sizeof(cl_int), nullptr, &err);
    OPENCL_CHECK_ERROR(err);
    auto queue = workspace->GetQueue(thr->device);
    OPENCL_CALL(clEnqueueWriteBuffer(queue, cl_buf, false, 0, BUFF_SIZE * sizeof(cl_int), tmp_buf,
                                     0, nullptr, &ev));
    OPENCL_CALL(clReleaseMemObject(cl_buf));
    workspace->events[thr->device.device_id].push_back(ev);
    nested_timer->Stop();
    nested_time_sum += nested_timer->SyncAndGetElapsedNanos();
  }
  init_timer->Stop();

  delete[] tmp_buf;
  int64_t elapsed = init_timer->SyncAndGetElapsedNanos();
  CHECK_EQ(elapsed, nested_time_sum);
}
