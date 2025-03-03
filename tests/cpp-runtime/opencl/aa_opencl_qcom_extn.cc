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

// Note:: This should be first tests to be executed.
// hence, crafted the filename accordingly

#include <gtest/gtest.h>
#include <tvm/runtime/container/optional.h>

#include "../src/runtime/opencl/opencl_common.h"

using namespace tvm::runtime;
using namespace tvm::runtime::cl;

#ifdef USE_OPENCL_EXTN_QCOM
#pragma message("Qualcomm OpenCL Extn GTests: enabled")
TEST(QCOMExtn, ContextPriorityHint) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();
  cl_context_properties properties[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_LOW_QCOM,
                                        0};
  // Only allow one time
  ASSERT_EQ(workspace->Init(properties), true);
  // Subsequent calls will be failure
  ASSERT_EQ(workspace->Init(properties), false);
}

TEST(QCOMExtn, ContextPerfHint) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();
  auto dev = DLDevice{kDLOpenCL, 0};
  workspace->SetPerfHint(dev, CL_PERF_HINT_HIGH_QCOM);
}
#else
#pragma message("Qualcomm OpenCL Extn GTests: disabled")
#endif
