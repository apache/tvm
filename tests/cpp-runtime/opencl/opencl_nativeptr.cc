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
#include <tvm/runtime/container/optional.h>

#include "../src/runtime/opencl/opencl_common.h"

using namespace tvm::runtime;
using namespace tvm::runtime::cl;

#if defined(OPENCL_ENABLE_HOST_PTR)
TEST(OpenCLNDArray, native_ptr) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();

  auto A = tvm::runtime::NDArray::Empty({128, 128}, {kDLFloat, 32, 1}, {kDLOpenCL, 0});
  void* nptr = workspace->GetNativePtr(A);
  memset(nptr, 0x0, 128 * 128 * 4);
}
#endif
