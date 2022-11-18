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
 * \file tests/cpp/runtime/contrib/ethosn/inference_test.cc
 * \brief Tests to check Arm(R) Ethos(TM)-N runtime components used during inference.
 */

#ifdef ETHOSN_HW

#include <gtest/gtest.h>

#include "../../../../../src/runtime/contrib/ethosn/ethosn_device.cc"

namespace tvm {
namespace runtime {
namespace ethosn {

TEST(WaitForInference, InferenceScheduled) {
  const int inference_result = 0 /* Scheduled */;
  const int timeout = 0;

  dl::Inference inference = dl::Inference(inference_result);
  InferenceWaitStatus result = WaitForInference(&inference, timeout);

  ASSERT_EQ(result.GetErrorCode(), InferenceWaitErrorCode::kTimeout);
  ICHECK_EQ(result.GetErrorDescription(), "Timed out while waiting for the inference to complete.");
}

TEST(WaitForInference, InferenceRunning) {
  const int inference_result = 1 /* Running */;
  const int timeout = 0;

  dl::Inference inference = dl::Inference(inference_result);
  InferenceWaitStatus result = WaitForInference(&inference, timeout);

  ASSERT_EQ(result.GetErrorCode(), InferenceWaitErrorCode::kTimeout);
  std::cout << result.GetErrorDescription() << std::endl;
  ICHECK_EQ(result.GetErrorDescription(), "Timed out while waiting for the inference to complete.");
}

TEST(WaitForInference, InferenceError) {
  const int inference_result = 3 /* Error */;
  const int timeout = 0;

  dl::Inference inference = dl::Inference(inference_result);
  InferenceWaitStatus result = WaitForInference(&inference, timeout);

  ASSERT_EQ(result.GetErrorCode(), InferenceWaitErrorCode::kError);
  ICHECK_EQ(result.GetErrorDescription(),
            "Failed to read inference result status (No such file or directory)");
}

}  // namespace ethosn
}  // namespace runtime
}  // namespace tvm

#endif
