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
extern "C" {
#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <qurt_error.h>
#include <qurt_hvx.h>
}

#include "hexagon_common.h"
#include "hexagon_hvx.h"

namespace tvm {
namespace runtime {
namespace hexagon {

HexagonHvx::HexagonHvx() { Acquire(); }

HexagonHvx::~HexagonHvx() { Release(); }

void HexagonHvx::Acquire() {
  reserved_count_ = qurt_hvx_reserve(QURT_HVX_RESERVE_ALL);
  CHECK(reserved_count_ == QURT_HVX_RESERVE_ALL) << "error reserving HVX: " << reserved_count_;
}

void HexagonHvx::Release() {
  int rel = qurt_hvx_cancel_reserve();
  CHECK(rel == 0) << "error releasing HVX: " << rel;
}

void HexagonHvx::Lock() {
  int lck = qurt_hvx_lock(QURT_HVX_MODE_128B);
  CHECK(lck == 0) << "error locking HVX: " << lck;
}

void HexagonHvx::Unlock() {
  int unl = qurt_hvx_unlock();
  CHECK(unl == 0) << "error unlocking HVX: " << unl;
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
