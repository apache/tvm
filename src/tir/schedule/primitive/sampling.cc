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
#include <tvm/target/target.h>
#include <tvm/tir/schedule/schedule.h>

#include "../primitive.h"
#include "../utils.h"

namespace tvm {
namespace tir {

int SampleInt(TRandState* rand_state, int min_inclusive, int max_exclusive) {
  RandEngine rand_(rand_state);

  if (min_inclusive + 1 == max_exclusive) {
    return min_inclusive;
  }
  std::uniform_int_distribution<> dist(min_inclusive, max_exclusive - 1);
  return dist(rand_);
}

}  // namespace tir
}  // namespace tvm
