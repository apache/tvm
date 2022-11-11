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
#include <tvm/meta_schedule/schedule/generic/winograd.h>

namespace tvm {
namespace meta_schedule {

using namespace tvm::tir;

/*!
 * \brief Get the producer block of a given block.
 * If there is a constant winograd transform matrix, inline it.
 * \return The only producer block.
 */
BlockRV GetWinogradProducerAndInlineConst(Schedule sch, BlockRV block) {
  Array<BlockRV> producers = sch->GetProducers(block);
  Array<BlockRV> results;
  for (const BlockRV& producer : producers) {
    if (sch->Get(producer)->reads.empty()) {
      sch->ComputeInline(producer);
    } else {
      results.push_back(producer);
    }
  }
  ICHECK_EQ(results.size(), 1);
  return results[0];
}

}  // namespace meta_schedule
}  // namespace tvm
