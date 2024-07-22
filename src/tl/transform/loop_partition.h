/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file loop_partition.h
 * \brief Partition parallel loops onto threads
 */

#ifndef TVM_TL_LOOP_PARTITION_H_
#define TVM_TL_LOOP_PARTITION_H_

#include <tvm/tir/op.h>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tir;

For PartitionLoop(For op, Var thread_var, arith::Analyzer* analyzer, Fragment loop_layout);

Fragment PlanLoopPartition(For op, size_t num_thread, int vectorize_size);

For LoopPragmaUnroll(For stmt);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_LOOP_PARTITION_H_
