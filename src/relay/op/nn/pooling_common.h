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
 * \file src/relay/op/nn/pooling_common.h
 * \brief Common functions for pooling operator definition.
 */
#ifndef TVM_RELAY_OP_NN_POOLING_COMMON_H_
#define TVM_RELAY_OP_NN_POOLING_COMMON_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>

#include <string>
#include <utility>
#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relay {

inline IndexExpr calculate_pool_dimension(IndexExpr in_dimension, IndexExpr pad_amount,
                                          IndexExpr pool_size, IndexExpr dilation,
                                          IndexExpr stride_size, bool ceil_mode) {
  IndexExpr numerator = in_dimension + pad_amount - ((pool_size - 1) * dilation + 1);
  IndexExpr denominator = stride_size;

  // Emulate the behavior of running ceil on numerator / denominator rather than floor
  if (ceil_mode) {
    numerator += denominator - 1;
  }

  return numerator / denominator + 1;
}

template <typename T>
InferCorrectLayoutOutput PoolInferCorrectLayout(const Attrs& attrs,
                                                const Array<Layout>& new_in_layouts,
                                                const Array<Layout>& old_in_layouts,
                                                const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<T>();
  ICHECK(attrs_ptr);
  ObjectPtr<T> params = make_object<T>(*attrs_ptr);

  if (params->out_layout != "") {
    // when users specify the out_layout of pooling, follow user's preference
    ICHECK_EQ(params->layout, params->out_layout)
        << "Pooling input/output layouts mismatch: " << params->layout << " vs. "
        << params->out_layout;
  } else if (new_in_layouts.defined()) {
    // the pooling is using an inferred layout (i.e., new_in_layouts[0]) given by relay caller
    // ICHECK_EQ(new_in_layouts.size(), 1);
    params->layout = new_in_layouts[0].name();
  }

  return InferCorrectLayoutOutput({params->layout}, {params->layout}, Attrs(params));
}
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_POOLING_COMMON_H_
