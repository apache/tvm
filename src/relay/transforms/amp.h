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
 * \file amp.h
 * \brief Utilities and common types used for automatic mixed precision pass.
 */
#ifndef TVM_RELAY_TRANSFORMS_AMP_H_
#define TVM_RELAY_TRANSFORMS_AMP_H_

#include <tvm/ir/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {

struct MixedPrecisionOpOutDType {
  DataType accumulation_dtype;
  DataType output_dtype;
};

// MIXED_PRECISION_ALWAYS ops should always be done in lower precision due to the speed and memory
// savings. MIXED_PRECISION_FOLLOW ops can be done in lower precision but don't have speedups to
// justify a cast. MIXED_PRECISION_NEVER colored ops should not be done in lower precision due to
// numerical reasons.
enum MixedTypeConversionCategory {
  MIXED_PRECISION_ALWAYS,
  MIXED_PRECISION_FOLLOW,
  MIXED_PRECISION_NEVER
};

using OpStringSet = std::unordered_set<std::string>;

// Default lists inspired from TF's classifications:
// github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
// They have a bias toward Nvidia Tensor Cores so modify lists per your hardware choice.
OpStringSet DEFAULT_ALWAYS_LIST({
    "nn.conv1d",
    "nn.conv2d",
    "nn.conv3d",
    "nn.conv1d_transpose",
    "nn.conv2d_transpose",
    "nn.conv3d_transpose",
    "nn.dense",
    "nn.batch_matmul",
});
OpStringSet DEFAULT_FOLLOW_LIST({
    // These ops add new data or change shape
    "nn.pad",
    "nn.batch_flatten",
    "concatenate",
    "zeros",
    "split",
    "squeeze",
    "transpose",
    "expand_dims",
    "reshape",
    "dyn.reshape",
    "broadcast_to_like",
    "dyn.broadcast_to",
    "strided_slice",
    "dyn.strided_slice",
    "take",
    "argwhere",
    "where",
    "tile",
    "dyn.tile",
    "scatter",
    "full",
    "dyn.full",
    // Comparison
    "less",
    "greater",
    "less_equal",
    "greater_equal",
    // By definition copy and cast will depend on inputs for output.
    "copy",
    "cast",
    "cast_like",
    // Simple arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    "nn.bias_add",
    "nn.batch_norm",
    "sum",
    "mean",
    "sqrt",
    "shape_of",
    // Simple activations
    "max",
    "min",
    "maximum",
    "minimum",
    "nn.relu",
    "nn.leaky_relu",
    "nn.prelu",
    "nn.dropout",
    // Complicated activations which saturate in a narrow range
    "sigmoid",
    "tanh",
    // Pooling operations
    "nn.max_pool1d",
    "nn.max_pool2d",
    "nn.max_pool3d",
    "nn.avg_pool1d",
    "nn.avg_pool2d",
    "nn.avg_pool3d",
    // "nn.global_max_pool1d", // does not exist yet
    "nn.global_max_pool2d",
    // "nn.global_max_pool3d", // does not exist yet
    // "nn.global_avg_pool1d", // does not exist yet
    "nn.global_avg_pool2d",
    // "nn.global_avg_pool3d", // does not exist yet
    "nn.adaptive_max_pool1d",
    "nn.adaptive_max_pool2d",
    "nn.adaptive_max_pool3d",
    "nn.adaptive_avg_pool1d",
    "nn.adaptive_avg_pool2d",
    "nn.adaptive_avg_pool3d",
});
OpStringSet DEFAULT_NEVER_LIST({
    // In general if |f(x)| >> |x| for expected inputs then put the op here.
    "exp",
    "power",
    "nn.cross_entropy",
    "nn.cross_entropy_with_logits",
    "nn.softmax",
    "nn.l2_normalize",
    // Error function doesn't seem to be able to be lowered into fp16 version in llvm.
    // Move to follow list when it does.
    "erf",
});

class DefaultMixedPrecisionColorer {
  /* The default class to initially color ops for conversion using lists.
  Default lists are for NVidia Tensor Cores and FP16.

  Creates a callable which given a CallNode* returns the node's color.
  */
 private:
  std::unordered_map<std::string, MixedTypeConversionCategory> op_to_initial_color;

 public:
  DefaultMixedPrecisionColorer(OpStringSet never_list = DEFAULT_NEVER_LIST,
                               OpStringSet follow_list = DEFAULT_FOLLOW_LIST,
                               OpStringSet always_list = DEFAULT_ALWAYS_LIST) {
    std::vector<std::pair<OpStringSet, MixedTypeConversionCategory>> lists_and_colors{
        {never_list, MIXED_PRECISION_NEVER},
        {follow_list, MIXED_PRECISION_FOLLOW},
        {always_list, MIXED_PRECISION_ALWAYS}};

    for (auto list_and_color : lists_and_colors) {
      OpStringSet ops = list_and_color.first;
      MixedTypeConversionCategory color = list_and_color.second;
      for (std::string op_name : ops) {
        op_to_initial_color.insert({{op_name, color}});
      }
    }
  }

  MixedTypeConversionCategory operator()(const CallNode* call, bool ignore_missing = true) {
    if (auto* op_node = (call->op).as<tvm::OpNode>()) {
      std::string op_name = op_node->name;
      auto color = op_to_initial_color.find(op_name);

      if (color == op_to_initial_color.end()) {
        (ignore_missing ? LOG(WARNING) : LOG(FATAL))
            << "Op name " << op_name << " not in included in conversion lists!";
        return MIXED_PRECISION_NEVER;
      }

      return color->second;
    } else if ((call->op).as<FunctionNode>()) {
      // Make MIXED_PRECISION_NEVER to avoid messing with function headers.
      return MIXED_PRECISION_NEVER;
    } else {
      LOG(FATAL) << "Conversion only supports call nodes with OpNodes or Functions got "
                 << call->op;
      return MIXED_PRECISION_NEVER;
    }
  }
};

class DefaultMixedPrecisionOpDefinition {
  /* The default callable for determining accumulation_dtypes for ops.

  Assumes accumulatable operations accumulate to one type and outputs are
  all of the same type.*/

  const DataType default_output_dtype;
  const DataType default_accumulation_dtype;

 public:
  DefaultMixedPrecisionOpDefinition(DataType default_output_dtype = DataType::Float(16),
                                    DataType default_accumulation_dtype = DataType::Float(32))
      : default_output_dtype(default_output_dtype),
        default_accumulation_dtype(default_accumulation_dtype) {}

  MixedPrecisionOpOutDType operator()(const CallNode* call) {
    // TODO(AndrewZhaoLuo): remove when batch_matmul handles accumulation dtypes well.
    // Batched matmul has inconsistent support for mixed precision operations.
    // Many schedules ignore the out_dtype attribute which leads to errors when
    // input types do not match the out_dtype. Therefore, accumulate to output_dtype.
    if (auto op_node = call->op.as<OpNode>()) {
      if (op_node->name == "nn.batch_matmul") {
        return {default_output_dtype, default_output_dtype};
      }
    }

    // We assume the "out_dtype" field is always an accumulation dtype specification.
    if (call->attrs != NullValue<Attrs>()) {
      Array<AttrFieldInfo> fields = call->attrs->ListFieldInfo();
      for (AttrFieldInfo field_info : fields) {
        if (field_info->name == "out_dtype")
          return {default_accumulation_dtype, default_output_dtype};
      }
    }

    return {default_output_dtype, default_output_dtype};
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_AMP_H_
