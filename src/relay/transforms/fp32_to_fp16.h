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
 * \file fp32_to_fp16.h
 * \brief Utilities and common types used for FP32->FP16 pass.
 */
#ifndef TVM_RELAY_TRANSFORMS_FP32_TO_FP16_H_
#define TVM_RELAY_TRANSFORMS_FP32_TO_FP16_H_

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

struct FP16OpDType {
  DataType accumulation_dtype;
  DataType output_dtype;
};

// GREEN colored ops should always be done in FP16 due to the speed and memory savings
// GRAY colored ops can be done in FP16 but don't have speedups to justify a dedicated cast.
// RED colored ops should not be done in FP16 due to numerical reasons.
enum FP16ConversionCategory { RED, GRAY, GREEN };
std::unordered_map<FP16ConversionCategory, std::string> conversion_category_strings(
    {{RED, "Red"}, {GRAY, "Gray"}, {GREEN, "Green"}});

using OpStringSet = std::unordered_set<std::string>;

// Default lists inspired from TF's classifications:
// github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
// They have a bias toward Nvidia Tensor Cores so modify lists per your hardware choice.
OpStringSet DEFAULT_GREEN_LIST({
    "nn.conv1d",
    "nn.conv2d",
    "nn.conv3d",
    "nn.conv1d_transpose",
    "nn.conv2d_transpose",
    "nn.conv3d_transpose",
    "nn.dense",
    "nn.batch_matmul",
});
OpStringSet DEFAULT_GRAY_LIST({
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
    // By definition copy and cast will become green or red based on inputs
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
OpStringSet DEFAULT_RED_LIST({
    // In general if |f(x)| >> |x| for expected inputs then put the op here.
    "exp",
    "power",
    "nn.cross_entropy",
    "nn.cross_entropy_with_logits",
    "nn.softmax",
    "nn.l2_normalize",
    // Error function doesn't seem to be able to be lowered into fp16 version in llvm.
    // Move to gray list when it does.
    "erf",
});

class DefaultFP16Colorer {
  /* The default class to initially color ops for conversion using lists.

  Creates a callable which given a CallNode* returns the node's color.
  */
 private:
  std::unordered_map<std::string, FP16ConversionCategory> op_to_initial_color;

 public:
  DefaultFP16Colorer(OpStringSet red_list = DEFAULT_RED_LIST,
                     OpStringSet gray_list = DEFAULT_GRAY_LIST,
                     OpStringSet green_list = DEFAULT_GREEN_LIST) {
    std::vector<std::pair<OpStringSet, FP16ConversionCategory>> lists_and_colors{
        {red_list, RED}, {gray_list, GRAY}, {green_list, GREEN}};

    for (auto list_and_color : lists_and_colors) {
      OpStringSet ops = list_and_color.first;
      FP16ConversionCategory color = list_and_color.second;
      for (std::string op_name : ops) {
        op_to_initial_color.insert({{op_name, color}});
      }
    }
  }

  FP16ConversionCategory operator()(const CallNode* call, bool ignore_missing = true) {
    if (auto* op_node = (call->op).as<tvm::OpNode>()) {
      std::string op_name = op_node->name;
      auto color = op_to_initial_color.find(op_name);

      if (color == op_to_initial_color.end()) {
        if (ignore_missing) {
          LOG(WARNING) << "Op name " << op_name << " not in included in fp16 conversion lists!.";
          return RED;
        } else {
          LOG(FATAL) << "Op name " << op_name << " not in included in fp16 lists!.";
        }
      }

      return color->second;
    } else if (auto* func_node = (call->op).as<FunctionNode>()) {
      // Make RED to avoid messing with function headers.
      return RED;
    } else {
      LOG(FATAL) << "FP16 conversion only supports call nodes with OpNodes or Functions got "
                 << call->op;
      return RED;
    }
  }
};

class DefaultFP16OpDefinition {
  /* The default callable for determining accumulation_dtypes for ops. */
 public:
  FP16OpDType operator()(const CallNode* call) {
    // TODO(AndrewZhaoLuo): remove when batch_matmul handles accumulation dtypes well.
    // Batched matmul has inconsistent support for mixed precision operations.
    // Many schedules ignore the out_dtype attribute which leads to errors when
    // input types do not match the out_dtype. Therefore, accumulate to fp16 if green.
    if (auto op_node = call->op.as<OpNode>()) {
      if (op_node->name == "nn.batch_matmul") {
        return {DataType::Float(16), DataType::Float(16)};
      }
    }

    // We assume the "out_dtype" field is always an accumulation dtype specification.
    if (call->attrs != NullValue<Attrs>()) {
      Array<AttrFieldInfo> fields = call->attrs->ListFieldInfo();
      for (AttrFieldInfo field_info : fields) {
        if (field_info->name == "out_dtype") return {DataType::Float(32), DataType::Float(16)};
      }
    }

    return {DataType::Float(16), DataType::Float(16)};
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_FP32_TO_FP16_H_
