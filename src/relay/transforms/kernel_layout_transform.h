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
#ifndef TVM_RELAY_TRANSFORMS_KERNEL_LAYOUT_TRANSFORM_H_
#define TVM_RELAY_TRANSFORMS_KERNEL_LAYOUT_TRANSFORM_H_

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <deque>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../../ansor/compute_dag.h"
#include "pattern_util.h"

namespace tvm {
namespace relay {

/*! \brief A visitor to gather the optimal kernel layout for all conv2d nodes. */
class KernelLayoutVisitor : public ExprVisitor {
 public:
  void VisitExpr_(const CallNode* n) {
    if (n && n->op.as<OpNode>() &&
        (std::find(op_white_lists.begin(), op_white_lists.end(), n->op.as<OpNode>()->name) !=
         op_white_lists.end()) &&
        n->args[1]->type_as<TensorTypeNode>()->shape[3].as<IntImmNode>()->value > 1 &&
        !global_ori_layouts_queue.empty() && !global_new_layouts_queue.empty()) {
      ori_layouts_map[n] = global_ori_layouts_queue.front();
      new_layouts_map[n] = global_new_layouts_queue.front();
      // std::cout << "ori_layout " << global_ori_layouts_queue.front()
      //     << " Filter_shape " << n->args[1]->type_as<TensorTypeNode>()->shape << std::endl;
      global_ori_layouts_queue.pop_front();
      global_new_layouts_queue.pop_front();
    }
    ExprVisitor::VisitExpr_(n);
  }

  std::unordered_map<const CallNode*, std::string> ori_layouts_map;
  std::unordered_map<const CallNode*, std::string> new_layouts_map;
  std::vector<std::string> op_white_lists{"nn.contrib_conv2d_winograd_without_weight_transform",
                                          "nn.conv2d", "nn.conv3d"};

  static std::deque<std::string> global_ori_layouts_queue;
  static std::deque<std::string> global_new_layouts_queue;
};

/*! \brief A mutator to rewrite kernel layout for all conv2d nodes */
class KernelLayoutTransformer : public ExprMutator {
 public:
  explicit KernelLayoutTransformer(KernelLayoutVisitor* visitor)
      : ExprMutator(), visitor_(visitor) {}

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);

    const auto* call = new_n.as<CallNode>();
    std::vector<std::string> op_white_lists{"nn.contrib_conv2d_winograd_without_weight_transform",
                                            "nn.conv2d", "nn.conv3d"};
    if (call && call->op.as<OpNode>() &&
        (std::find(op_white_lists.begin(), op_white_lists.end(), n->op.as<OpNode>()->name) !=
             op_white_lists.end() &&
         n->args[1]->type_as<TensorTypeNode>()->shape[3].as<IntImmNode>()->value > 1)) {
      auto ori_layout_iter = visitor_->ori_layouts_map.find(n);
      auto new_layout_iter = visitor_->new_layouts_map.find(n);
      if (ori_layout_iter != visitor_->ori_layouts_map.end() &&
          new_layout_iter != visitor_->new_layouts_map.end()) {
        const std::string& ori_layout = ori_layout_iter->second;
        const std::string& new_layout = new_layout_iter->second;
        Expr updated_kernel = MakeKernelLayoutTransform(call->args[1], ori_layout, new_layout);
        Array<Expr> updated_args = {call->args[0], updated_kernel};
        new_n = Call(call->op, updated_args, call->attrs);
      }
    }
    return new_n;
  }

 private:
  KernelLayoutVisitor* visitor_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_KERNEL_LAYOUT_TRANSFORM_H_
