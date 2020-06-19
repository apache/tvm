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

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/te/operation.h>
#include <functional>
#include "kernel_layout_transform.h"

namespace tvm {
namespace relay {

// Todo: do not use global variables
std::deque<std::string> KernelLayoutVisitor::global_ori_layouts_queue;
std::deque<std::string> KernelLayoutVisitor::global_new_layouts_queue;

Expr KernelLayoutTransform(const Expr& expr) {
  KernelLayoutVisitor visitor;

  // Do a pre-order DFS to gather the optimal kernel layouts for all conv2d nodes.
  // These layouts were written to global static variables in python function `prepare_layout_rewrite`
  visitor.VisitExpr(expr);

  // Do a post-order DSF to mutate layout for all conv2d nodes
  return KernelLayoutTransformer(&visitor).Mutate(expr);
}

namespace transform {

Pass KernelLayoutTransform() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(relay::KernelLayoutTransform(f));
  };
  return CreateFunctionPass(pass_func, 3, "KernelLayoutTransform",
                            {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.KernelLayoutTransform")
.set_body_typed(KernelLayoutTransform);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
