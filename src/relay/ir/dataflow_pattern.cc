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
 * \file src/tvm/relay/dataflow_pattern.cc
 * \brief The dataflow pattern language for Relay.
 */
#include <tvm/relay/dataflow_pattern.h>

namespace tvm {
namespace relay {

ExprPattern::ExprPattern(Expr expr) {
  ObjectPtr<ExprPatternNode> n = make_object<ExprPatternNode>();
  n->expr = std::move(expr);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExprPatternNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.ExprPattern")
.set_body_typed([](Expr e) {
    return ExprPattern(e);
  });

// TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
// .set_dispatch<ConstantNode>([](const ObjectRef& ref, ReprPrinter* p) {
//     auto* node = static_cast<const ConstantNode*>(ref.get());
//     const PackedFunc* fprint = Registry::Get("relay._constant_repr");
//     CHECK(fprint) << "unable to find printing function for constants";
//     std::string data = (*fprint)(GetRef<Constant>(node));
//     p->stream << "Constant(" << data << ")";
//   });

}  // namespace relay
}  // namespace tvm
