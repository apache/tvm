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
 *
 * \file src/tir/analysis/collect_call_map.cc
 *
 * \brief Collect cross-IR call graph
 */

#include <tvm/ir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

namespace {
using ir::CalleeCollector;

struct Visitor : StmtExprVisitor {
  explicit Visitor(CalleeCollector* collector) : collector(collector) {}
  CalleeCollector* collector;
  void VisitExpr_(const CallNode* node) override {
    StmtExprVisitor::VisitExpr_(node);
    if (auto opt_gvar = node->op.as<GlobalVar>()) {
      collector->Mark(opt_gvar.value());
    }
  }
};

}  // namespace

TVM_STATIC_IR_FUNCTOR(CalleeCollector, vtable)
    .set_dispatch<tir::PrimFuncNode>([](const ObjectRef& func, CalleeCollector* collector) {
      Visitor visitor{collector};
      visitor(Downcast<tir::PrimFunc>(func)->body);
    });

}  // namespace tir
}  // namespace tvm
