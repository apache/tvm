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
 * \file tvm/arithmetic/ir_visitor_with_analyzer.h
 * \brief IR visitor class with an analyzer context.
 */

#ifndef TVM_ARITHMETIC_IR_VISITOR_WITH_ANALYZER_H_
#define TVM_ARITHMETIC_IR_VISITOR_WITH_ANALYZER_H_

#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>

namespace tvm {
namespace ir {

class IRVisitorWithAnalyzer final : public IRVisitor {
 public:
  Expr Simplify(const Expr& expr) {
    return analyzer_.Simplify(expr);
  }

  void Visit_(const For* op) {
    analyzer_.Bind(op->loop_var,
                   Range::make_by_min_extent(op->min, op->extent));
    return IRVisitor::Visit_(op);
  }

  void Visit_(const AttrStmt* op) {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::virtual_thread) {
      IterVar iv(op->node.node_);
      CHECK_NE(iv->thread_tag.length(), 0U);
      analyzer_.Bind(iv->var,
                      Range::make_by_min_extent(0, op->value));
      IRVisitor::Visit_(op);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Reduce* op) {
    // Setup the domain information before simplification.
    for (const IterVar& iv : op->axis) {
      analyzer_.Bind(iv->var, iv->dom);
    }
    // Recursively call simplification when necessary.
    IRVisitor::Visit_(op);
  }

 protected:
  /*! \brief internal analyzer field. */
  arith::Analyzer analyzer_;
};

}  // namespace ir
}  // namespace tvm
#endif  // TVM_ARITHMETIC_IR_VISITOR_WITH_ANALYZER_H_
