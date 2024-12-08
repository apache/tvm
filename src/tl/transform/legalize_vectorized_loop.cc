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
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <queue>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../op/parallel.h"
#include "loop_partition.h"
#include "loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

// Class to legalize vectorized loops by transforming them appropriately
class LoopVectorizedLegalizer : IRMutatorWithAnalyzer {
 public:
  // Static method to substitute and transform the given PrimFunc
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    // Create an instance of the legalizer with the analyzer
    LoopVectorizedLegalizer substituter(&analyzer);
    // Get a mutable copy of the function node
    PrimFuncNode* fptr = f.CopyOnWrite();
    // Apply the legalizer to the function body
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

 private:
  // Constructor initializing the base class with the analyzer
  LoopVectorizedLegalizer(arith::Analyzer* analyzer) : arith::IRMutatorWithAnalyzer(analyzer) {}

  // Override the VisitStmt_ method to handle ForNode (loop statements)
  Stmt VisitStmt_(const ForNode* op) final {
    // Visit and potentially modify the loop node
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    // If the loop is not vectorized, proceed with the default behavior
    if (for_node->kind != ForKind::kVectorized) {
      return IRMutatorWithAnalyzer::VisitStmt_(op);
    }
    // Change the loop kind from vectorized to serial
    for_node.CopyOnWrite()->kind = ForKind::kSerial;
    // Apply vectorization transformation to the loop
    return VectorizeLoop(std::move(for_node));
  }
};

// Create a pass that legalizes vectorized loops in the IRModule
tvm::transform::Pass LegalizeVectorizedLoop() {
  using namespace tir::transform;
  // Define the transformation function to be applied
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LoopVectorizedLegalizer::Substitute(std::move(f));
  };
  // Create and return a PrimFunc pass with the transformation function
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeVectorizedLoop", {});
}

// Register the pass globally so it can be used in the compilation pipeline
TVM_REGISTER_GLOBAL("tl.LegalizeVectorizedLoop").set_body_typed(LegalizeVectorizedLoop);

}  // namespace tl
}  // namespace tvm
