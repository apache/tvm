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
 * \file tir/transforms/convert_for_loops_serial.cc
 * \brief Convert all for loops to serial for lesser memory consumption
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class ForLoopSerialConverter : public StmtExprMutator {
 public:
  ForLoopSerialConverter() = default;
  Stmt operator()(const PrimFunc& func);

 private:
  Stmt VisitStmt_(const ForNode* op) override;
};

Stmt ForLoopSerialConverter::VisitStmt_(const ForNode* op) {
  if (op->kind == ForKind::kParallel) {
    return For(op->loop_var, op->min, op->extent, ForKind::kSerial, op->body, op->thread_binding,
               op->annotations, op->span);
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt ForLoopSerialConverter::operator()(const PrimFunc& func) {
  return this->VisitStmt(func->body);
}

PrimFunc ConvertForLoopsToSerial(PrimFunc func) {
  PrimFuncNode* fptr = func.CopyOnWrite();
  fptr->body = ForLoopSerialConverter()(func);
  return func;
}

namespace transform {

Pass ConvertForLoopsToSerial() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return ConvertForLoopsToSerial(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ConvertForLoopsToSerial", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ConvertForLoopsToSerial")
    .set_body_typed(ConvertForLoopsToSerial);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
