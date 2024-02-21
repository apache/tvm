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
 * \file src/relax/transform/annotate_tir_op_pattern.cc
 * \brief Annotate Op Pattern for TIR functions. It is a pass works on TIR PrimFuncs,
 *        but they are needed for relax fusion. So we put them in the relax namespace.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace relax {

tir::PrimFunc AnnotateOpPattern(tir::PrimFunc f) {
  if (f->HasNonzeroAttr("op_pattern")) {
    return f;
  } else {
    relay::OpPatternKind kind = AnalyzeOpPatternKind(f);
    return WithAttr(std::move(f), "op_pattern", Integer(static_cast<int>(kind)));
  }
}

namespace transform {

Pass AnnotateTIROpPattern() {
  auto pass_func = [=](tir::PrimFunc f, IRModule m, PassContext ctx) {
    return AnnotateOpPattern(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "AnnotateTIROpPattern", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AnnotateTIROpPattern").set_body_typed(AnnotateTIROpPattern);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
