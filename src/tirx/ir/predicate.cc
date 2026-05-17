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
 * \file predicate.cc
 */

#include "tvm/tirx/predicate.h"

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() { PredicateNode::RegisterReflection(); }

PrimExpr PredicateNode::Apply(const ffi::Array<PrimExpr>& indices) const {
  TVM_FFI_ICHECK_EQ(indices.size(), vars.size());

  ffi::Map<Var, PrimExpr> vmap;

  for (size_t i = 0; i < vars.size(); i++) {
    vmap.Set(vars[i], indices[i]);
  }

  return SubstituteWithDataTypeLegalization(std::move(pred),
                                            [&](const Var& var) { return vmap.Get(var); });
}

Predicate::Predicate(ffi::Array<Var> vars, PrimExpr pred) {
  auto n = ffi::make_object<PredicateNode>();
  n->vars = std::move(vars);
  n->pred = std::move(pred);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Predicate",
                        [](ffi::Array<Var> vars, PrimExpr pred) { return Predicate(vars, pred); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.PredicateApply", [](Predicate pred, ffi::Array<PrimExpr> indices) {
    return pred->Apply(indices);
  });
}

}  // namespace tirx
}  // namespace tvm
