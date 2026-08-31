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
 * \file reduce.cc
 * \brief TE reduction expression definitions.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_map>

namespace tvm {
namespace te {
using namespace tvm::tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  CommReducerNode::RegisterReflection();
  ReduceNode::RegisterReflection();
}

// CommReducer
CommReducer::CommReducer(ffi::Array<PrimVar> lhs, ffi::Array<PrimVar> rhs,
                         ffi::Array<PrimExpr> result, ffi::Array<PrimExpr> identity_element,
                         Span span) {
  size_t n_group = result.size();
  TVM_FFI_CHECK_EQ(lhs.size(), n_group, ValueError)
      << "The number of vars in `lhs` must equal to the "
         "number of elements in `results`";
  TVM_FFI_CHECK_EQ(rhs.size(), n_group, ValueError)
      << "The number of vars in `rhs` must equal to the "
         "number of elements in `results`";
  TVM_FFI_CHECK_EQ(identity_element.size(), n_group, ValueError)
      << "The number of identities must equal to the number of elements in `results`";

  // Change the dtype of input vars to adapt to the dtype of identities
  ffi::ArrayObj* p_lhs = lhs.CopyOnWrite();
  ffi::ArrayObj* p_rhs = rhs.CopyOnWrite();
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  var_map.reserve(n_group * 2);
  for (int i = 0; i < static_cast<int>(n_group); ++i) {
    PrimType dtype = identity_element[i].ty();
    PrimVar l = lhs[i].CopyWithDType(dtype);
    PrimVar r = rhs[i].CopyWithDType(dtype);
    var_map[lhs[i].get()] = l;
    var_map[rhs[i].get()] = r;

    p_lhs->SetItem(i, l);
    p_rhs->SetItem(i, r);
  }

  ffi::ArrayObj* p_result = result.CopyOnWrite();
  for (int i = 0; i < static_cast<int>(n_group); ++i) {
    p_result->SetItem(i, Substitute(result[i], var_map));
  }

  auto node = ffi::make_object<CommReducerNode>();
  node->lhs = lhs;
  node->rhs = rhs;
  node->result = result;
  node->identity_element = identity_element;
  node->span = std::move(span);
  data_ = std::move(node);
}

ffi::Array<PrimExpr> CommReducerNode::operator()(ffi::Array<PrimExpr> a,
                                                 ffi::Array<PrimExpr> b) const {
  TVM_FFI_ICHECK_EQ(a.size(), b.size());
  TVM_FFI_ICHECK_EQ(lhs.size(), a.size());
  TVM_FFI_ICHECK_EQ(rhs.size(), b.size());
  ffi::Map<Var, PrimExpr> value_map;
  for (size_t i = 0; i < a.size(); ++i) {
    value_map.Set(lhs[i], a[i]);
    value_map.Set(rhs[i], b[i]);
  }
  return Substitute(this->result, value_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("te.CommReducer",
           [](ffi::Array<PrimVar> lhs, ffi::Array<PrimVar> rhs, ffi::Array<PrimExpr> result,
              ffi::Array<PrimExpr> identity_element,
              Span span) { return CommReducer(lhs, rhs, result, identity_element, span); })
      .def_method("te.CommReducerCombine", &te::CommReducerNode::operator());
}

// Reduce
Reduce::Reduce(CommReducer combiner, ffi::Array<PrimExpr> source, ffi::Array<IterVar> axis,
               PrimExpr condition, int value_index, ffi::Array<PrimExpr> init, Span span) {
  for (size_t i = 0; i < axis.size(); ++i) {
    TVM_FFI_ICHECK_EQ(axis[i]->iter_type, kCommReduce)
        << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = IntImm::Bool(true);
  }
  auto n = ffi::make_object<ReduceNode>();
  TVM_FFI_ICHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    TVM_FFI_ICHECK(axis[i].defined());
  }
  if (!init.empty()) {
    TVM_FFI_ICHECK_EQ(init.size(), source.size()) << "Number of inits should match number of exprs";
    for (size_t i = 0; i < init.size(); i++) {
      TVM_FFI_ICHECK(init[i].defined()) << "Init value must be defined";
      if (te::IsTensorLoad(init[i])) {
        te::GetTensorFromLoad(init[i].as_or_throw<Call>());
      } else {
        TVM_FFI_ICHECK(init[i]->IsInstance<IntImmNode>() || init[i]->IsInstance<FloatImmNode>())
            << "init can only be an IntImm, FloatImm or Tensor-load Call, "
            << "but received " << init[i] << " of type " << init[i]->GetTypeKey();
      }
    }
  }
  n->ExprNode::ty = source[value_index].ty();
  n->combiner = std::move(combiner);
  n->source = std::move(source);
  n->init = std::move(init);
  n->axis = std::move(axis);
  n->condition = condition;
  n->value_index = value_index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "te.Reduce", [](CommReducer combiner, ffi::Array<PrimExpr> source, ffi::Array<IterVar> axis,
                      PrimExpr condition, int value_index, ffi::Array<PrimExpr> init, Span span) {
        return Reduce(combiner, source, axis, condition, value_index, init, span);
      });
}

}  // namespace te
}  // namespace tvm
