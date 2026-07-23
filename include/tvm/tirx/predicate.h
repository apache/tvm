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
 *//*!
 * \file tvm/tir/predicate.h
 * \brief Definition of predicate
 */

#ifndef TVM_TIRX_PREDICATE_H_
#define TVM_TIRX_PREDICATE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/object.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/var.h>
namespace tvm {
namespace tirx {

class PredicateNode : public ffi::Object {
 public:
  /*! \brief The variables in the predicate */
  Array<PrimVar> vars;
  /*! \brief The predicate */
  PrimExpr pred;

  /*! \brief Replace the variables in the predicate with the given indices */
  PrimExpr Apply(const Array<PrimExpr>& indices) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PredicateNode>()
        .def_ro("vars", &PredicateNode::vars, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("pred", &PredicateNode::pred);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Predicate", PredicateNode, ffi::Object);
};

class Predicate : public ffi::ObjectRef {
 public:
  explicit Predicate(Array<PrimVar> vars, PrimExpr pred);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Predicate, ffi::ObjectRef, PredicateNode);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_PREDICATE_H_
