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
 * \file tvm/src/pass/lower_custom_datatypes.cc
 * \brief Pass for lowering custom datatypes
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/ir_pass.h>
#include "../../target/datatype/registry.h"

namespace tvm {
namespace tir {

/*!
 * \brief Helper mutator to implement lowering of custom datatypes.
 *
 * Lowering datatypes works as follows: for every expression containing a custom
 * datatype, we search for a global (registered by the implementer of the custom
 * datatype) for lowering this type of expression, and uses it to lower the
 * expression.
 */
class CustomDatatypesLowerer : public StmtExprMutator {
 public:
  explicit CustomDatatypesLowerer(const std::string& target) : target_(target) {}

  inline PrimExpr VisitExpr_(const CastNode* op) final {
    auto type_code = op->dtype.code();
    auto src_type_code = op->value.dtype().code();
    // If either datatype is a registered custom datatype, we must lower.
    bool toBeLowered = datatype::Registry::Global()->GetTypeRegistered(type_code) ||
                       datatype::Registry::Global()->GetTypeRegistered(src_type_code);
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<CastNode>();
    if (toBeLowered) {
      auto lower = datatype::GetCastLowerFunc(target_, type_code, src_type_code);
      CHECK(lower) << "Cast lowering function for target " << target_ << " destination type "
                   << static_cast<unsigned>(type_code) << " source type "
                   << static_cast<unsigned>(src_type_code) << " not found";
      return (*lower)(expr);
    }
    return expr;
  }

  inline PrimExpr VisitExpr_(const FloatImmNode* imm) final {
    auto type_code = imm->dtype.code();
    auto e = GetRef<PrimExpr>(imm);
    if (datatype::Registry::Global()->GetTypeRegistered(type_code)) {
      auto lower = datatype::GetFloatImmLowerFunc(target_, type_code);
      CHECK(lower) << "FloatImm lowering function for target " << target_ << " type "
                   << static_cast<unsigned>(type_code) << " not found";
      return (*lower)(e);
    }
    return e;
  }

  inline Stmt VisitStmt_(const AllocateNode* allocate) final {
    bool toBeLowered = datatype::Registry::Global()->GetTypeRegistered(allocate->dtype.code());
    Stmt stmt = StmtExprMutator::VisitStmt_(allocate);
    allocate = stmt.as<AllocateNode>();

    if (toBeLowered) {
      auto new_allocate_type = DataType::UInt(allocate->dtype.bits(), allocate->dtype.lanes());
      return AllocateNode::make(allocate->buffer_var, new_allocate_type, allocate->extents,
                            allocate->condition, allocate->body, allocate->new_expr,
                            allocate->free_function);
    }
    return stmt;
  }

  inline PrimExpr VisitExpr_(const LoadNode* load) final {
    bool toBeLowered = datatype::Registry::Global()->GetTypeRegistered(load->dtype.code());
    PrimExpr expr = StmtExprMutator::VisitExpr_(load);
    load = expr.as<LoadNode>();
    if (toBeLowered) {
      auto new_load_type = DataType::UInt(load->dtype.bits());
      return LoadNode::make(new_load_type, load->buffer_var, load->index, load->predicate);
    }
    return expr;
  }

#define DEFINE_MUTATE__(OP, NodeName)                                              \
  inline PrimExpr VisitExpr_(const NodeName* op) final {                                     \
    auto type_code = op->dtype.code();                                             \
    bool toBeLowered = datatype::Registry::Global()->GetTypeRegistered(type_code); \
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);                                   \
    op = expr.as<NodeName>();                                                            \
    if (toBeLowered) {                                                             \
      auto lower = datatype::Get##OP##LowerFunc(target_, type_code);               \
      CHECK(lower) << #OP " lowering function for target " << target_ << " type "  \
                   << static_cast<unsigned>(type_code) << " not found";            \
      return (*lower)(expr);                                                       \
    }                                                                              \
    return expr;                                                                   \
  }

  DEFINE_MUTATE__(Add, AddNode);
  DEFINE_MUTATE__(Sub, SubNode);
  DEFINE_MUTATE__(Mul, MulNode);
  DEFINE_MUTATE__(Div, DivNode);
  DEFINE_MUTATE__(Mod, ModNode);
  DEFINE_MUTATE__(Min, MinNode);
  DEFINE_MUTATE__(Max, MaxNode);
  DEFINE_MUTATE__(EQ, EQNode);
  DEFINE_MUTATE__(NE, NENode);
  DEFINE_MUTATE__(LT, LTNode);
  DEFINE_MUTATE__(LE, LENode);
  DEFINE_MUTATE__(GT, GTNode);
  DEFINE_MUTATE__(GE, GENode);
  // Later changes may need to add more mutate functions as we support workloads with more ops.

 private:
  std::string target_;
};

LoweredFunc LowerCustomDatatypes(LoweredFunc f, const std::string& target) {
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = CustomDatatypesLowerer(target)(n->body);
  return LoweredFunc(n);
}

}  // namespace tir
}  // namespace tvm
