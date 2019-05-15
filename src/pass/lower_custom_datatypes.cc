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
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/pass/lower_custom_datatypes.cc
 * \brief Pass for lowering custom datatypes
 */

#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/packed_func_ext.h>
#include "../codegen/datatype/registry.h"

namespace tvm {
namespace ir {

/*!
 * \brief Helper mutator to implement lowering of custom datatypes.
 *
 * Lowering datatypes works as follows: for every expression containing a custom
 * datatype, we search for a global (registered by the implementer of the custom
 * datatype) for lowering this type of expression, and uses it to lower the
 * expression.
 */
class CustomDatatypesLowerer : public IRMutator {
 public:
  explicit CustomDatatypesLowerer(const std::string& target) : target_(target) {}

  inline Expr Mutate_(const Cast* op, const Expr& e) final {
    auto type_code = op->type.code();
    auto src_type_code = op->value.type().code();
    // If either datatype is a registered custom datatype, we must lower.
    bool toBeLowered = datatype::Registry::Global()->GetTypeRegistered(type_code) ||
                       datatype::Registry::Global()->GetTypeRegistered(src_type_code);
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Cast>();
    if (toBeLowered) {
      auto lower = datatype::GetCastLowerFunc(target_, type_code, src_type_code);
      CHECK(lower) << "Cast lowering function for target " << target_ << " destination type "
                   << static_cast<unsigned>(type_code) << " source type "
                   << static_cast<unsigned>(src_type_code) << " not found";
      return (*lower)(expr);
    }
    return expr;
  }

  inline Expr Mutate_(const FloatImm* imm, const Expr& e) final {
    auto type_code = imm->type.code();
    if (datatype::Registry::Global()->GetTypeRegistered(type_code)) {
      auto lower = datatype::GetFloatImmLowerFunc(target_, type_code);
      CHECK(lower) << "FloatImm lowering function for target " << target_ << " type "
                   << static_cast<unsigned>(type_code) << " not found";
      return (*lower)(e);
    }
    return e;
  }

  inline Stmt Mutate_(const Allocate* allocate, const Stmt& s) final {
    bool toBeLowered = datatype::Registry::Global()->GetTypeRegistered(allocate->type.code());
    Stmt stmt = IRMutator::Mutate_(allocate, s);
    allocate = stmt.as<Allocate>();

    if (toBeLowered) {
      auto new_allocate_type = UInt(allocate->type.bits(), allocate->type.lanes());
      return Allocate::make(allocate->buffer_var, new_allocate_type, allocate->extents,
                            allocate->condition, allocate->body, allocate->new_expr,
                            allocate->free_function);
    }
    return stmt;
  }

  inline Expr Mutate_(const Load* load, const Expr& e) final {
    bool toBeLowered = datatype::Registry::Global()->GetTypeRegistered(load->type.code());
    Expr expr = IRMutator::Mutate_(load, e);
    load = expr.as<Load>();
    if (toBeLowered) {
      auto new_load_type = UInt(load->type.bits());
      return Load::make(new_load_type, load->buffer_var, load->index, load->predicate);
    }
    return expr;
  }

#define DEFINE_MUTATE__(OP)                                                        \
  inline Expr Mutate_(const OP* op, const Expr& e) final {                         \
    auto type_code = op->type.code();                                              \
    bool toBeLowered = datatype::Registry::Global()->GetTypeRegistered(type_code); \
    Expr expr = IRMutator::Mutate_(op, e);                                         \
    op = expr.as<OP>();                                                            \
    if (toBeLowered) {                                                             \
      auto lower = datatype::Get##OP##LowerFunc(target_, type_code);               \
      CHECK(lower) << #OP " lowering function for target " << target_ << " type "  \
                   << static_cast<unsigned>(type_code) << " not found";            \
      return (*lower)(expr);                                                       \
    }                                                                              \
    return expr;                                                                   \
  }

  DEFINE_MUTATE__(Add)
  DEFINE_MUTATE__(Sub)
  DEFINE_MUTATE__(Mul)
  DEFINE_MUTATE__(Div)
  DEFINE_MUTATE__(Mod)
  DEFINE_MUTATE__(Min)
  DEFINE_MUTATE__(Max)
  DEFINE_MUTATE__(EQ)
  DEFINE_MUTATE__(NE)
  DEFINE_MUTATE__(LT)
  DEFINE_MUTATE__(LE)
  DEFINE_MUTATE__(GT)
  DEFINE_MUTATE__(GE)
  // Later changes may need to add more mutate functions as we support workloads with more ops.

 private:
  std::string target_;
};

LoweredFunc LowerCustomDatatypes(LoweredFunc f, const std::string& target) {
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  n->body = CustomDatatypesLowerer(target).Mutate(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
