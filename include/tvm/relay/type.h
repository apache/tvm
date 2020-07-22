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
 * \file tvm/relay/type.h
 * \brief Relay typed AST nodes.
 */
#ifndef TVM_RELAY_TYPE_H_
#define TVM_RELAY_TYPE_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/ir/tensor_type.h>
#include <tvm/ir/type.h>
#include <tvm/ir/type_relation.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>

#include <string>

#include "base.h"

namespace tvm {
namespace relay {

// namespace update for backward compact
// will be removed later.
using AnyNode = tvm::tir::AnyNode;
using Any = tvm::tir::Any;
using Kind = TypeKind;
using Type = tvm::Type;
using TypeNode = tvm::TypeNode;
using TypeVar = tvm::TypeVar;
using TypeVarNode = tvm::TypeVarNode;
using GlobalTypeVar = tvm::GlobalTypeVar;
using GlobalTypeVarNode = tvm::GlobalTypeVarNode;
using TupleType = tvm::TupleType;
using TupleTypeNode = tvm::TupleTypeNode;
using TypeConstraint = tvm::TypeConstraint;
using TypeConstraintNode = tvm::TypeConstraintNode;
using FuncType = tvm::FuncType;
using FuncTypeNode = tvm::FuncTypeNode;
using IncompleteType = tvm::IncompleteType;
using IncompleteTypeNode = tvm::IncompleteTypeNode;
using RelayRefType = tvm::RelayRefType;
using RelayRefTypeNode = tvm::RelayRefTypeNode;
using TensorType = tvm::TensorType;
using TensorTypeNode = tvm::TensorTypeNode;
using TypeCall = tvm::TypeCall;
using TypeCallNode = tvm::TypeCallNode;
using TypeRelation = tvm::TypeRelation;
using TypeRelationNode = tvm::TypeRelationNode;
using TypeRelationFn = tvm::TypeRelationFn;
using TypeReporter = tvm::TypeReporter;
using TypeReporterNode = tvm::TypeReporterNode;

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TYPE_H_
