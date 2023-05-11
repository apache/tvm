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
 * \file src/relay/collage/utils.cc
 * \brief Misc helpers.
 */

#include "./utils.h"

#include "../../support/scalars.h"
#include "../op/memory/device_copy.h"

namespace tvm {
namespace relay {
namespace collage {

String GetSpecName(const Target& target) {
  if (target.IsExternalCodegen()) {
    return target->kind->name;
  } else {
    return std::string(kTVMSpecNamePrefix) + target->kind->name;
  }
}

String UnionLabels(String left, String right) {
  if (left.empty()) {
    return right;
  }
  if (right.empty()) {
    return left;
  }
  return left + "+" + right;
}

String NestLabels(String left, String right) {
  if (left.empty()) {
    return right;
  }
  if (right.empty()) {
    return left;
  }
  if (right.size() > left.size()) {
    std::string right_str = right;
    if (right_str.substr(0, left.size()) == left) {
      return right;
    }
  }
  return left + "." + right;
}

std::string KindToString(OpPatternKind kind) {
  switch (kind) {
    case kElemWise:
      return "E";
    case kBroadcast:
      return "B";
    case kInjective:
      return "I";
    case kCommReduce:
      return "R";
    case kOutEWiseFusable:
      return "A";
    case kTuple:
      return "T";
    case kOpaque:
      return "O";
  }
  return "?";
}

OpPatternKind CombineKinds(OpPatternKind left, OpPatternKind right) {
  return std::max(left, right);
}

bool CanInline(const Expr& expr) {
  if (expr.as<OpNode>() || expr.as<ConstructorNode>() || expr.as<FunctionNode>()) {
    return true;
  }
  if (const auto* constant_node = expr.as<ConstantNode>()) {
    return support::IsSimpleScalar(constant_node);
  }
  return false;
}

bool IsSpecialOp(const OpNode* op_node) {
  auto op = GetRef<Op>(op_node);
  static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");
  if (fnoncomputational.count(op) && fnoncomputational[op]) {
    // Operator has been marked as non-computational.
    return true;
  }
  // TODO(mbs): This is incomplete.
  static auto shape_of_op_ = Op::Get("shape_of");
  static auto vm_shape_of_op_ = Op::Get("vm.shape_of");
  if (op == DeviceCopyOp() || op == shape_of_op_ || op == vm_shape_of_op_) {
    // Operator is compiled away by the VM compilation flow.
    return true;
  }
  return false;
}

bool MustBeLowered(const Expr& expr) {
  if (const auto* call_node = expr.as<CallNode>()) {
    if (const auto* function_node = call_node->op.as<FunctionNode>()) {
      if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
        // We've already committed to this call being to one or more operators which must be
        // lowered.
        return true;
      }
    } else if (const auto* op_node = call_node->op.as<OpNode>()) {
      if (!IsSpecialOp(op_node)) {
        // The VM compilation path won't rewrite this call.
        return true;
      }
    }
  }
  return false;
}

std::vector<std::string> SplitString(std::string stmt, const char* del) {
  std::vector<std::string> str_tokens;
  int start = 0;
  int end = stmt.find(del, 0);
  str_tokens.emplace_back(stmt.substr(start, end));
  while (end != -1) {
    stmt = stmt.substr(end + 1, stmt.size());
    end = stmt.find(del, 0);
    str_tokens.emplace_back(stmt.substr(start, end));
  }
  return str_tokens;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
