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
 * \file src/ir/expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/registry.h>
// NOTE: reverse dependency on top/tir.
// These dependencies do not happen at the interface-level,
// and are only used in minimum cases where they are clearly marked.
//
// Rationale: convert from IterVar and top::Tensor
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

namespace tvm {

PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm(DataType::Int(32), value)) {}

PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(DataType::Float(32), value)) {}

PrimExpr PrimExpr::FromObject_(ObjectRef ref) {
  using runtime::ObjectTypeChecker;
  if (auto* ptr = ref.as<tir::IterVarNode>()) {
    return GetRef<tir::IterVar>(ptr)->var;
  }
  if (auto* ptr = ref.as<te::TensorNode>()) {
    return GetRef<te::Tensor>(ptr)();
  }
  if (auto* ptr = ref.as<runtime::StringObj>()) {
    return tir::StringImm(GetRef<runtime::String>(ptr));
  }
  Optional<String> actual_type = ObjectTypeChecker<PrimExpr>::CheckAndGetMismatch(ref.get());
  ICHECK(!actual_type.defined()) << "Expected type " << ObjectTypeChecker<PrimExpr>::TypeName()
                                 << " but got " << actual_type.value();
  return Downcast<PrimExpr>(ref);
}

IntImm::IntImm(DataType dtype, int64_t value, Span span) {
  ICHECK(dtype.is_scalar()) << "ValueError: IntImm can only take scalar, but " << dtype
                            << " was supplied.";
  ICHECK(dtype.is_int() || dtype.is_uint())
      << "ValueError: IntImm supports only int or uint type, but " << dtype << " was supplied.";
  if (dtype.is_uint()) {
    ICHECK_GE(value, 0U);
  }
  ObjectPtr<IntImmNode> node = make_object<IntImmNode>();
  node->dtype = dtype;
  node->value = value;
  node->span = span;
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("ir.IntImm").set_body_typed([](DataType dtype, int64_t value, Span span) {
  return IntImm(dtype, value, span);
});

TVM_REGISTER_NODE_TYPE(IntImmNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IntImmNode*>(node.get());
      if (op->dtype == DataType::Int(32)) {
        p->stream << op->value;
      } else {
        p->stream << "(" << op->dtype << ")" << op->value;
      }
    });

FloatImm::FloatImm(DataType dtype, double value, Span span) {
  ICHECK_EQ(dtype.lanes(), 1) << "ValueError: FloatImm can only take scalar.";
  ObjectPtr<FloatImmNode> node = make_object<FloatImmNode>();
  node->dtype = dtype;
  node->value = value;
  node->span = span;
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("ir.FloatImm").set_body_typed([](DataType dtype, double value, Span span) {
  return FloatImm(dtype, value, span);
});

TVM_REGISTER_NODE_TYPE(FloatImmNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FloatImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FloatImmNode*>(node.get());
      auto& stream = p->stream;
      switch (op->dtype.bits()) {
        case 64:
          stream << op->value;
          break;
        case 32:
          stream << op->value << 'f';
          break;
        case 16:
          stream << op->value << 'h';
          break;
        default:
          LOG(FATAL) << "Unknown float type bits=" << op->dtype.bits();
      }
    });

Range::Range(PrimExpr begin, PrimExpr end, Span span)
    : Range(make_object<RangeNode>(begin, tir::is_zero(begin) ? end : (end - begin), span)) {}

Range Range::FromMinExtent(PrimExpr min, PrimExpr extent, Span span) {
  return Range(make_object<RangeNode>(min, extent, span));
}

TVM_REGISTER_GLOBAL("ir.Range_from_min_extent").set_body_typed(Range::FromMinExtent);

TVM_REGISTER_GLOBAL("ir.Range").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = Range(args[0], args[1], args[2]);
});

TVM_REGISTER_NODE_TYPE(RangeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RangeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RangeNode*>(node.get());
      p->stream << "range(min=" << op->min << ", ext=" << op->extent << ')';
    });

GlobalVar::GlobalVar(String name_hint) {
  ObjectPtr<GlobalVarNode> n = make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(GlobalVarNode);

TVM_REGISTER_GLOBAL("ir.GlobalVar").set_body_typed([](String name) { return GlobalVar(name); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<GlobalVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const GlobalVarNode*>(ref.get());
      p->stream << "GlobalVar(" << node->name_hint << ")";
    });

TVM_REGISTER_GLOBAL("ir.DebugPrint").set_body_typed([](ObjectRef ref) {
  std::stringstream ss;
  ss << ref;
  return ss.str();
});

}  // namespace tvm
