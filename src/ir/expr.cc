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
#include <tvm/arith/analyzer.h>
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

#include "../support/scalars.h"

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
  if (const auto* buffer_region = ref.as<tir::BufferRegionNode>()) {
    Array<PrimExpr> indices;
    indices.reserve(buffer_region->region.size());
    for (const Range& r : buffer_region->region) {
      if (tvm::tir::is_one(r->extent)) {
        indices.push_back(r->min);
      } else if (const auto* extent = r->extent.as<IntImmNode>()) {
        indices.push_back(tir::Ramp(r->min, tvm::tir::make_const(r->min->dtype, 1), extent->value));
      } else {
        LOG(FATAL) << "ValueError: Cannot convert to BufferLoad: " << ref;
      }
    }
    return tir::BufferLoad(buffer_region->buffer, indices);
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
    ICHECK_GE(value, 0U) << "ValueError: Literal value " << value
                         << " is negative for unsigned integer type " << dtype;
    if (dtype.bits() < 64) {
      ICHECK_LT(value, 1LL << dtype.bits())
          << "ValueError: Literal value " << value << " exceeds maximum of " << dtype;
    }
  } else if (dtype.bits() == 1) {
    // int(1)
    ICHECK(value == 0 || value == 1) << "ValueError: " << value << " exceeds range of " << dtype;
  } else if (dtype.bits() < 64) {
    ICHECK_GE(value, -(1LL << (dtype.bits() - 1)))
        << "ValueError: Literal value " << value << " exceeds minimum of " << dtype;
    ICHECK_LT(value, 1LL << (dtype.bits() - 1))
        << "ValueError: Literal value " << value << " exceeds maximum of " << dtype;
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

  ICHECK(dtype.is_float() || dtype.is_bfloat16() || dtype.code() >= DataType::kCustomBegin)
      << "ValueError: FloatImm supports only float, but " << dtype << " was supplied.";

  // check range for float32 and float16 since they have specified range.
  if (!std::isinf(value) && !std::isnan(value)) {
    if (dtype.bits() == 32) {
      ICHECK_GE(value, std::numeric_limits<float>::lowest())
          << "ValueError: Literal value " << value << " exceeds minimum of " << dtype;
      ICHECK_LE(value, std::numeric_limits<float>::max())
          << "ValueError: Literal value " << value << " exceeds maximum of " << dtype;
    } else if (dtype.is_float16()) {
      ICHECK_GE(value, -support::kMaxFloat16)
          << "ValueError: Literal value " << value << " exceeds minimum of " << dtype;
      ICHECK_LE(value, support::kMaxFloat16)
          << "ValueError: Literal value " << value << " exceeds maximum of " << dtype;
    }
  }
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

GlobalVar::GlobalVar(String name_hint, Type type, Span span) {
  ObjectPtr<GlobalVarNode> n = make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  n->checked_type_ = std::move(type);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(GlobalVarNode);

TVM_REGISTER_GLOBAL("ir.GlobalVar").set_body_typed([](String name, Type type) {
  return GlobalVar(name, type);
});

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
