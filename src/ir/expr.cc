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
#include <tvm/runtime/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
// NOTE: reverse dependency on top/tir.
// These dependencies do not happen at the interface-level,
// and are only used in minimum cases where they are clearly marked.
//
// Rationale: convert from IterVar and top::Tensor
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

namespace tvm {

PrimExpr::PrimExpr(int32_t value)
    : PrimExpr(IntImm(DataType::Int(32), value)) {}

PrimExpr::PrimExpr(float value)
    : PrimExpr(FloatImm(DataType::Float(32), value)) {}

PrimExpr PrimExpr::FromObject_(ObjectRef ref) {
  using runtime::ObjectTypeChecker;
  if (auto* ptr = ref.as<tir::IterVarNode>()) {
    return GetRef<tir::IterVar>(ptr)->var;
  }
  if (auto* ptr = ref.as<te::TensorNode>()) {
    return GetRef<te::Tensor>(ptr)();
  }
  if (auto* ptr = ref.as<runtime::StringObj>()) {
    return tir::StringImmNode::make(GetRef<runtime::String>(ptr));
  }
  CHECK(ObjectTypeChecker<PrimExpr>::Check(ref.get()))
      << "Expect type " << ObjectTypeChecker<PrimExpr>::TypeName()
      << " but get " << ref->GetTypeKey();
  return Downcast<PrimExpr>(ref);
}


IntImm::IntImm(DataType dtype, int64_t value) {
  CHECK(dtype.is_scalar())
      << "ValueError: IntImm can only take scalar.";
  CHECK(dtype.is_int() || dtype.is_uint())
      << "ValueError: IntImm can only take scalar.";
  if (dtype.is_uint()) {
    CHECK_GE(value, 0U);
  }
  ObjectPtr<IntImmNode> node = make_object<IntImmNode>();
  node->dtype = dtype;
  node->value = value;
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("ir.IntImm")
.set_body_typed([](DataType dtype, int64_t value) {
  return IntImm(dtype, value);
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

FloatImm::FloatImm(DataType dtype, double value) {
  CHECK_EQ(dtype.lanes(), 1)
      << "ValueError: FloatImm can only take scalar.";
  ObjectPtr<FloatImmNode> node = make_object<FloatImmNode>();
  node->dtype = dtype;
  node->value = value;
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("ir.FloatImm")
.set_body_typed([](DataType dtype, double value) {
  return FloatImm(dtype, value);
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


Range::Range(PrimExpr begin, PrimExpr end)
    : Range(make_object<RangeNode>(
          begin,
          tir::is_zero(begin) ? end : (end - begin))) {
}

Range Range::make_by_min_extent(PrimExpr min, PrimExpr extent) {
  return Range(make_object<RangeNode>(min, extent));
}

TVM_REGISTER_GLOBAL("ir.range_by_min_extent")
.set_body_typed(Range::make_by_min_extent);

TVM_REGISTER_GLOBAL("ir.Range")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
  *ret = Range(args[0], args[1]);
  });

TVM_REGISTER_NODE_TYPE(RangeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RangeNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const RangeNode*>(node.get());
    p->stream << "range(min=" << op->min << ", ext=" << op->extent << ')';
  });


GlobalVar::GlobalVar(std::string name_hint) {
  ObjectPtr<GlobalVarNode> n = make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(GlobalVarNode);

TVM_REGISTER_GLOBAL("ir.GlobalVar")
.set_body_typed([](std::string name){
  return GlobalVar(name);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<GlobalVarNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const GlobalVarNode*>(ref.get());
    p->stream << "GlobalVar(" << node->name_hint << ")";
  });

// Container printer
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ArrayNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const ArrayNode*>(node.get());
    p->stream << '[';
    for (size_t i = 0 ; i < op->data.size(); ++i) {
      if (i != 0) {
        p->stream << ", ";
      }
      p->Print(op->data[i]);
    }
    p->stream << ']';
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<MapNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const MapNode*>(node.get());
    p->stream << '{';
    for (auto it = op->data.begin(); it != op->data.end(); ++it) {
      if (it != op->data.begin()) {
        p->stream << ", ";
      }
      p->Print(it->first);
      p->stream << ": ";
      p->Print(it->second);
    }
    p->stream << '}';
  });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<StrMapNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const StrMapNode*>(node.get());
    p->stream << '{';
    for (auto it = op->data.begin(); it != op->data.end(); ++it) {
      if (it != op->data.begin()) {
        p->stream << ", ";
      }
      p->stream << '\"' << it->first << "\": ";
      p->Print(it->second);
    }
    p->stream << '}';
  });
}  // namespace tvm
