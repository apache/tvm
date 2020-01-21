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
 * \file src/tvm/ir/expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
#include <tvm/runtime/registry.h>
#include <tvm/ir/expr.h>
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

PrimExpr::PrimExpr(std::string str)
    : PrimExpr(tir::StringImmNode::make(str)) {}

PrimExpr PrimExpr::FromObject_(ObjectPtr<Object> ptr) {
  using runtime::ObjectTypeChecker;
  if (ptr->IsInstance<tir::IterVarNode>()) {
    return tir::IterVar(ptr)->var;
  }
  if (ptr->IsInstance<te::TensorNode>()) {
    return te::Tensor(ptr)();
  }
  CHECK(ObjectTypeChecker<PrimExpr>::Check(ptr.get()))
      << "Expect type " << ObjectTypeChecker<PrimExpr>::TypeName()
      << " but get " << ptr->GetTypeKey();
  return PrimExpr(ptr);
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

TVM_REGISTER_GLOBAL("make.IntImm")
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

TVM_REGISTER_GLOBAL("make.FloatImm")
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RangeNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const RangeNode*>(node.get());
    p->stream << "range(min=" << op->min << ", ext=" << op->extent << ')';
  });

TVM_REGISTER_NODE_TYPE(ArrayNode);
TVM_REGISTER_NODE_TYPE(MapNode);
TVM_REGISTER_NODE_TYPE(StrMapNode);
TVM_REGISTER_NODE_TYPE(RangeNode);


GlobalVar::GlobalVar(std::string name_hint) {
  ObjectPtr<GlobalVarNode> n = make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(GlobalVarNode);

TVM_REGISTER_GLOBAL("relay._make.GlobalVar")
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
