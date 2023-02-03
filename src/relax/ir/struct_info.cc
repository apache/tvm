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
 * \file src/relax/ir/struct_info.cc
 * \brief Relax struct info.
 */
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {

ObjectStructInfo::ObjectStructInfo(Span span) {
  ObjectPtr<ObjectStructInfoNode> n = make_object<ObjectStructInfoNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ObjectStructInfoNode);

TVM_REGISTER_GLOBAL("relax.ObjectStructInfo").set_body_typed([](Span span) {
  return ObjectStructInfo(span);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ObjectStructInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      p->stream << "ObjectStructInfo()";
    });

// Prim
PrimStructInfo::PrimStructInfo(DataType dtype, Span span) {
  ObjectPtr<PrimStructInfoNode> n = make_object<PrimStructInfoNode>();
  n->dtype = dtype;
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PrimStructInfoNode);

TVM_REGISTER_GLOBAL("relax.PrimStructInfo").set_body_typed([](DataType dtype, Span span) {
  return PrimStructInfo(dtype, span);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimStructInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = static_cast<const PrimStructInfoNode*>(ref.get());
      p->stream << "PrimStructInfo(" << node->dtype << ")";
    });

// Shape
ShapeStructInfo::ShapeStructInfo(Array<PrimExpr> values, Span span) {
  ObjectPtr<ShapeStructInfoNode> n = make_object<ShapeStructInfoNode>();
  n->ndim = static_cast<int>(values.size());
  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmNode>()) {
      return tvm::cast(DataType::Int(64), value);
    }
    ICHECK(value.dtype() == DataType::Int(64))
        << "the value in ShapeStructInfo can only have dtype of int64";
    return value;
  });
  n->span = span;
  data_ = std::move(n);
}

ShapeStructInfo::ShapeStructInfo(int ndim, Span span) {
  ObjectPtr<ShapeStructInfoNode> n = make_object<ShapeStructInfoNode>();
  CHECK_GE(ndim, -1) << "ndim of ShapeStructInfo must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ShapeStructInfoNode);

TVM_REGISTER_GLOBAL("relax.ShapeStructInfo")
    .set_body_typed([](Optional<Array<PrimExpr>> values, int ndim, Span span) {
      if (values.defined()) {
        CHECK_EQ(ndim, kUnknownNDim) << "ValueError: Cannot both specify values and ndim";
        return ShapeStructInfo(values.value(), span);
      } else {
        return ShapeStructInfo(ndim, span);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShapeStructInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = static_cast<const ShapeStructInfoNode*>(ref.get());
      if (node->values.defined()) {
        p->stream << "ShapeStructInfo(" << node->values.value() << ")";
      } else {
        p->stream << "ShapeStructInfo(ndim=" << node->ndim << ")";
      }
    });

// Tensor
TensorStructInfo::TensorStructInfo(Expr shape, DataType dtype, Span span) {
  ObjectPtr<TensorStructInfoNode> n = make_object<TensorStructInfoNode>();
  // assign ndim before move
  Optional<ShapeStructInfo> sinfo = MatchStructInfo<ShapeStructInfo>(shape);
  ICHECK(sinfo) << "We expect shape to contain pre-set shape struct info";
  ICHECK(shape.defined()) << "Must provide a shape in this constructor";
  ICHECK(shape->IsInstance<ShapeExprNode>() || shape->IsInstance<VarNode>())
      << "We require shape to be normalized when constructing TensorStructInfo";
  n->ndim = sinfo.get()->ndim;
  // assign rest of the fields.
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->span = span;
  data_ = std::move(n);
}

TensorStructInfo::TensorStructInfo(DataType dtype, int ndim, Span span) {
  ObjectPtr<TensorStructInfoNode> n = make_object<TensorStructInfoNode>();
  CHECK_GE(ndim, -1) << "ndim of TensorStructInfo must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->dtype = dtype;
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TensorStructInfoNode);

TVM_REGISTER_GLOBAL("relax.TensorStructInfo")
    .set_body_typed([](Optional<Expr> shape, DataType dtype, int ndim, Span span) {
      if (shape.defined()) {
        CHECK_EQ(ndim, kUnknownNDim) << "ValueError: Cannot both specify shape and ndim";
        return TensorStructInfo(shape.value(), dtype, span);
      } else {
        return TensorStructInfo(dtype, ndim, span);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorStructInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = static_cast<const TensorStructInfoNode*>(ref.get());
      if (node->shape.defined()) {
        p->stream << "TensorStructInfo(" << node->shape.value() << ", " << node->dtype << ")";
      } else {
        p->stream << "TensorStructInfo(" << node->dtype << ", ndim=" << node->ndim << ")";
      }
    });

// Tuple
TupleStructInfo::TupleStructInfo(Array<StructInfo> fields, Span span) {
  ObjectPtr<TupleStructInfoNode> n = make_object<TupleStructInfoNode>();
  n->fields = std::move(fields);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleStructInfoNode);

TVM_REGISTER_GLOBAL("relax.TupleStructInfo")
    .set_body_typed([](Array<StructInfo> fields, Span span) {
      return TupleStructInfo(fields, span);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleStructInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = static_cast<const TupleStructInfoNode*>(ref.get());
      p->stream << "TupleStructInfo(" << node->fields << ")";
    });

// Func
FuncStructInfo::FuncStructInfo(Array<StructInfo> params, StructInfo ret, Span span) {
  ObjectPtr<FuncStructInfoNode> n = make_object<FuncStructInfoNode>();
  n->params = std::move(params);
  n->ret = std::move(ret);
  n->span = span;
  data_ = std::move(n);
}

FuncStructInfo FuncStructInfo::OpaqueFunc(StructInfoDeriveFunc derive_func, Span span) {
  ObjectPtr<FuncStructInfoNode> n = make_object<FuncStructInfoNode>();
  n->derive_func = std::move(derive_func);
  n->ret = ObjectStructInfo();
  n->span = span;
  return FuncStructInfo(n);
}

FuncStructInfo FuncStructInfo::OpaqueFunc(StructInfo ret, Span span) {
  ObjectPtr<FuncStructInfoNode> n = make_object<FuncStructInfoNode>();
  n->ret = std::move(ret);
  n->span = span;
  return FuncStructInfo(n);
}

TVM_REGISTER_NODE_TYPE(FuncStructInfoNode);

TVM_REGISTER_GLOBAL("relax.FuncStructInfo")
    .set_body_typed([](Array<StructInfo> params, StructInfo ret, Span span) {
      return FuncStructInfo(params, ret, span);
    });

TVM_REGISTER_GLOBAL("relax.FuncStructInfoOpaqueFunc")
    .set_body_typed([](Optional<StructInfo> ret, Optional<StructInfoDeriveFunc> derive_func,
                       Span span) {
      if (derive_func.defined()) {
        ICHECK(!ret.defined()) << "ValueError: Cannot specify both ret and derive_func";
        return FuncStructInfo::OpaqueFunc(derive_func.value(), span);
      } else {
        return FuncStructInfo::OpaqueFunc(ret.value_or(ObjectStructInfo()), span);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FuncStructInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = static_cast<const FuncStructInfoNode*>(ref.get());
      p->stream << "FuncStructInfo(" << node->params << ", " << node->ret << ")";
    });

// Helper functions
// TODO(unity-team): add UpdateStructInfo once analysis.cc is upstreamed

TVM_REGISTER_GLOBAL("ir.ExprStructInfo").set_body_typed([](Expr expr) {
  return GetStructInfo(expr);
});

}  // namespace relax
}  // namespace tvm
