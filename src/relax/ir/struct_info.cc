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
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/tirx/expr.h>

#include "script_print_utils.h"

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  StructInfoNode::RegisterReflection();
  TensorStructInfoNode::RegisterReflection();
  TupleStructInfoNode::RegisterReflection();
  FuncStructInfoNode::RegisterReflection();
  ObjectStructInfoNode::RegisterReflection();
  PrimStructInfoNode::RegisterReflection();
  ShapeStructInfoNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax._func_si_args", [](FuncStructInfo node) -> ffi::List<ffi::Any> {
        ffi::List<ffi::Any> args;
        if (!node->IsOpaque()) {
          args.push_back(node->params.value());
          args.push_back(node->ret);
          args.push_back(static_cast<bool>(node->purity));
        }
        return args;
      })
      .def("relax._func_si_kwargs", [](FuncStructInfo node) -> ffi::Dict<ffi::String, ffi::Any> {
        ffi::Dict<ffi::String, ffi::Any> kwargs;
        if (node->IsOpaque()) {
          if (!node->ret->IsInstance<ObjectStructInfoNode>()) {
            kwargs.Set(ffi::String("ret"), node->ret);
          }
          if (node->purity) {
            kwargs.Set(ffi::String("purity"), true);
          }
        }
        return kwargs;
      })
      .def("relax._empty_array",
          [](ObjectStructInfo) -> ffi::Array<ffi::ObjectRef> {
            return ffi::Array<ffi::ObjectRef>();
          })
      .def("relax._prim_si_args", [](PrimStructInfo node) -> ffi::List<ffi::Any> {
        ffi::List<ffi::Any> args;
        if (node->value.defined()) {
          return args;
        }
        DataType dt = node->dtype;
        if (dt.is_void()) {
          return args;
        }
        ffi::String dtype_str = ffi::DLDataTypeToString(static_cast<DLDataType>(dt));
        args.push_back(dtype_str);
        return args;
      })
      .def("relax._prim_si_kwargs", [](PrimStructInfo node) -> ffi::Dict<ffi::String, ffi::Any> {
        ffi::Dict<ffi::String, ffi::Any> kwargs;
        if (node->value.defined()) {
          PrimExpr value = node->value.value();
          if (const auto* var = value.as<tirx::VarNode>()) {
            kwargs.Set(ffi::String("value"), ffi::String(var->name_hint));
          } else {
            kwargs.Set(ffi::String("value"), value);
          }
        }
        return kwargs;
      })
      .def("relax._shape_si_args", [](ShapeStructInfo node) -> ffi::List<ffi::Any> {
        ffi::List<ffi::Any> args;
        if (node->values.defined()) {
          ffi::Array<PrimExpr> values = node->values.value();
          ffi::List<ffi::Any> shape_items;
          for (int i = 0; i < static_cast<int>(values.size()); ++i) {
            PrimExpr v = values[i];
            if (const auto* int_imm = v.as<IntImmNode>()) {
              shape_items.push_back(int_imm->value);
            } else {
              shape_items.push_back(v);
            }
          }
          args.push_back(shape_items);
        }
        return args;
      })
      .def("relax._shape_si_kwargs", [](ShapeStructInfo node) -> ffi::Dict<ffi::String, ffi::Any> {
        ffi::Dict<ffi::String, ffi::Any> kwargs;
        if (!node->values.defined()) {
          kwargs.Set(ffi::String("ndim"), static_cast<int64_t>(node->ndim));
        }
        return kwargs;
      });
}

ObjectStructInfo::ObjectStructInfo(Span span) {
  ObjectPtr<ObjectStructInfoNode> n = ffi::make_object<ObjectStructInfoNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ObjectStructInfo", [](Span span) { return ObjectStructInfo(span); });
}

// Prim
PrimStructInfo::PrimStructInfo(PrimExpr value, Span span) {
  ObjectPtr<PrimStructInfoNode> n = ffi::make_object<PrimStructInfoNode>();
  n->dtype = value->dtype;
  n->value = std::move(value);
  n->span = span;
  data_ = std::move(n);
}

PrimStructInfo::PrimStructInfo(DataType dtype, Span span) {
  ObjectPtr<PrimStructInfoNode> n = ffi::make_object<PrimStructInfoNode>();
  n->dtype = dtype;
  n->value = std::nullopt;
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.PrimStructInfoFromDtype",
           [](DataType dtype, Span span) { return PrimStructInfo(dtype, span); })
      .def("relax.PrimStructInfoFromValue",
           [](PrimExpr value, Span span) { return PrimStructInfo(value, span); });
}

// Shape
ShapeStructInfo::ShapeStructInfo(ffi::Array<PrimExpr> values, Span span) {
  ObjectPtr<ShapeStructInfoNode> n = ffi::make_object<ShapeStructInfoNode>();
  n->ndim = static_cast<int>(values.size());
  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmNode>()) {
      return tvm::cast(DataType::Int(64), value);
    }
    TVM_FFI_ICHECK(value.dtype() == DataType::Int(64))
        << "the value in ShapeStructInfo can only have dtype of int64";
    return value;
  });
  n->span = span;
  data_ = std::move(n);
}

ShapeStructInfo::ShapeStructInfo(int ndim, Span span) {
  ObjectPtr<ShapeStructInfoNode> n = ffi::make_object<ShapeStructInfoNode>();
  TVM_FFI_ICHECK_GE(ndim, -1) << "ndim of ShapeStructInfo must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.ShapeStructInfo", [](ffi::Optional<ffi::Array<PrimExpr>> values, int ndim, Span span) {
        if (values.defined()) {
          TVM_FFI_CHECK_EQ(ndim, kUnknownNDim, ValueError) << "Cannot both specify values and ndim";
          return ShapeStructInfo(values.value(), span);
        } else {
          return ShapeStructInfo(ndim, span);
        }
      });
}

// Tensor
TensorStructInfo::TensorStructInfo(Expr shape, DataType dtype, ffi::Optional<VDevice> vdevice,
                                   Span span) {
  ObjectPtr<TensorStructInfoNode> n = ffi::make_object<TensorStructInfoNode>();
  // assign ndim before move
  ffi::Optional<ShapeStructInfo> sinfo = MatchStructInfo<ShapeStructInfo>(shape);
  TVM_FFI_ICHECK(sinfo) << "We expect shape to contain pre-set shape struct info";
  TVM_FFI_ICHECK(shape.defined()) << "Must provide a shape in this constructor";
  TVM_FFI_ICHECK(shape->IsInstance<ShapeExprNode>() || shape->IsInstance<VarNode>())
      << "We require shape to be normalized when constructing TensorStructInfo";
  n->ndim = sinfo.value()->ndim;
  // assign rest of the fields.
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->vdevice = vdevice;
  n->span = span;
  data_ = std::move(n);
}

TensorStructInfo::TensorStructInfo(DataType dtype, int ndim, ffi::Optional<VDevice> vdevice,
                                   Span span) {
  ObjectPtr<TensorStructInfoNode> n = ffi::make_object<TensorStructInfoNode>();
  TVM_FFI_ICHECK_GE(ndim, -1) << "ndim of TensorStructInfo must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->dtype = dtype;
  n->vdevice = vdevice;
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.TensorStructInfo", [](ffi::Optional<Expr> shape, ffi::Optional<DataType> dtype,
                                   int ndim, VDevice vdevice, Span span) {
        if (shape.defined()) {
          TVM_FFI_CHECK_EQ(ndim, kUnknownNDim, ValueError) << "Cannot both specify shape and ndim";
          return TensorStructInfo(shape.value(), dtype.value_or(DataType::Void()), vdevice, span);
        } else {
          return TensorStructInfo(dtype.value_or(DataType::Void()), ndim, vdevice, span);
        }
      });
}

// Tuple
TupleStructInfo::TupleStructInfo(ffi::Array<StructInfo> fields, Span span) {
  ObjectPtr<TupleStructInfoNode> n = ffi::make_object<TupleStructInfoNode>();
  n->fields = std::move(fields);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.TupleStructInfo", [](ffi::Array<StructInfo> fields, Span span) {
    return TupleStructInfo(fields, span);
  });
}

// Func
FuncStructInfo::FuncStructInfo(ffi::Array<StructInfo> params, StructInfo ret, bool purity,
                               Span span) {
  ObjectPtr<FuncStructInfoNode> n = ffi::make_object<FuncStructInfoNode>();
  n->params = std::move(params);
  n->ret = std::move(ret);
  n->purity = std::move(purity);
  n->span = span;
  data_ = std::move(n);
}

FuncStructInfo FuncStructInfo::OpaqueFunc(StructInfoDeriveFunc derive_func, bool purity,
                                          Span span) {
  ObjectPtr<FuncStructInfoNode> n = ffi::make_object<FuncStructInfoNode>();
  n->derive_func = std::move(derive_func);
  n->ret = ObjectStructInfo();
  n->purity = std::move(purity);
  n->span = span;
  return FuncStructInfo(n);
}

FuncStructInfo FuncStructInfo::OpaqueFunc(StructInfo ret, bool purity, Span span) {
  ObjectPtr<FuncStructInfoNode> n = ffi::make_object<FuncStructInfoNode>();
  n->ret = std::move(ret);
  n->purity = std::move(purity);
  n->span = span;
  return FuncStructInfo(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.FuncStructInfo",
           [](ffi::Array<StructInfo> params, StructInfo ret, bool purity, Span span) {
             return FuncStructInfo(params, ret, purity, span);
           })
      .def("relax.FuncStructInfoOpaqueFunc", [](ffi::Optional<StructInfo> ret,
                                                ffi::Optional<StructInfoDeriveFunc> derive_func,
                                                bool purity, Span span) {
        if (derive_func.defined()) {
          TVM_FFI_CHECK(!ret.defined(), ValueError) << "Cannot specify both ret and derive_func";
          return FuncStructInfo::OpaqueFunc(derive_func.value(), purity, span);
        } else {
          return FuncStructInfo::OpaqueFunc(ret.value_or(ObjectStructInfo()), purity, span);
        }
      });
}

// Helper functions
void UpdateStructInfo(Expr expr, StructInfo struct_info) {
  TVM_FFI_ICHECK(!expr->struct_info_.defined())
      << "To ensure idempotency, "
      << "the expression passed to UpdateStructInfo "
      << "must not have any prior StructInfo.  "
      << "However, expression " << expr << " has struct info " << expr->struct_info_
      << ", which cannot be overwritten with " << struct_info;
  expr->struct_info_ = struct_info;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.UpdateStructInfo",
           [](Expr expr, StructInfo struct_info) { UpdateStructInfo(expr, struct_info); })
      .def("ir.ExprStructInfo", [](Expr expr) { return GetStructInfo(expr); });
}

// ---- __ffi_text_print__ overrides ----

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // PrimStructInfo: R.Prim(dtype) or R.Prim(value=...)
  refl::TypeAttrDef<PrimStructInfoNode>().def(
      "__ffi_text_print__",
      [](PrimStructInfo n, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        ffi::List<text::ExprAST> args;
        ffi::List<ffi::String> kwargs_keys;
        ffi::List<text::ExprAST> kwargs_values;
        if (n->value.defined()) {
          kwargs_keys.push_back(ffi::String("value"));
          kwargs_values.push_back(PrintShapeValue(n->value.value(), path->Attr("value"), printer, false));
        } else {
          args.push_back(text::LiteralAST::Str(DType2Str(n->dtype)));
        }
        return text::ExprCallKw(Relax("Prim"), std::move(args),
                          std::move(kwargs_keys), std::move(kwargs_values));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // ShapeStructInfo: R.Shape([dims]) or R.Shape(ndim=N)
  refl::TypeAttrDef<ShapeStructInfoNode>().def(
      "__ffi_text_print__",
      [](ShapeStructInfo n, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        if (n->values.defined()) {
          ffi::Array<PrimExpr> shape = n->values.value();
          text::AccessPath shape_p = path->Attr("values");
          ffi::List<text::ExprAST> shape_docs;
          for (int i = 0, ndim = shape.size(); i < ndim; ++i) {
            shape_docs.push_back(PrintShapeValue(shape[i], shape_p->ArrayItem(i), printer, false));
          }
          return text::ExprCall(Relax("Shape"), {text::ListAST({}, std::move(shape_docs))});
        }
        return text::ExprCallKw(Relax("Shape"), {},
                          {ffi::String("ndim")}, {text::LiteralAST::Int(n->ndim)});
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // TensorStructInfo: R.Tensor((shape,), dtype=...) etc.
  refl::TypeAttrDef<TensorStructInfoNode>().def(
      "__ffi_text_print__",
      [](TensorStructInfo n, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        ffi::List<text::ExprAST> args;
        ffi::List<ffi::String> kwargs_keys;
        ffi::List<text::ExprAST> kwargs_values;
        if (n->shape.defined()) {
          // Dig into ShapeExpr to get individual dims
          if (const auto* shape = n->shape.value().as<ShapeExprNode>()) {
            auto shape_expr = ffi::GetRef<ShapeExpr>(shape);
            text::AccessPath shape_p = path->Attr("shape")->Attr("values");
            ffi::List<text::ExprAST> shape_docs;
            for (int i = 0, ndim = shape_expr->values.size(); i < ndim; ++i) {
              shape_docs.push_back(PrintShapeValue(shape_expr->values[i],
                                                    shape_p->ArrayItem(i), printer,
                                                    /*stringify_vars=*/false));
            }
            args.push_back(text::TupleAST({}, std::move(shape_docs)));
          } else {
            args.push_back(Print(printer, n->shape.value(), path->Attr("shape")));
          }
        }
        if (!n->IsUnknownDtype()) {
          kwargs_keys.push_back(ffi::String("dtype"));
          kwargs_values.push_back(text::LiteralAST::Str(DType2Str(n->dtype)));
        }
        if (!n->shape.defined() && !n->IsUnknownNdim()) {
          kwargs_keys.push_back(ffi::String("ndim"));
          kwargs_values.push_back(text::LiteralAST::Int(n->ndim));
        }
        // vdevice (matching V1 logic)
        if (n->vdevice.defined() && n->vdevice.value()->target.defined()) {
          kwargs_keys.push_back(ffi::String("vdevice"));
          VDevice vdev = n->vdevice.value();
          // Look up pre-computed "kind:kind_index:scope" from module.cc
          if (auto opt = printer->VarGet(vdev)) {
            kwargs_values.push_back(opt.value());
          } else {
            // Fallback: use target kind name and vdevice_id
            std::string dev_kind = vdev->target->kind->name;
            kwargs_values.push_back(text::LiteralAST::Str(
                dev_kind + ":" + std::to_string(vdev->vdevice_id) + ":" +
                std::string(vdev->memory_scope)));
          }
        }
        if (args.empty() && kwargs_keys.empty()) {
          return Relax("Tensor");
        }
        return text::ExprCallKw(Relax("Tensor"), std::move(args),
                          std::move(kwargs_keys), std::move(kwargs_values));
      });
}

}  // namespace relax
}  // namespace tvm
