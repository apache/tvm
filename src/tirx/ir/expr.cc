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
 * \file expr.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/extra/ir_traits.h>
#include <tvm/ffi/extra/pyast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include <optional>

#include "../../arith/scalable_expression.h"
#include "../../ir/printer_utils.h"
#include "../../support/str_escape.h"
#include "buffer_common.h"
#include "script_print_utils.h"

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;

  VarNode::RegisterReflection();
  SizeVarNode::RegisterReflection();
  IterVarNode::RegisterReflection();
  StringImmNode::RegisterReflection();
  CastNode::RegisterReflection();
  AddNode::RegisterReflection();
  SubNode::RegisterReflection();
  MulNode::RegisterReflection();
  DivNode::RegisterReflection();
  ModNode::RegisterReflection();
  FloorDivNode::RegisterReflection();
  FloorModNode::RegisterReflection();
  MinNode::RegisterReflection();
  MaxNode::RegisterReflection();
  EQNode::RegisterReflection();
  NENode::RegisterReflection();
  LTNode::RegisterReflection();
  LENode::RegisterReflection();
  GTNode::RegisterReflection();
  GENode::RegisterReflection();
  AndNode::RegisterReflection();
  OrNode::RegisterReflection();
  NotNode::RegisterReflection();
  SelectNode::RegisterReflection();
  BufferLoadNode::RegisterReflection();
  ProducerLoadNode::RegisterReflection();
  RampNode::RegisterReflection();
  BroadcastNode::RegisterReflection();
  LetNode::RegisterReflection();
  CallNode::RegisterReflection();
  ShuffleNode::RegisterReflection();
  CommReducerNode::RegisterReflection();
  ReduceNode::RegisterReflection();

  refl::GlobalDef()
      .def("tirx._tir_call_callee",
          [](ffi::pyast::IRPrinter printer, tirx::Call call) -> ffi::Any {
            if (!call->op->IsInstance<GlobalVarNode>()) {
              return ffi::Any();
            }
            namespace text = ::tvm::ffi::pyast;
            if (auto gv_doc = printer->VarGet(call->op)) {
              return ffi::Any(gv_doc.value());
            }
            GlobalVar op_gv = Downcast<GlobalVar>(call->op);
            for (const auto& kv : printer->obj2info) {
              if (const auto* gv_node = kv.first.as<GlobalVarNode>()) {
                if (gv_node->name_hint == op_gv->name_hint) {
                  return ffi::Any(kv.second->creator().cast<text::ExprAST>());
                }
              }
            }
            return printer->operator()(ffi::Any(call->op),
                                       text::AccessPath::Root()->Attr("op"));
          });

  // Global function definitions for all computed methods
  refl::GlobalDef()
      // VarNode / SizeVarNode type annotation helpers
      .def("tirx._var_type_or_null", [](ffi::AnyView /*ctx*/, Var node) -> ffi::Optional<Type> {
        if (!node->type_annotation.defined()) return ffi::Optional<Type>();
        if (const auto* tt = node->type_annotation.as<TupleTypeNode>()) {
          if (tt->fields.empty()) return ffi::Optional<Type>();
        }
        return ffi::Optional<Type>(node->type_annotation);
      })
      .def("tirx._sizevar_type_or_null", [](ffi::AnyView /*ctx*/, SizeVar node) -> ffi::Optional<Type> {
        if (!node->type_annotation.defined()) return ffi::Optional<Type>();
        if (const auto* tt = node->type_annotation.as<TupleTypeNode>()) {
          if (tt->fields.empty()) return ffi::Optional<Type>();
        }
        return ffi::Optional<Type>(node->type_annotation);
      })
      // IterVar args
      .def("tirx._iter_var_args", [](ffi::AnyView /*ctx*/, IterVar node) -> ffi::Array<ObjectRef> {
        const char* type_str;
        switch (static_cast<int>(node->iter_type)) {
          case kDataPar: type_str = "DataPar"; break;
          case kThreadIndex: type_str = "ThreadIndex"; break;
          case kCommReduce: type_str = "CommReduce"; break;
          case kOrdered: type_str = "Ordered"; break;
          case kOpaque: type_str = "DimInfo"; break;
          default: type_str = "Unrolled"; break;
        }
        ffi::Array<ObjectRef> result;
        result.push_back(node->var);
        result.push_back(node->dom);
        result.push_back(StringImm(ffi::String(type_str)));
        if (!node->thread_tag.empty()) {
          result.push_back(StringImm(ffi::String(std::string(node->thread_tag))));
        }
        return result;
      })
      // Cast args
      .def("tirx._cast_args", [](ffi::AnyView /*ctx*/, Cast node) -> ffi::Array<ObjectRef> {
        StringImm dtype_str(ffi::DLDataTypeToString(node->dtype));
        return {dtype_str, node->value};
      })
      // BinOp sugar checks: verify that re-constructing via the sugar function
      // produces the same node (i.e. the sugar round-trips).
#define TVM_TIRX_BINOP_SUGAR(lower, NodeTy, sugar_fn)                         \
      .def("tirx._" #lower "_sugar", [](ffi::AnyView /*ctx*/, tirx::NodeTy node) -> bool {  \
        PrimExpr ret = sugar_fn(node->a, node->b);                             \
        if (const auto* p = ret.as<NodeTy##Node>()) {                          \
          return p->a.same_as(node->a) && p->b.same_as(node->b);              \
        }                                                                      \
        return false;                                                          \
      })
      TVM_TIRX_BINOP_SUGAR(add, Add, tvm::add)
      TVM_TIRX_BINOP_SUGAR(sub, Sub, tvm::sub)
      TVM_TIRX_BINOP_SUGAR(mul, Mul, tvm::mul)
      TVM_TIRX_BINOP_SUGAR(floordiv, FloorDiv, tvm::floordiv)
      TVM_TIRX_BINOP_SUGAR(floormod, FloorMod, tvm::floormod)
      TVM_TIRX_BINOP_SUGAR(eq, EQ, tvm::equal)
      TVM_TIRX_BINOP_SUGAR(ne, NE, tvm::not_equal)
      TVM_TIRX_BINOP_SUGAR(lt, LT, tvm::less)
      TVM_TIRX_BINOP_SUGAR(le, LE, tvm::less_equal)
      TVM_TIRX_BINOP_SUGAR(gt, GT, tvm::greater)
      TVM_TIRX_BINOP_SUGAR(ge, GE, tvm::greater_equal)
      TVM_TIRX_BINOP_SUGAR(and, And, tvm::logical_and)
      TVM_TIRX_BINOP_SUGAR(or, Or, tvm::logical_or)
#undef TVM_TIRX_BINOP_SUGAR
      // Div sugar is special: also rejects integer-typed operands
      .def("tirx._div_sugar", [](ffi::AnyView /*ctx*/, tirx::Div node) -> bool {
        PrimExpr ret = tvm::div(node->a, node->b);
        if (!ret->IsInstance<DivNode>()) return false;
        if ((node->a->dtype.is_int() || node->a->dtype.is_uint()) &&
            (node->b->dtype.is_int() || node->b->dtype.is_uint())) {
          return false;
        }
        return true;
      })
      // Select args
      .def("tirx._select_args", [](ffi::AnyView /*ctx*/, Select node) -> ffi::Array<PrimExpr> {
        return {node->condition, node->true_value, node->false_value};
      })
      // Ramp args
      .def("tirx._ramp_args", [](ffi::AnyView /*ctx*/, Ramp node) -> ffi::Array<PrimExpr> {
        return {node->base, node->stride, node->lanes};
      })
      // Broadcast args
      .def("tirx._broadcast_args", [](ffi::AnyView /*ctx*/, Broadcast node) -> ffi::Array<PrimExpr> {
        return {node->value, node->lanes};
      })
      // Call callee and args
      .def("tirx._call_callee", [](ffi::AnyView /*ctx*/, tirx::Call node) -> ffi::String {
        if (auto* op = node->op.as<OpNode>()) {
          static const OpAttrMap<ffi::String> op_names =
              Op::GetAttrMap<ffi::String>("TScriptPrinterName");
          Op op_ref = ffi::GetRef<Op>(op);
          if (op_names.count(op_ref)) {
            return ffi::String("T." + std::string(op_names[op_ref]));
          }
          std::string full(op->name);
          auto pos = full.rfind('.');
          return ffi::String("T." + ((pos != std::string::npos) ? full.substr(pos + 1) : full));
        }
        return ffi::String("T.call");
      })
      .def("tirx._call_args", [](ffi::AnyView /*ctx*/, tirx::Call node) -> ffi::Array<ObjectRef> {
        ffi::Array<ObjectRef> result;
        int print_location = static_cast<int>(ScriptDtypePrintLocation::kNone);
        if (auto* op = node->op.as<OpNode>()) {
          static const OpAttrMap<TScriptDtypePrintLocation> dtype_locations =
              Op::GetAttrMap<TScriptDtypePrintLocation>("TScriptDtypePrintLocation");
          Op op_ref = ffi::GetRef<Op>(op);
          if (dtype_locations.count(op_ref)) {
            print_location = dtype_locations[op_ref].IntValue();
          }
        }
        std::string dtype_str = node->dtype.is_void() ? "void"
                                                      : ffi::DLDataTypeToString(node->dtype);
        bool is_llvm_intrin = false;
        if (auto* op = node->op.as<OpNode>()) {
          static const OpAttrMap<ffi::String> op_names =
              Op::GetAttrMap<ffi::String>("TScriptPrinterName");
          Op op_ref = ffi::GetRef<Op>(op);
          if (op_names.count(op_ref)) {
            std::string name(op_names[op_ref]);
            is_llvm_intrin = (name == "call_llvm_pure_intrin" || name == "call_llvm_intrin");
          }
        }
        if (print_location == static_cast<int>(ScriptDtypePrintLocation::kFirst)) {
          result.push_back(StringImm(ffi::String(dtype_str)));
        }
        for (int i = 0; i < static_cast<int>(node->args.size()); ++i) {
          if (i == 0 && is_llvm_intrin) {
            auto f_lookup = ffi::Function::GetGlobal("target.llvm_get_intrinsic_name");
            if (f_lookup.has_value() && node->args[0].as<IntImmNode>()) {
              int64_t id = node->args[0].as<IntImmNode>()->value;
              ffi::Any ret;
              ffi::AnyView args_view[1] = {ffi::AnyView(id)};
              f_lookup.value().CallPacked(args_view, 1, &ret);
              ffi::String name = ret.cast<ffi::String>();
              result.push_back(StringImm(name));
            } else {
              result.push_back(node->args[i]);
            }
          } else {
            result.push_back(node->args[i]);
          }
        }
        if (print_location == static_cast<int>(ScriptDtypePrintLocation::kLast)) {
          result.push_back(StringImm(ffi::String(dtype_str)));
        }
        return result;
      })
      // Shuffle args
      .def("tirx._shuffle_args", [](ffi::AnyView /*ctx*/, Shuffle node) -> ffi::Array<ObjectRef> {
        ffi::Array<ObjectRef> result;
        result.push_back(node->vectors);
        result.push_back(node->indices);
        return result;
      })
      // Reduce positional and kwargs
      .def("tirx._reduce_positional", [](ffi::AnyView /*ctx*/, Reduce node) -> ffi::Array<ObjectRef> {
        return {node->combiner};
      })
      .def("tirx._reduce_kwargs", [](ffi::AnyView /*ctx*/, Reduce node) -> ffi::Map<ffi::String, ObjectRef> {
        ffi::Map<ffi::String, ObjectRef> result;
        result.Set(ffi::String("source"), node->source);
        result.Set(ffi::String("init"), node->init);
        result.Set(ffi::String("axis"), node->axis);
        result.Set(ffi::String("condition"), node->condition);
        result.Set(ffi::String("value_index"), IntImm(DataType::Int(32), node->value_index));
        return result;
      })
      // BufferLoad indices: convert Ramp(base, stride, lanes) → [start, stop, step?]
      // Skip conversion when predicate is set (vload fallback needs raw indices)
      .def("tirx._load_indices", [](ffi::AnyView /*ctx*/, BufferLoad node) -> ffi::Array<ObjectRef> {
        if (node->predicate.defined()) return node->indices;
        ffi::Array<ObjectRef> result;
        for (const auto& idx : node->indices) {
          if (const auto* ramp = idx.as<RampNode>()) {
            if (ramp->stride.as<IntImmNode>()) {
              ffi::Array<PrimExpr> slice;
              slice.push_back(ramp->base);
              slice.push_back(ramp->base + ramp->lanes * ramp->stride);
              if (!is_one(ramp->stride)) {
                slice.push_back(ramp->stride);
              }
              result.push_back(slice);
              continue;
            }
          }
          result.push_back(idx);
        }
        return result;
      })
      // BufferStore indices: same Ramp→slice conversion
      // Skip conversion when predicate is set (vstore fallback needs raw indices)
      .def("tirx._store_indices", [](ffi::AnyView /*ctx*/, BufferStore node) -> ffi::Array<ObjectRef> {
        if (node->predicate.defined()) return node->indices;
        ffi::Array<ObjectRef> result;
        for (const auto& idx : node->indices) {
          if (const auto* ramp = idx.as<RampNode>()) {
            if (ramp->stride.as<IntImmNode>()) {
              ffi::Array<PrimExpr> slice;
              slice.push_back(ramp->base);
              slice.push_back(ramp->base + ramp->lanes * ramp->stride);
              if (!is_one(ramp->stride)) {
                slice.push_back(ramp->stride);
              }
              result.push_back(slice);
              continue;
            }
          }
          result.push_back(idx);
        }
        return result;
      })
      // BufferRegion indices
      .def("tirx._buf_region_indices", [](ffi::AnyView /*ctx*/, BufferRegion node) -> ffi::Array<ObjectRef> {
        ffi::Array<ObjectRef> result;
        for (const auto& r : node->region) {
          if (is_one(r->extent)) {
            // Single-point access: plain index instead of slice
            result.push_back(r->min);
          } else {
            // Range access: [min, min + extent] → slice
            ffi::Array<PrimExpr> pair;
            pair.push_back(r->min);
            pair.push_back(r->min + r->extent);
            result.push_back(pair);
          }
        }
        return result;
      });
}


/* \brief Convert an object to a PrimExpr
 *
 * All conversions to a PrimExpr are performed as part of the FFI,
 * when calling a function that accepts a PrimExpr as an argument.  If
 * a function must normalize to a PrimExpr (e.g. before accessing the
 * `expr.dtype` field), this function allows the FFI conversions to be
 * explicitly invoked.
 */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.convert",
                        [](ffi::Variant<PrimExpr, ffi::Array<PrimExpr>> expr) { return expr; });
  // Register __ffi_repr__ for Var/SizeVar so repr shows just the name
  refl::TypeAttrDef<VarNode>().def(refl::type_attr::kRepr,
                                   [](Var var, ffi::Function) -> ffi::String {
                                     return std::string(var->name_hint);
                                   });
  refl::TypeAttrDef<SizeVarNode>().def(refl::type_attr::kRepr,
                                       [](SizeVar var, ffi::Function) -> ffi::String {
                                         return std::string(var->name_hint);
                                       });
}

#define TVM_DEFINE_BINOP_CONSTRUCTOR(Name)                                    \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                             \
    using T = Name::ContainerType;                                            \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";             \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";             \
    TVM_FFI_CHECK(a.dtype() == b.dtype(), TypeError)                          \
        << "mismatched types. " << a.dtype() << " vs. " << b.dtype() << "\n"; \
    ObjectPtr<T> node = ffi::make_object<T>();                                \
    node->dtype = a.dtype();                                                  \
    node->a = std::move(a);                                                   \
    node->b = std::move(b);                                                   \
    node->span = std::move(span);                                             \
    data_ = std::move(node);                                                  \
  }

#define TVM_DEFINE_CMPOP_CONSTRUCTOR(Name)                                                  \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                           \
    using T = Name::ContainerType;                                                          \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                           \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                           \
    TVM_FFI_CHECK(a.dtype() == b.dtype(), TypeError)                                        \
        << "mismatched types. " << a.dtype() << " vs. " << b.dtype() << "\n";               \
    ObjectPtr<T> node = ffi::make_object<T>();                                              \
    DataType a_dtype = a.dtype();                                                           \
    node->dtype =                                                                           \
        DataType::Bool(a_dtype.get_lanes_or_vscale_factor(), a_dtype.is_scalable_vector()); \
    node->a = std::move(a);                                                                 \
    node->b = std::move(b);                                                                 \
    node->span = std::move(span);                                                           \
    data_ = std::move(node);                                                                \
  }

// Var
Var::Var(ffi::String name_hint, DataType dtype, Span span) {
  auto n = ffi::make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->type_annotation = GetTypeFromRuntimeDataType(dtype);
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  data_ = std::move(n);
}

Var::Var(ffi::String name_hint, Type type_annotation, Span span) {
  auto n = ffi::make_object<VarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = GetRuntimeDataType(type_annotation);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

Var Var::copy_with_name(const ffi::String& name) const {
  const VarNode* node = get();
  ObjectPtr<VarNode> new_ptr;
  if (auto* ptr = this->as<SizeVarNode>()) {
    new_ptr = ffi::make_object<SizeVarNode>(*ptr);
  } else {
    new_ptr = ffi::make_object<VarNode>(*node);
  }
  new_ptr->name_hint = name;
  return Var(new_ptr);
}

Var Var::copy_with_suffix(const ffi::String& suffix) const {
  return this->copy_with_name(get()->name_hint + suffix);
}

Var Var::copy_with_dtype(DataType dtype) const {
  const VarNode* node = get();
  ObjectPtr<VarNode> new_ptr;
  if (auto* ptr = this->as<SizeVarNode>()) {
    new_ptr = ffi::make_object<SizeVarNode>(*ptr);
  } else {
    new_ptr = ffi::make_object<VarNode>(*node);
  }
  new_ptr->type_annotation = GetTypeFromRuntimeDataType(dtype);
  new_ptr->dtype = std::move(dtype);
  return Var(new_ptr);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Var", [](ffi::String name_hint, ffi::AnyView type, Span span) {
    if (type.as<Type>()) {
      return Var(name_hint, type.cast<Type>(), span);
    } else {
      return Var(name_hint, type.cast<DataType>(), span);
    }
  });
}

// SizeVar
SizeVar::SizeVar(ffi::String name_hint, DataType dtype, Span span) {
  auto n = ffi::make_object<SizeVarNode>();
  n->name_hint = std::move(name_hint);
  n->type_annotation = GetTypeFromRuntimeDataType(dtype);
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  data_ = std::move(n);
}

SizeVar::SizeVar(ffi::String name_hint, Type type_annotation, Span span) {
  auto n = ffi::make_object<SizeVarNode>();
  n->name_hint = std::move(name_hint);
  n->dtype = GetRuntimeDataType(type_annotation);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.SizeVar",
                        [](ffi::String s, DataType t, Span span) { return SizeVar(s, t, span); });
}

// IterVar
IterVar::IterVar(Range dom, Var var, IterVarType t, ffi::String thread_tag, Span span) {
  ObjectPtr<IterVarNode> n = ffi::make_object<IterVarNode>();
  if (dom.defined() && dom->extent.defined()) {
    TVM_FFI_ICHECK(dom->extent.dtype().is_int())
        << "The dtype of the domain of an IterVar must be an integer type. However, the domain's "
           "dtype is "
        << dom->extent.dtype();
    TVM_FFI_ICHECK_EQ(dom->extent.dtype(), var.dtype())
        << "The dtype of the extent of an IterVar (" << dom->extent.dtype()
        << ") must match its associated Var's dtype (" << var.dtype() << ")";
  }
  n->dom = dom;
  n->var = var;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.IterVar", [](Range dom, Var var, int iter_type, ffi::String thread_tag, Span span) {
        return IterVar(dom, var, static_cast<IterVarType>(iter_type), thread_tag, span);
      });
}

// StringImm
StringImm::StringImm(ffi::String value, Span span) {
  ObjectPtr<StringImmNode> node = ffi::make_object<StringImmNode>();
  node->dtype = DataType::Handle();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.StringImm",
                        [](ffi::String value, Span span) { return StringImm(value, span); });
}

// Cast
Cast::Cast(DataType t, PrimExpr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK_EQ(t.get_lanes_or_vscale_factor(), value.dtype().get_lanes_or_vscale_factor());
  TVM_FFI_ICHECK(t.is_scalable_vector() == value.dtype().is_scalable_vector());
  ObjectPtr<CastNode> node = ffi::make_object<CastNode>();
  node->dtype = t;
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Cast", [](DataType dtype, PrimExpr value, Span span) {
    return Cast(dtype, value, span);
  });
}

// Add
TVM_DEFINE_BINOP_CONSTRUCTOR(Add);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Add",
                        [](PrimExpr a, PrimExpr b, Span span) { return Add(a, b, span); });
}

// Sub
TVM_DEFINE_BINOP_CONSTRUCTOR(Sub);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Sub",
                        [](PrimExpr a, PrimExpr b, Span span) { return Sub(a, b, span); });
}

// Mul
TVM_DEFINE_BINOP_CONSTRUCTOR(Mul);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Mul",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mul(a, b, span); });
}

// Div
TVM_DEFINE_BINOP_CONSTRUCTOR(Div);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Div",
                        [](PrimExpr a, PrimExpr b, Span span) { return Div(a, b, span); });
}

// Mod
TVM_DEFINE_BINOP_CONSTRUCTOR(Mod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Mod",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mod(a, b, span); });
}

// FloorDiv
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorDiv);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.FloorDiv",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorDiv(a, b, span); });
}

// FloorMod
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorMod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.FloorMod",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorMod(a, b, span); });
}

// Min
TVM_DEFINE_BINOP_CONSTRUCTOR(Min);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Min",
                        [](PrimExpr a, PrimExpr b, Span span) { return Min(a, b, span); });
}

// Max
TVM_DEFINE_BINOP_CONSTRUCTOR(Max);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Max",
                        [](PrimExpr a, PrimExpr b, Span span) { return Max(a, b, span); });
}

// EQ
TVM_DEFINE_CMPOP_CONSTRUCTOR(EQ);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.EQ",
                        [](PrimExpr a, PrimExpr b, Span span) { return EQ(a, b, span); });
}

// NE
TVM_DEFINE_CMPOP_CONSTRUCTOR(NE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.NE",
                        [](PrimExpr a, PrimExpr b, Span span) { return NE(a, b, span); });
}

// LT
TVM_DEFINE_CMPOP_CONSTRUCTOR(LT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.LT",
                        [](PrimExpr a, PrimExpr b, Span span) { return LT(a, b, span); });
}

// LE
TVM_DEFINE_CMPOP_CONSTRUCTOR(LE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.LE",
                        [](PrimExpr a, PrimExpr b, Span span) { return LE(a, b, span); });
}

// GT
TVM_DEFINE_CMPOP_CONSTRUCTOR(GT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.GT",
                        [](PrimExpr a, PrimExpr b, Span span) { return GT(a, b, span); });
}

// GE
TVM_DEFINE_CMPOP_CONSTRUCTOR(GE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.GE",
                        [](PrimExpr a, PrimExpr b, Span span) { return GE(a, b, span); });
}

// And
And::And(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  TVM_FFI_ICHECK(a.dtype().is_bool());
  TVM_FFI_ICHECK(b.dtype().is_bool());
  TVM_FFI_CHECK(a.dtype() == b.dtype(), TypeError) << "mismatched types";

  ObjectPtr<AndNode> node = ffi::make_object<AndNode>();
  node->dtype =
      DataType::Bool(a.dtype().get_lanes_or_vscale_factor(), a.dtype().is_scalable_vector());
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.And",
                        [](PrimExpr a, PrimExpr b, Span span) { return And(a, b, span); });
}

// Or
Or::Or(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  TVM_FFI_ICHECK(a.dtype().is_bool());
  TVM_FFI_ICHECK(b.dtype().is_bool());
  TVM_FFI_CHECK(a.dtype() == b.dtype(), TypeError) << "mismatched types";

  ObjectPtr<OrNode> node = ffi::make_object<OrNode>();
  node->dtype =
      DataType::Bool(a.dtype().get_lanes_or_vscale_factor(), a.dtype().is_scalable_vector());
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Or",
                        [](PrimExpr a, PrimExpr b, Span span) { return Or(a, b, span); });
}

// Not
Not::Not(PrimExpr a, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_ICHECK(a.dtype().is_bool());

  ObjectPtr<NotNode> node = ffi::make_object<NotNode>();
  DataType a_dtype = a.dtype();
  node->dtype = DataType::Bool(a_dtype.get_lanes_or_vscale_factor(), a_dtype.is_scalable_vector());
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Not", [](PrimExpr a, Span span) { return Not(a, span); });
}

// Select
Select::Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
  TVM_FFI_CHECK(condition.defined(), ValueError) << "condition is undefined";
  TVM_FFI_CHECK(true_value.defined(), ValueError) << "true_value is undefined";
  TVM_FFI_CHECK(false_value.defined(), ValueError) << "true_value is undefined";
  TVM_FFI_ICHECK(condition.dtype().is_bool());
  TVM_FFI_ICHECK(condition.dtype().get_lanes_or_vscale_factor() ==
                     true_value.dtype().get_lanes_or_vscale_factor() ||
                 condition.dtype().is_scalar());
  TVM_FFI_CHECK(false_value.dtype() == true_value.dtype(), TypeError)
      << "mismatched types. "
      << "False type: " << false_value.dtype() << "; True type: " << true_value.dtype();

  ObjectPtr<SelectNode> node = ffi::make_object<SelectNode>();
  node->dtype = true_value.dtype();
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.Select", [](PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
        return Select(condition, true_value, false_value, span);
      });
}

// Ramp
Ramp::Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span) {
  TVM_FFI_ICHECK(base.defined());
  TVM_FFI_ICHECK(stride.defined());
  TVM_FFI_ICHECK(base.dtype().is_scalar());
  TVM_FFI_ICHECK(stride.dtype().is_scalar());
  if (stride.dtype() != base.dtype()) {
    stride = cast(base.dtype(), stride);
  }

  ObjectPtr<RampNode> node = ffi::make_object<RampNode>();
  auto* lanes_as_int = lanes.as<IntImmNode>();
  if (lanes_as_int) {
    int lanes = static_cast<int>(lanes_as_int->value);
    TVM_FFI_ICHECK_GT(lanes, 1);
    node->dtype = base.dtype().with_lanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = arith::ExtractVscaleFactor(lanes);
    TVM_FFI_ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->dtype = base.dtype().with_scalable_vscale_factor(vscale_factor.value());
    lanes = Mul(Call(DataType::Int(32), tirx::builtin::vscale(), {}), vscale_factor.value());
    node->lanes = lanes;
  }
  node->base = base;
  node->stride = stride;
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Ramp", [](PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span) {
    return Ramp(base, stride, lanes, span);
  });
}

// Broadcast
Broadcast::Broadcast(PrimExpr value, PrimExpr lanes, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(value.dtype().is_scalar());

  ObjectPtr<BroadcastNode> node = ffi::make_object<BroadcastNode>();
  auto* lanes_int = lanes.as<IntImmNode>();
  if (lanes_int) {
    int lanes = static_cast<int>(lanes_int->value);
    TVM_FFI_ICHECK_GT(lanes, 1);
    node->dtype = value.dtype().with_lanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = arith::ExtractVscaleFactor(lanes);
    TVM_FFI_ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->dtype = value.dtype().with_scalable_vscale_factor(vscale_factor.value());
    lanes = Mul(Call(DataType::Int(32), tirx::builtin::vscale(), {}), vscale_factor.value());
    node->lanes = lanes;
  }
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = node;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Broadcast", [](PrimExpr value, PrimExpr lanes, Span span) {
    return Broadcast(value, lanes, span);
  });
}

// Let
Let::Let(Var var, PrimExpr value, PrimExpr body, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(body.defined());
  TVM_FFI_ICHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetNode> node = ffi::make_object<LetNode>();
  node->dtype = body.dtype();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Let", [](Var var, PrimExpr value, PrimExpr body, Span span) {
    return Let(var, value, body, span);
  });
}

// Call
Call::Call(DataType dtype, RelaxExpr op, ffi::Array<PrimExpr> args, Span span) {
  for (size_t i = 0; i < args.size(); ++i) {
    TVM_FFI_ICHECK(args[i].defined()) << "arg " << i << " is not defined()";
  }

  ObjectPtr<CallNode> node = ffi::make_object<CallNode>();
  node->dtype = dtype;
  node->op = std::move(op);
  node->args = std::move(args);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.Call",
      [](ffi::Optional<DataType> dtype, RelaxExpr op,
         ffi::Array<ffi::Variant<ffi::String, DLDataType, IterVar, BufferRegion, PrimExpr>> args,
         Span span) {
        ffi::Array<PrimExpr> prim_expr_args;
        for (const auto& it : args) {
          if (auto opt_str = it.as<ffi::String>()) {
            prim_expr_args.push_back(StringImm(opt_str.value()));
          } else if (auto opt_dtype = it.as<DLDataType>()) {
            prim_expr_args.push_back(StringImm(ffi::DLDataTypeToString(opt_dtype.value())));
          } else if (const auto* iter_var = it.as<IterVarNode>()) {
            prim_expr_args.push_back(iter_var->var);
          } else if (const auto* br = it.as<BufferRegionNode>()) {
            ffi::Array<PrimExpr> indices;
            for (Range r : br->region) {
              if (is_one(r->extent)) {
                indices.push_back(r->min);
              } else if (r->extent.as<IntImmNode>()) {
                indices.push_back(tirx::Ramp(r->min, make_const(r->min->dtype, 1), r->extent));
              } else {
                TVM_FFI_THROW(ValueError)
                    << "Cannot convert to BufferLoad: " << ffi::GetRef<BufferRegion>(br);
              }
            }
            prim_expr_args.push_back(BufferLoad(br->buffer, indices));
          } else {
            prim_expr_args.push_back(Downcast<PrimExpr>(it));
          }
        }
        return Call(dtype.value_or(DataType::Void()), op, prim_expr_args, span);
      });
}

// Shuffle
Shuffle::Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
  TVM_FFI_ICHECK_NE(vectors.size(), 0U);
  TVM_FFI_ICHECK_NE(indices.size(), 0U);

  DataType base_type = vectors[0].dtype().element_of();
  int total_lanes = 0;

  for (PrimExpr val : vectors) {
    TVM_FFI_ICHECK(val.dtype().element_of() == base_type);
    total_lanes += val.dtype().lanes();
  }
  TVM_FFI_ICHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  ObjectPtr<ShuffleNode> node = ffi::make_object<ShuffleNode>();
  node->dtype = base_type.with_lanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = node;
}

PrimExpr Shuffle::Concat(ffi::Array<PrimExpr> vectors, Span span) {
  TVM_FFI_ICHECK_NE(vectors.size(), 0);
  if (vectors.size() == 1) {
    return vectors[0];
  }
  ffi::Array<PrimExpr> indices;
  int index = 0;
  for (const PrimExpr& e : vectors) {
    for (int i = 0; i < e.dtype().lanes(); ++i) {
      indices.push_back(IntImm(DataType::Int(32), index++));
    }
  }
  return Shuffle(vectors, indices, span);
}

PrimExpr Shuffle::ExtractElement(PrimExpr vector, int index, Span span) {
  return Shuffle({vector}, {Integer(index)}, span);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Shuffle",
                        [](ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
                          return Shuffle(vectors, indices, span);
                        });
}

// CommReducer
CommReducer::CommReducer(ffi::Array<Var> lhs, ffi::Array<Var> rhs, ffi::Array<PrimExpr> result,
                         ffi::Array<PrimExpr> identity_element, Span span) {
  size_t n_group = result.size();
  TVM_FFI_CHECK_EQ(lhs.size(), n_group, ValueError)
      << "The number of vars in `lhs` must equal to the "
         "number of elements in `results`";
  TVM_FFI_CHECK_EQ(rhs.size(), n_group, ValueError)
      << "The number of vars in `rhs` must equal to the "
         "number of elements in `results`";
  TVM_FFI_CHECK_EQ(identity_element.size(), n_group, ValueError)
      << "The number of identities must equal to the number of elements in `results`";

  // Change the dtype of input vars to adapt to the dtype of identities
  ffi::ArrayObj* p_lhs = lhs.CopyOnWrite();
  ffi::ArrayObj* p_rhs = rhs.CopyOnWrite();
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  var_map.reserve(n_group * 2);
  for (int i = 0; i < static_cast<int>(n_group); ++i) {
    DataType dtype = identity_element[i].dtype();
    Var l = lhs[i].copy_with_dtype(dtype);
    Var r = rhs[i].copy_with_dtype(dtype);
    var_map[lhs[i].get()] = l;
    var_map[rhs[i].get()] = r;

    p_lhs->SetItem(i, l);
    p_rhs->SetItem(i, r);
  }

  ffi::ArrayObj* p_result = result.CopyOnWrite();
  for (int i = 0; i < static_cast<int>(n_group); ++i) {
    p_result->SetItem(i, Substitute(result[i], var_map));
  }

  auto node = ffi::make_object<CommReducerNode>();
  node->lhs = lhs;
  node->rhs = rhs;
  node->result = result;
  node->identity_element = identity_element;
  node->span = std::move(span);
  data_ = std::move(node);
}

ffi::Array<PrimExpr> CommReducerNode::operator()(ffi::Array<PrimExpr> a,
                                                 ffi::Array<PrimExpr> b) const {
  TVM_FFI_ICHECK_EQ(a.size(), b.size());
  TVM_FFI_ICHECK_EQ(lhs.size(), a.size());
  TVM_FFI_ICHECK_EQ(rhs.size(), b.size());
  ffi::Map<Var, PrimExpr> value_map;
  for (size_t i = 0; i < a.size(); ++i) {
    value_map.Set(lhs[i], a[i]);
    value_map.Set(rhs[i], b[i]);
  }
  return Substitute(this->result, value_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.CommReducer",
           [](ffi::Array<Var> lhs, ffi::Array<Var> rhs, ffi::Array<PrimExpr> result,
              ffi::Array<PrimExpr> identity_element,
              Span span) { return CommReducer(lhs, rhs, result, identity_element, span); })
      .def_method("tirx.CommReducerCombine", &tirx::CommReducerNode::operator());
}

// Reduce
Reduce::Reduce(CommReducer combiner, ffi::Array<PrimExpr> source, ffi::Array<IterVar> axis,
               PrimExpr condition, int value_index, ffi::Array<PrimExpr> init, Span span) {
  for (size_t i = 0; i < axis.size(); ++i) {
    TVM_FFI_ICHECK_EQ(axis[i]->iter_type, kCommReduce)
        << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = const_true();
  }
  auto n = ffi::make_object<ReduceNode>();
  TVM_FFI_ICHECK(source.defined());
  for (size_t i = 0; i < axis.size(); ++i) {
    TVM_FFI_ICHECK(axis[i].defined());
  }
  if (!init.empty()) {
    TVM_FFI_ICHECK_EQ(init.size(), source.size()) << "Number of inits should match number of exprs";
    for (size_t i = 0; i < init.size(); i++) {
      TVM_FFI_ICHECK(init[i].defined()) << "Init value must be defined";
      TVM_FFI_ICHECK(init[i]->IsInstance<ProducerLoadNode>() || init[i]->IsInstance<IntImmNode>() ||
                     init[i]->IsInstance<FloatImmNode>())
          << "init can only be a IntImm, FloatImm or ProducerLoad, "
          << "but received " << init[i] << " of type " << init[i]->GetTypeKey();
    }
  }
  n->dtype = source[value_index].dtype();
  n->combiner = std::move(combiner);
  n->source = std::move(source);
  n->init = std::move(init);
  n->axis = std::move(axis);
  n->condition = condition;
  n->value_index = value_index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.Reduce", [](CommReducer combiner, ffi::Array<PrimExpr> source, ffi::Array<IterVar> axis,
                        PrimExpr condition, int value_index, ffi::Array<PrimExpr> init, Span span) {
        return Reduce(combiner, source, axis, condition, value_index, init, span);
      });
}

// BufferLoad
void BufferLoadNode::LegalizeDType() {
  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    TVM_FFI_ICHECK(indices[i].dtype().is_scalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  if (indices.empty()) {
    this->dtype = buffer->dtype;
  } else {
    auto index_dtype = indices.back().dtype();
    bool is_buffer_dtype_scalable = buffer->dtype.is_scalable_vector();
    bool is_index_scalable = index_dtype.is_scalable_vector();

    TVM_FFI_ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
        << "Index dtype and buffer dtype can't both be scalable.";

    if (is_index_scalable) {
      this->dtype = buffer->dtype.with_scalable_vscale_factor(index_dtype.vscale_factor() *
                                                              buffer->dtype.lanes());
    } else if (is_buffer_dtype_scalable) {
      this->dtype = buffer->dtype.with_scalable_vscale_factor(buffer->dtype.vscale_factor() *
                                                              index_dtype.lanes());
    } else {
      this->dtype = buffer->dtype.with_lanes(index_dtype.lanes() * buffer->dtype.lanes());
    }
  }
}

BufferLoad::BufferLoad(Buffer buffer, ffi::Array<PrimExpr> indices,
                       ffi::Optional<PrimExpr> predicate, Span span) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), indices.size())
      << "Buffer " << buffer->name << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  if (predicate.defined()) {
    DataType predicate_dtype = predicate.value().dtype();

    bool is_index_scalable = indices.empty() ? false : indices.back().dtype().is_scalable_vector();
    bool is_predicate_scalable = predicate_dtype.is_scalable_vector();
    TVM_FFI_ICHECK_EQ(is_index_scalable, is_predicate_scalable)
        << "Predicate mask dtype and load indices must both be scalable.";

    int buffer_lanes = buffer->dtype.get_lanes_or_vscale_factor();
    int index_lanes = indices.empty() ? 1 : indices.back().dtype().get_lanes_or_vscale_factor();
    int predicate_lanes = predicate_dtype.get_lanes_or_vscale_factor();
    TVM_FFI_ICHECK_EQ(index_lanes * buffer_lanes, predicate_lanes)
        << "Got a predicate mask with " << predicate_lanes
        << " lanes, but trying to load a vector with " << index_lanes
        << " lanes. The number of lanes must match.";

    DataType predicate_element_dtype = predicate_dtype.element_of();
    TVM_FFI_ICHECK(predicate_element_dtype.is_predicate_dtype())
        << "Predicate mask elements must be boolean values, but got " << predicate_element_dtype
        << ".";
  }

  ObjectPtr<BufferLoadNode> node = ffi::make_object<BufferLoadNode>();
  node->buffer = std::move(buffer);
  node->indices = std::move(indices);
  node->predicate = std::move(predicate);
  node->span = std::move(span);
  node->LegalizeDType();
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.BufferLoad", [](Buffer buffer, ffi::Array<PrimExpr> indices,
                                              ffi::Optional<PrimExpr> predicate, Span span) {
    return BufferLoad(buffer, indices, predicate, span);
  });
}

// ProducerLoad
ProducerLoad::ProducerLoad(DataProducer producer, ffi::Array<PrimExpr> indices, Span span) {
  ObjectPtr<ProducerLoadNode> node = ffi::make_object<ProducerLoadNode>();
  node->dtype = producer->GetDataType();
  node->producer = std::move(producer);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ProducerLoad",
                        [](DataProducer producer, ffi::Array<PrimExpr> indices, Span span) {
                          return ProducerLoad(producer, indices, span);
                        });
}

// ---------------------------------------------------------------------------
// __ffi_text_print__ overrides
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;

  // CommReducer: lambda construction -- genuinely irreducible
  refl::TypeAttrDef<CommReducerNode>().def(
      "__ffi_text_print__",
      [](CommReducer node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        ffi::List<text::ExprAST> lhs_vars, rhs_vars, results;
        for (int i = 0; i < static_cast<int>(node->lhs.size()); ++i)
          lhs_vars.push_back(Print(printer, node->lhs[i], path->Attr("lhs")->ArrayItem(i)));
        for (int i = 0; i < static_cast<int>(node->rhs.size()); ++i)
          rhs_vars.push_back(Print(printer, node->rhs[i], path->Attr("rhs")->ArrayItem(i)));
        for (int i = 0; i < static_cast<int>(node->result.size()); ++i)
          results.push_back(Print(printer, node->result[i], path->Attr("result")->ArrayItem(i)));
        ffi::List<text::ExprAST> params;
        params.insert(params.end(), lhs_vars.begin(), lhs_vars.end());
        params.insert(params.end(), rhs_vars.begin(), rhs_vars.end());
        text::ExprAST lambda_body = (results.size() == 1) ? results[0] : text::TupleAST({}, results);
        text::LambdaAST lambda_ast({}, params, lambda_body);
        return text::ExprCall(TIR("comm_reducer"),
                        {lambda_ast, printer->PrintList(node->identity_element,
                                                        path->Attr("identity_element"))});
      });

  // IndexMap: T.index_map(lambda vars: (exprs...), [inverse_index_map=...])
  refl::TypeAttrDef<IndexMapNode>().def(
      "__ffi_text_print__",
      [](IndexMap node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        ffi::List<text::ExprAST> params;
        for (int i = 0; i < static_cast<int>(node->initial_indices.size()); ++i) {
          params.push_back(
              Print(printer, node->initial_indices[i],
                    path->Attr("initial_indices")->ArrayItem(i)));
        }
        ffi::List<text::ExprAST> exprs;
        for (int i = 0; i < static_cast<int>(node->final_indices.size()); ++i) {
          exprs.push_back(
              Print(printer, node->final_indices[i],
                    path->Attr("final_indices")->ArrayItem(i)));
        }
        text::ExprAST body = (exprs.size() == 1) ? exprs[0] : text::TupleAST({}, std::move(exprs));
        text::LambdaAST lambda_ast({}, std::move(params), body);
        if (node->inverse_index_map.defined()) {
          IndexMap inv = Downcast<IndexMap>(node->inverse_index_map);
          ffi::List<text::ExprAST> inv_params;
          for (int i = 0; i < static_cast<int>(inv->initial_indices.size()); ++i) {
            inv_params.push_back(
                Print(printer, inv->initial_indices[i],
                      path->Attr("inverse_index_map")->Attr("initial_indices")->ArrayItem(i)));
          }
          ffi::List<text::ExprAST> inv_exprs;
          for (int i = 0; i < static_cast<int>(inv->final_indices.size()); ++i) {
            inv_exprs.push_back(
                Print(printer, inv->final_indices[i],
                      path->Attr("inverse_index_map")->Attr("final_indices")->ArrayItem(i)));
          }
          text::ExprAST inv_body = (inv_exprs.size() == 1) ? inv_exprs[0]
                              : text::TupleAST({}, std::move(inv_exprs));
          text::LambdaAST inv_lambda({}, std::move(inv_params), inv_body);
          return text::ExprCallKw(TIR("index_map"), {lambda_ast},
                            {ffi::String("inverse_index_map")}, {inv_lambda});
        }
        return text::ExprCall(TIR("index_map"), {lambda_ast});
      });

  // Let: T.Let(body, where={var: value})
  refl::TypeAttrDef<LetNode>().def(
      "__ffi_text_print__",
      [](Let node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        using namespace printer;
        if (!printer->VarGet(node->var).has_value() && !printer->frames.empty()) {
          text::DefaultFrame frame = printer->frames.back().cast<text::DefaultFrame>();
          DefineNewTIRVar(node->var, printer, frame);
        }
        text::ExprAST body_doc = Print(printer, node->body, path->Attr("body"));
        text::ExprAST var_doc = Print(printer, node->var, path->Attr("var"));
        text::ExprAST val_doc = Print(printer, node->value, path->Attr("value"));
        text::DictAST where_dict({var_doc}, {val_doc});
        return text::ExprCallKw(TIR("Let"), {body_doc},
                          {ffi::String("where")}, {where_dict});
      });
}

}  // namespace tirx
}  // namespace tvm
