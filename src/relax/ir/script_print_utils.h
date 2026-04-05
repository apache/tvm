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
#ifndef TVM_RELAX_IR_SCRIPT_PRINT_UTILS_H_
#define TVM_RELAX_IR_SCRIPT_PRINT_UTILS_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/global_info.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/distributed/global_info.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "../../ir/printer_utils.h"

namespace tvm {
namespace printer {

using namespace relax;

// Thread-local flag: true when printing function param/return annotations.
// Used by PrintShapeValue to decide whether to stringify symbolic TIR vars.
inline thread_local bool g_printing_func_annotation = false;

/*!
 * \brief Determine the function name from context.
 *
 * Priority: (1) GlobalVar from text::AccessPath MapItem key (module context),
 *           (2) global_symbol attribute,
 *           (3) fallback "main".
 */
inline ffi::String FindFuncName(const relax::Function& func, const text::IRPrinter& printer,
                    const text::AccessPath& path) {
  // Priority 1: binding name from module context (set via VarDefNoName in IRModule prologue)
  if (auto binding_expr = printer->VarGet(func)) {
    if (const auto* id_node = binding_expr.value().as<text::IdASTObj>()) {
      return id_node->name;
    }
  }
  // Priority 2: GlobalVar from text::AccessPath MapItem key
  if (path->step.defined()) {
    const auto& step = path->step.value();
    if (step->kind == ffi::reflection::AccessKind::kMapItem) {
      if (const auto* gv = step->key.as<GlobalVarNode>()) {
        return gv->name_hint;
      }
    }
  }
  // Priority 3: global_symbol attribute
  if (func->attrs.defined()) {
    auto it = func->attrs->dict.find("global_symbol");
    if (it != func->attrs->dict.end()) {
      return (*it).second.cast<ffi::String>();
    }
  }
  return "main";
}

/*!
 * \brief Check if this function is at top level in a module.
 *
 * A function is at the top level if it is reached via
 * Root->Attr("functions")->MapItem(gv).
 */
inline bool AtTopLevelInModule(const text::AccessPath& path) {
  // path should be MapItem(gv)
  if (!path->step.defined()) return false;
  if (path->step.value()->kind != ffi::reflection::AccessKind::kMapItem) return false;
  // parent should be Attr("functions")
  auto parent_opt = path->GetParent();
  if (!parent_opt.has_value()) return false;
  text::AccessPath parent = parent_opt.value();
  if (!parent->step.defined()) return false;
  if (parent->step.value()->kind != ffi::reflection::AccessKind::kAttr) return false;
  return true;
}

/*!
 * \brief Collect all tirx::Var references from a PrimExpr (e.g. shape dims).
 */
inline void CollectTIRVarsFromPrimExpr(const PrimExpr& expr,
                                std::vector<tirx::Var>* out,
                                std::unordered_set<const Object*>* seen) {
  tirx::PostOrderVisit(expr, [&](const ffi::ObjectRef& obj) {
    if (const auto* tv = obj.as<tirx::VarNode>()) {
      if (seen->insert(tv).second) {
        out->push_back(ffi::GetRef<tirx::Var>(tv));
      }
    }
  });
}

/*!
 * \brief Collect all tirx::Var references from a relax param's struct_info.
 *
 * Walks TensorStructInfo shapes, ShapeStructInfo values, and PrimStructInfo
 * values to find symbolic dimension variables.
 */
inline void CollectTIRVarsFromStructInfo(const StructInfo& sinfo,
                                  std::vector<tirx::Var>* out,
                                  std::unordered_set<const Object*>* seen) {
  if (const auto* tsi = sinfo.as<TensorStructInfoNode>()) {
    if (tsi->shape.defined()) {
      if (const auto* se = tsi->shape.value().as<ShapeExprNode>()) {
        for (const auto& dim : se->values) {
          CollectTIRVarsFromPrimExpr(dim, out, seen);
        }
      }
    }
  } else if (const auto* ssi = sinfo.as<ShapeStructInfoNode>()) {
    if (ssi->values.defined()) {
      for (const auto& val : ssi->values.value()) {
        CollectTIRVarsFromPrimExpr(val, out, seen);
      }
    }
  } else if (const auto* psi = sinfo.as<PrimStructInfoNode>()) {
    if (psi->value.defined()) {
      CollectTIRVarsFromPrimExpr(psi->value.value(), out, seen);
    }
  } else if (const auto* tsi = sinfo.as<TupleStructInfoNode>()) {
    for (const auto& field : tsi->fields) {
      CollectTIRVarsFromStructInfo(field, out, seen);
    }
  } else if (const auto* fsi = sinfo.as<FuncStructInfoNode>()) {
    if (fsi->params.defined()) {
      for (const auto& param : fsi->params.value()) {
        CollectTIRVarsFromStructInfo(param, out, seen);
      }
    }
    CollectTIRVarsFromStructInfo(fsi->ret, out, seen);
  }
}

/*!
 * \brief Print a PrimExpr for use in struct_info shape contexts.
 *
 * In V1, the "relax" dispatch for IntImm/FloatImm prints them as plain
 * integer/float literals (not T.int64(10)). This matches that behavior.
 */
inline text::ExprAST PrintShapeValue(const PrimExpr& e, const text::AccessPath& e_p, const text::IRPrinter& printer,
                        bool stringify_vars = false) {
  if (const auto* int_imm = e.as<IntImmNode>()) {
    if (int_imm->dtype.is_bool()) {
      return text::LiteralAST::Bool(int_imm->value != 0);
    }
    return text::LiteralAST::Int(int_imm->value);
  }
  if (const auto* float_imm = e.as<FloatImmNode>()) {
    return text::LiteralAST::Float(float_imm->value);
  }
  // For PrimExpr containing symbolic TIR Vars, stringify them (matching V1 PrintShapeVar).
  // Only do this in param/return annotation contexts (g_printing_func_annotation).
  if (stringify_vars || g_printing_func_annotation) {
    bool has_tir_var = false;
    tirx::PostOrderVisit(e, [&](const ffi::ObjectRef& obj) {
      if (obj->IsInstance<tirx::VarNode>()) has_tir_var = true;
    });
    if (has_tir_var) {
      // Helper: get the defined name for a TIR var (uses VarGet for the
      // printer-assigned name, which may differ from name_hint when there
      // are naming collisions, e.g. two different Vars both named "N").
      auto get_var_name = [&](const tirx::VarNode* v) -> std::string {
        tirx::Var var_ref = ffi::GetRef<tirx::Var>(v);
        if (auto defined = printer->VarGet(var_ref)) {
          if (const auto* id = defined.value().as<text::IdASTObj>()) {
            return id->name;
          }
        }
        return std::string(v->name_hint);
      };
      // Simple Var: just use defined name as string
      if (const auto* v = e.as<tirx::VarNode>()) {
        return text::LiteralAST::Str(get_var_name(v));
      }
      // Compound expressions (n * 2, etc.): build string from parts.
      // Precedence-aware so (N+3)//4 renders correctly.
      auto get_prec = [](const PrimExpr& expr) -> int {
        if (expr.as<tirx::OrNode>()) return 4;
        if (expr.as<tirx::AndNode>()) return 5;
        if (expr.as<tirx::NotNode>()) return 6;
        if (expr.as<tirx::EQNode>() || expr.as<tirx::NENode>() ||
            expr.as<tirx::LTNode>() || expr.as<tirx::LENode>() ||
            expr.as<tirx::GTNode>() || expr.as<tirx::GENode>()) return 7;
        if (expr.as<tirx::AddNode>() || expr.as<tirx::SubNode>()) return 12;
        if (expr.as<tirx::MulNode>() || expr.as<tirx::FloorDivNode>() ||
            expr.as<tirx::FloorModNode>()) return 13;
        return 100;
      };
      std::function<std::string(const PrimExpr&, int)> stringify;
      stringify = [&](const PrimExpr& expr, int parent_prec) -> std::string {
        if (const auto* v = expr.as<tirx::VarNode>()) return get_var_name(v);
        if (const auto* imm = expr.as<IntImmNode>()) return std::to_string(imm->value);
        int my_prec = get_prec(expr);
        std::string result;
        if (const auto* add = expr.as<tirx::AddNode>()) {
          result = stringify(add->a, my_prec) + " + " + stringify(add->b, my_prec + 1);
        } else if (const auto* sub = expr.as<tirx::SubNode>()) {
          result = stringify(sub->a, my_prec) + " - " + stringify(sub->b, my_prec + 1);
        } else if (const auto* mul = expr.as<tirx::MulNode>()) {
          result = stringify(mul->a, my_prec) + " * " + stringify(mul->b, my_prec + 1);
        } else if (const auto* div = expr.as<tirx::FloorDivNode>()) {
          result = stringify(div->a, my_prec) + " // " + stringify(div->b, my_prec + 1);
        } else if (const auto* mod = expr.as<tirx::FloorModNode>()) {
          result = stringify(mod->a, my_prec) + " % " + stringify(mod->b, my_prec + 1);
        } else if (const auto* mn = expr.as<tirx::MinNode>()) {
          return "T.min(" + stringify(mn->a, 0) + ", " + stringify(mn->b, 0) + ")";
        } else if (const auto* mx = expr.as<tirx::MaxNode>()) {
          return "T.max(" + stringify(mx->a, 0) + ", " + stringify(mx->b, 0) + ")";
        } else if (const auto* cast = expr.as<tirx::CastNode>()) {
          return stringify(cast->value, parent_prec);
        } else {
          std::ostringstream os;
          os << expr;
          return os.str();
        }
        if (my_prec < parent_prec) {
          return "(" + result + ")";
        }
        return result;
      };
      return text::LiteralAST::Str(stringify(e, 0));
    }
  }
  // Handle binary ops recursively to ensure child IntImm values print as
  // plain literals (matching V1's "relax" dispatch behavior).
  // Without this, children go through the generic trait printer which wraps
  // int64 values in T.int64().
  using Op = text::OperationASTObj;
  if (const auto* add = e.as<tirx::AddNode>()) {
    return text::OperationAST(Op::kAdd,
                        {PrintShapeValue(add->a, e_p->Attr("a"), printer),
                         PrintShapeValue(add->b, e_p->Attr("b"), printer)});
  }
  if (const auto* sub = e.as<tirx::SubNode>()) {
    return text::OperationAST(Op::kSub,
                        {PrintShapeValue(sub->a, e_p->Attr("a"), printer),
                         PrintShapeValue(sub->b, e_p->Attr("b"), printer)});
  }
  if (const auto* mul = e.as<tirx::MulNode>()) {
    return text::OperationAST(Op::kMult,
                        {PrintShapeValue(mul->a, e_p->Attr("a"), printer),
                         PrintShapeValue(mul->b, e_p->Attr("b"), printer)});
  }
  if (const auto* div = e.as<tirx::FloorDivNode>()) {
    return text::OperationAST(Op::kFloorDiv,
                        {PrintShapeValue(div->a, e_p->Attr("a"), printer),
                         PrintShapeValue(div->b, e_p->Attr("b"), printer)});
  }
  if (const auto* mod = e.as<tirx::FloorModNode>()) {
    return text::OperationAST(Op::kMod,
                        {PrintShapeValue(mod->a, e_p->Attr("a"), printer),
                         PrintShapeValue(mod->b, e_p->Attr("b"), printer)});
  }
  if (const auto* mn = e.as<tirx::MinNode>()) {
    return text::ExprCall(TIR("min"),
                    {PrintShapeValue(mn->a, e_p->Attr("a"), printer),
                     PrintShapeValue(mn->b, e_p->Attr("b"), printer)});
  }
  if (const auto* mx = e.as<tirx::MaxNode>()) {
    return text::ExprCall(TIR("max"),
                    {PrintShapeValue(mx->a, e_p->Attr("a"), printer),
                     PrintShapeValue(mx->b, e_p->Attr("b"), printer)});
  }
  // Comparison operators
  if (const auto* eq = e.as<tirx::EQNode>()) {
    return text::OperationAST(Op::kEq,
                        {PrintShapeValue(eq->a, e_p->Attr("a"), printer),
                         PrintShapeValue(eq->b, e_p->Attr("b"), printer)});
  }
  if (const auto* ne = e.as<tirx::NENode>()) {
    return text::OperationAST(Op::kNotEq,
                        {PrintShapeValue(ne->a, e_p->Attr("a"), printer),
                         PrintShapeValue(ne->b, e_p->Attr("b"), printer)});
  }
  if (const auto* lt = e.as<tirx::LTNode>()) {
    return text::OperationAST(Op::kLt,
                        {PrintShapeValue(lt->a, e_p->Attr("a"), printer),
                         PrintShapeValue(lt->b, e_p->Attr("b"), printer)});
  }
  if (const auto* le = e.as<tirx::LENode>()) {
    return text::OperationAST(Op::kLtE,
                        {PrintShapeValue(le->a, e_p->Attr("a"), printer),
                         PrintShapeValue(le->b, e_p->Attr("b"), printer)});
  }
  if (const auto* gt = e.as<tirx::GTNode>()) {
    return text::OperationAST(Op::kGt,
                        {PrintShapeValue(gt->a, e_p->Attr("a"), printer),
                         PrintShapeValue(gt->b, e_p->Attr("b"), printer)});
  }
  if (const auto* ge = e.as<tirx::GENode>()) {
    return text::OperationAST(Op::kGtE,
                        {PrintShapeValue(ge->a, e_p->Attr("a"), printer),
                         PrintShapeValue(ge->b, e_p->Attr("b"), printer)});
  }
  // Logical operators
  if (const auto* and_n = e.as<tirx::AndNode>()) {
    return text::OperationAST(Op::kAnd,
                        {PrintShapeValue(and_n->a, e_p->Attr("a"), printer),
                         PrintShapeValue(and_n->b, e_p->Attr("b"), printer)});
  }
  if (const auto* or_n = e.as<tirx::OrNode>()) {
    return text::OperationAST(Op::kOr,
                        {PrintShapeValue(or_n->a, e_p->Attr("a"), printer),
                         PrintShapeValue(or_n->b, e_p->Attr("b"), printer)});
  }
  // Unary Not
  if (const auto* not_n = e.as<tirx::NotNode>()) {
    return text::OperationAST(Op::kNot,
                        {PrintShapeValue(not_n->a, e_p->Attr("a"), printer)});
  }
  // For tirx::Var: print using the printer (which will resolve to the defined IdAST)
  if (e->IsInstance<tirx::VarNode>()) {
    return Print(printer, e, e_p);
  }
  // For other PrimExpr types, use the general printer
  return Print(printer, e, e_p);
}

/*!
 * \brief Extract a scalar value from a 0-d CPU tensor as an text::ExprAST literal.
 *
 * Returns std::nullopt for non-scalar or non-CPU tensors, or unsupported dtypes.
 * Matches the V1 SpecialScalar logic.
 */
inline ffi::Optional<text::ExprAST> SpecialScalar(const runtime::Tensor& tensor, const text::AccessPath& p) {
  DataType dtype(tensor->dtype);
  const void* data = tensor->data;
  if (tensor->ndim != 0 || tensor->device.device_type != kDLCPU) {
    return std::nullopt;
  }
  if (dtype == DataType::Int(8)) {
    return text::LiteralAST::Int(*reinterpret_cast<const int8_t*>(data));
  } else if (dtype == DataType::Int(16)) {
    return text::LiteralAST::Int(*reinterpret_cast<const int16_t*>(data));
  } else if (dtype == DataType::Int(32)) {
    return text::LiteralAST::Int(*reinterpret_cast<const int32_t*>(data));
  } else if (dtype == DataType::Int(64)) {
    return text::LiteralAST::Int(*reinterpret_cast<const int64_t*>(data));
  } else if (dtype == DataType::Float(16)) {
    uint16_t bits = *reinterpret_cast<const uint16_t*>(data);
    uint16_t sign_bit = (bits & 0x8000) >> 15;
    uint16_t exponent = (bits & 0x7C00) >> 10;
    uint16_t fraction = (bits & 0x03FF) >> 0;
    double value;
    if (exponent == 0x1F && fraction == 0) {
      value = std::numeric_limits<double>::infinity();
    } else if (exponent == 0x1F) {
      value = std::numeric_limits<double>::quiet_NaN();
    } else if (exponent == 0 && fraction == 0) {
      value = 0.0;
    } else if (exponent == 0) {
      value = ::std::pow(2.0, -24) * static_cast<double>(fraction);
    } else {
      value = ::std::pow(2.0, static_cast<double>(exponent) - 25) *
              static_cast<double>(fraction | (1 << 10));
    }
    if (sign_bit) {
      value *= -1.0;
    }
    return text::LiteralAST::Float(value);
  } else if (dtype == DataType::Float(32)) {
    return text::LiteralAST::Float(*reinterpret_cast<const float*>(data));
  } else if (dtype == DataType::Float(64)) {
    return text::LiteralAST::Float(*reinterpret_cast<const double*>(data));
  } else if (dtype == DataType::Bool()) {
    return text::LiteralAST::Bool(*reinterpret_cast<const uint8_t*>(data) != 0);
  } else {
    return std::nullopt;
  }
}

/*!
 * \brief Print a SeqExpr body as a list of statements (no return).
 *
 * Used by the If and VarBinding printers to print SeqExpr branches
 * matching V1's PrintSeqExpr(seq, path, d, use_ret=false).
 */
inline ffi::List<text::StmtAST> PrintSeqExprBody(const relax::SeqExpr& seq, const text::AccessPath& seq_path,
                                const text::IRPrinter& printer) {
  ffi::List<text::StmtAST> stmts;
  for (int i = 0; i < static_cast<int>(seq->blocks.size()); ++i) {
    text::NodeAST block_ast = printer->operator()(
        ffi::Any(seq->blocks[i]),
        seq_path->Attr("blocks")->ArrayItem(i)).cast<text::NodeAST>();
    if (auto* sb = block_ast.as<text::StmtBlockASTObj>()) {
      for (const auto& s : sb->stmts) stmts.push_back(s);
    } else if (block_ast->IsInstance<text::StmtASTObj>()) {
      stmts.push_back(Downcast<text::StmtAST>(block_ast));
    }
  }
  // Body is the last expression (printed as ExprStmt, not return)
  text::ExprAST ret_expr = Print(printer, seq->body, seq_path->Attr("body"));
  stmts.push_back(text::ExprStmtAST(ret_expr));
  return stmts;
}

}  // namespace printer
}  // namespace tvm

#endif  // TVM_RELAX_IR_SCRIPT_PRINT_UTILS_H_
