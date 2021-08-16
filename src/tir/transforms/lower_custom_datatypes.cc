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
 * \file tvm/src/pass/lower_custom_datatypes.cc
 * \brief Pass for lowering custom datatypes
 */

#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../target/datatype/registry.h"

namespace tvm {
namespace tir {

/*!
 * \brief Helper mutator to implement lowering of custom datatypes.
 *
 * Lowering datatypes works as follows: for every expression containing a custom
 * datatype, we search for a global (registered by the implementer of the custom
 * datatype) for lowering this type of expression, and uses it to lower the
 * expression.
 */
class CustomDatatypesLowerer : public StmtExprMutator {
 public:
  explicit CustomDatatypesLowerer(const std::string& target) : target_(target) {}

  PrimExpr VisitExpr_(const CastNode* op) final {
    auto type_code = op->dtype.code();
    auto src_type_code = op->value.dtype().code();
    // If either datatype is a registered custom datatype, we must lower.
    bool to_be_lowered = datatype::Registry::Global()->GetTypeRegistered(type_code) ||
                         datatype::Registry::Global()->GetTypeRegistered(src_type_code);
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    if (to_be_lowered) {
      auto lower = datatype::GetCastLowerFunc(target_, type_code, src_type_code);
      ICHECK(lower) << "Cast lowering function for target " << target_ << " destination type "
                    << static_cast<unsigned>(type_code) << " source type "
                    << static_cast<unsigned>(src_type_code) << " not found";
      return (*lower)(expr);
    }
    return expr;
  }

  PrimExpr VisitExpr_(const FloatImmNode* imm) final {
    auto type_code = imm->dtype.code();
    auto e = GetRef<PrimExpr>(imm);
    if (datatype::Registry::Global()->GetTypeRegistered(type_code)) {
      auto lower = datatype::GetFloatImmLowerFunc(target_, type_code);
      ICHECK(lower) << "FloatImm lowering function for target " << target_ << " type "
                    << static_cast<unsigned>(type_code) << " not found";
      return (*lower)(e);
    }
    return e;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);

    auto itr = var_remap_.find(var);
    if (itr != var_remap_.end()) {
      return itr->second;
    } else {
      return std::move(var);
    }
  }

  Stmt VisitStmt_(const AllocateNode* allocate) final {
    bool to_be_lowered = datatype::Registry::Global()->GetTypeRegistered(allocate->dtype.code());

    if (to_be_lowered) {
      auto new_allocate_type = DataType::UInt(allocate->dtype.bits(), allocate->dtype.lanes());
      auto new_buffer_var =
          Var(allocate->buffer_var->name_hint, PointerType(PrimType(new_allocate_type)));
      var_remap_[allocate->buffer_var] = new_buffer_var;

      Stmt stmt = StmtExprMutator::VisitStmt_(allocate);
      allocate = stmt.as<AllocateNode>();

      return Allocate(new_buffer_var, new_allocate_type, allocate->extents, allocate->condition,
                      allocate->body);
    } else {
      return StmtExprMutator::VisitStmt_(allocate);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* load) final {
    bool to_be_lowered = datatype::Registry::Global()->GetTypeRegistered(load->dtype.code());
    PrimExpr expr = StmtExprMutator::VisitExpr_(load);
    load = expr.as<LoadNode>();
    if (to_be_lowered) {
      auto new_load_type = DataType::UInt(load->dtype.bits());
      auto buffer_var = load->buffer_var;
      auto it = var_remap_.find(buffer_var);
      if (it != var_remap_.end()) {
        buffer_var = it->second;
      }
      return Load(new_load_type, buffer_var, load->index, load->predicate);
    }
    return expr;
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<StoreNode>();

    auto it = var_remap_.find(op->buffer_var);
    if (it != var_remap_.end()) {
      return Store(it->second, op->value, op->index, op->predicate);
    } else {
      return ret;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    // Due to legacy reasons, some attr node can contain
    // information(e.g. alignment) of buffer variables.
    // remap these vars when needed
    // TODO(tvm-team): remove the rewriting once the buffer var
    // attrs are being refactored into the corresponding definition node
    if (const auto* var_node = op->node.as<VarNode>()) {
      auto it = var_remap_.find(GetRef<Var>(var_node));
      if (it != var_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

  PrimExpr VisitExpr_(const CallNode* call) final {
    bool to_be_lowered = datatype::Registry::Global()->GetTypeRegistered(call->dtype.code());
    PrimExpr expr = StmtExprMutator::VisitExpr_(call);
    call = expr.as<CallNode>();
    if (to_be_lowered) {
      auto op = call->op.as<OpNode>();
      ICHECK(op != nullptr) << "Lowering non-intrinsic Calls not implemented";
      auto lower = datatype::GetIntrinLowerFunc(target_, op->name, call->dtype.code());
      ICHECK(lower) << "Intrinsic lowering function for target " << target_ << ", intrinsic name "
                    << op->name << ", type " << static_cast<unsigned>(call->dtype.code())
                    << " not found";
      return (*lower)(expr);
    }
    return expr;
  }

#define TVM_DEFINE_MUTATE_CUSTOM_DTYPE(OP, NodeName)                                 \
  PrimExpr VisitExpr_(const NodeName* op) final {                                    \
    auto type_code = op->dtype.code();                                               \
    bool to_be_lowered = datatype::Registry::Global()->GetTypeRegistered(type_code); \
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);                                 \
    op = expr.as<NodeName>();                                                        \
    if (to_be_lowered) {                                                             \
      auto lower = datatype::Get##OP##LowerFunc(target_, type_code);                 \
      ICHECK(lower) << #OP " lowering function for target " << target_ << " type "   \
                    << static_cast<unsigned>(type_code) << " not found";             \
      return (*lower)(expr);                                                         \
    }                                                                                \
    return expr;                                                                     \
  }

  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(Add, AddNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(Sub, SubNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(Mul, MulNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(Div, DivNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(Mod, ModNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(Min, MinNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(Max, MaxNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(EQ, EQNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(NE, NENode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(LT, LTNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(LE, LENode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(GT, GTNode);
  TVM_DEFINE_MUTATE_CUSTOM_DTYPE(GE, GENode);
  // Later changes may need to add more mutate functions as we support workloads with more ops.

#undef TVM_DEFINE_MUTATE_CUSTOM_DTYPE

 private:
  std::string target_;
  // remap buffer vars
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_remap_;
};

namespace transform {

Pass LowerCustomDatatypes() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined()) << "LowerCustomDatatypes: Require the target attribute";

    n->body = CustomDatatypesLowerer(target.value()->kind->name)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerCustomDatatypes", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerCustomDatatypes").set_body_typed(LowerCustomDatatypes);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
