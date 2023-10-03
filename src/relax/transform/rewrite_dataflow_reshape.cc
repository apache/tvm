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
 * \file src/relax/transform/rewrite_dataflow_reshape.cc
 * \brief Transform all reshape within dataflow block to a relax.reshape operator
 */
#include <tvm/arith/analyzer.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>

#include <vector>

#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

std::vector<size_t> GetUsedArgsIndices(const tir::PrimFunc& fn, size_t num_args) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < num_args; ++i) {
    auto buffer_var = fn->buffer_map[fn->params[i]]->data;
    if (tir::UsesVar(fn->body, [=](const tir::VarNode* var) { return var == buffer_var.get(); })) {
      indices.push_back(i);
    }
  }
  return indices;
}

class DataflowReshapeRewriter : public ExprMutator {
 public:
  explicit DataflowReshapeRewriter(const IRModule& mod) : mod_(mod) {}

 private:
  using ExprMutator::VisitExpr_;

  BindingBlock VisitBindingBlock(const BindingBlock& block) final {
    // We only rewrite the bindings inside dataflow blocks.
    if (const auto* dataflow_block = block.as<DataflowBlockNode>()) {
      return VisitBindingBlock_(dataflow_block);
    } else {
      return block;
    }
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    // We only rewrite the bindings that are not dataflow output (which means they are not
    // externally referenced)
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      this->builder_->EmitNormalized(GetRef<VarBinding>(binding));
    } else {
      ExprMutator::VisitBinding_(binding);
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (call->op != call_tir_op) {
      return GetRef<Call>(call);
    }

    // We bring the calls of reshape PrimFunc back to calls of high-level
    // relax.reshape op, which will be lowered to calls of the ExternFunc
    // vm.builtin.reshape in the VMBuiltinLower pass.

    auto prim_fn = Downcast<tir::PrimFunc>(mod_->Lookup(Downcast<GlobalVar>(call->args[0])));

    auto args_expr = call->args[1];
    auto args_sinfo = GetStructInfo(args_expr).as<TupleStructInfoNode>();

    auto used_arg_indices = GetUsedArgsIndices(prim_fn, args_sinfo->fields.size());

    // The number of inputs to call_tir(reshape, (...)) might not be one, since FuseOps
    // can generate a fused TupleGetItem + reshape function whose input is a tuple. FuseTIR
    // then flattens the tuple input so that the fused TIR reshape function ends up having
    // multiple input buffers. But only one of them should be accessed and reshaped.
    if (used_arg_indices.size() != 1) {
      return GetRef<Call>(call);
    }
    size_t arg_index = used_arg_indices[0];

    auto arg_sinfo = Downcast<TensorStructInfo>(args_sinfo->fields[arg_index]);

    if (!IsCallingTIRReshape(call, arg_sinfo)) {
      return GetRef<Call>(call);
    }

    // Now we know that we're calling a reshape, but we don't yet know
    // on what.  Ideally, the arguments are either a tuple, or a
    // variable that is bound to a known tuple, but we may need to
    // fall back to a TupleGetItem.
    args_expr = UnwrapBindings(args_expr);
    auto arg = [&]() -> Expr {
      if (auto known_tuple = args_expr.as<TupleNode>()) {
        return known_tuple->fields[arg_index];
      } else {
        return TupleGetItem(args_expr, arg_index);
      }
    }();

    TensorStructInfo res_sinfo = Downcast<TensorStructInfo>(call->struct_info_.value());
    return reshape(arg, res_sinfo->shape.value());
  }

  bool IsCallingTIRReshape(const CallNode* call, TensorStructInfo inp_sinfo) {
    const GlobalVar& global_var = Downcast<GlobalVar>(call->args[0]);
    const auto* func = mod_->functions.Get(global_var).as<tir::PrimFuncNode>();
    ICHECK_NOTNULL(func);
    if (!HasReshapePattern(GetRef<tir::PrimFunc>(func))) {
      return false;
    }

    // The reshape operator expects that the number of elements in the source is the same
    // as the number of elements in the result. There are operators that could have a reshape
    // pattern that don't meet this requirement (e.g. strided_slice), and they should not be
    // converted to reshape.
    ICHECK(call->struct_info_.defined());
    TensorStructInfo res_sinfo = Downcast<TensorStructInfo>(call->struct_info_.value());

    if (inp_sinfo->IsUnknownDtype() || inp_sinfo->dtype != res_sinfo->dtype) {
      return false;
    }
    ICHECK(inp_sinfo->shape.defined() && res_sinfo->shape.defined());
    if (inp_sinfo->IsUnknownNdim() || res_sinfo->IsUnknownNdim()) {
      return false;
    }
    auto product = [](Array<PrimExpr> args) -> PrimExpr {
      PrimExpr p;
      if (args.empty()) {
        // Scalar tensors may be empty indicating a single element.
        p = 1;
      } else {
        p = args[0];
      }
      for (int i = 1, e = args.size(); i < e; ++i) p *= args[i];
      return p;
    };
    auto inp_count = product(inp_sinfo->GetShape().value());
    auto res_count = product(res_sinfo->GetShape().value());
    if (!arith::Analyzer().CanProveEqual(inp_count, res_count)) {
      return false;
    }

    return true;
  }

  const IRModule& mod_;
};

Expr RewriteDataflowReshape(const Function& f, const IRModule& mod) {
  return DataflowReshapeRewriter(mod)(f);
}

namespace transform {

Pass RewriteDataflowReshape() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(RewriteDataflowReshape(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "RewriteDataflowReshape", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RewriteDataflowReshape")
    .set_body_typed(RewriteDataflowReshape);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
