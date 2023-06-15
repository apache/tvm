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

#include <numeric>
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
    auto Unchanged = GetRef<Call>(call);

    if (!IsCallingTIRReshape(call)) {
      return Unchanged;
    }

    // We bring the calls of reshape PrimFunc back to calls of high-level
    // relax.reshape op, which will be lowered to calls of the ExternFunc
    // vm.builtin.reshape in the VMBuiltinLower pass.

    auto prim_fn = Downcast<tir::PrimFunc>(mod_->Lookup(Downcast<GlobalVar>(call->args[0])));
    auto arg_tuple = Downcast<Tuple>(call->args[1])->fields;
    auto used_arg_indices = GetUsedArgsIndices(prim_fn, arg_tuple.size());

    // The number of inputs to call_tir(reshape, (...)) might not be one, since FuseOps
    // can generate a fused TupleGetItem + reshape function whose input is a tuple. FuseTIR
    // then flattens the tuple input so that the fused TIR reshape function ends up having
    // multiple input buffers. But only one of them should be accessed and reshaped.
    ICHECK_EQ(used_arg_indices.size(), 1);

    auto arg = arg_tuple[used_arg_indices[0]];

    // The reshape operator expects that the number of elements in the source is the same
    // as the number of elements in the result. There are operators that could have a reshape
    // pattern that don't meet this requirement (e.g. strided_slice), and they should not be
    // converted to reshape.
    ICHECK(arg->struct_info_.defined() && call->struct_info_.defined());
    TensorStructInfo arg_sinfo = Downcast<TensorStructInfo>(arg->struct_info_.value());
    TensorStructInfo res_sinfo = Downcast<TensorStructInfo>(call->struct_info_.value());

    if (arg_sinfo->IsUnknownDtype() || arg_sinfo->dtype != res_sinfo->dtype) {
      return Unchanged;
    }
    ICHECK(arg_sinfo->shape.defined() && res_sinfo->shape.defined());
    if (arg_sinfo->IsUnknownNdim() || res_sinfo->IsUnknownNdim()) {
      return Unchanged;
    }
    auto product = [](Array<PrimExpr> args) -> PrimExpr {
      ICHECK(!args.empty());
      return std::reduce(args.begin(), args.end(), PrimExpr(1),
                         [](auto a, auto b) { return a * b; });
    };
    auto arg_count = product(arg_sinfo->GetShape().value());
    auto res_count = product(res_sinfo->GetShape().value());
    if (!arith::Analyzer().CanProveEqual(arg_count, res_count)) {
      return Unchanged;
    }

    return reshape(arg, res_sinfo->shape.value());
  }

  bool IsCallingTIRReshape(const CallNode* call) {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (call->op != call_tir_op) {
      return false;
    }
    const GlobalVar& global_var = Downcast<GlobalVar>(call->args[0]);
    const auto* func = mod_->functions.Get(global_var).as<tir::PrimFuncNode>();
    ICHECK_NOTNULL(func);
    return HasReshapePattern(GetRef<tir::PrimFunc>(func));
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
