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
 * \file src/relax/transform/convert_layout.cc
 * \brief Automatic layout conversion pass, especially for axis swapping.
 */

#include <tvm/node/serialization.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/index_map.h>

#include "../op/tensor/manipulate.h"
#include "infer_layout_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

using tir::IndexMap;
using tir::Layout;

/*!
 * \brief Main logic to convert the layout of conv2d. Other ops
 * can adapt to such layout conversion following conv2d accordingly.
 *
 * Structurally speaking, a Relax function is composed of a series of VarBinding and
 * MatchCast. And a specific class of VarBindings is the basic unit we want to rewrite.
 * Formally, they are of the form:
 *
 * var = Call(Op, [args], attrs)
 *
 * where Op is a specific op we want to rewrite, and attrs is the attributes of the op.
 * var and args are all exprs with type Tensor or Tuple of Tensors. They might
 * be vars, constants, or Tuple of vars and constants.
 *
 * We register the layout inference function for each op (FRelaxInferLayout), which accepts the
 * current call, the desired layout of conv2d ops, and the layout map of previous vars. The result
 * of the layout inference function is contained in an InferLayoutOutput object, which contains 3
 * fields: input_layouts, output_layouts, and attr, which represents the expected input layout,
 * output_layout and converted attrs of the new op call.
 *
 * The rewrite pass does the rewriting in a single forward pass, where for each Call(Op),
 * we collect the current Layout of each input var, and let the InferLayout function to infer the
 * desired layout of the output. The rewriter will use these info to convert
 * the layout of inputs and attrs of the op call, and note down the new layout of the output.
 *
 * The desired layout of conv2d ops is a map from the name of the op to the desired layout of the
 * desired feature map, weight and output. For example, if we want to convert the layout of conv2d
 * from NCHW to NHWC, we can set the desired layout of conv2d to be {"conv2d": ["NHWC", "OHWI"]}.
 *
 * The way we represent the layout of a var is a NLayout object, which is a nested tuple of Layout.
 * The incoming layout of the module will be set as the default layout (We use ABCD... as the
 * default) Note that for operators like conv, pool, people typically use NHWC to refer to the axes.
 * But to be generic and support more operators, we use ABCD... to refer to the axes.
 *
 * Note that currently the layout conversion of conv2d only support axis swapping, such as NCHW to
 * NWHC. Packed layout such as NCHW to NCHW4c is not supported now.
 */
class LayoutConvertMutator : public ExprMutator {
 public:
  explicit LayoutConvertMutator(const Map<String, Array<String>>& desired_layouts)
      : desired_layouts_(desired_layouts) {}

 private:
  Array<Integer> LayoutToIntegers(const Layout& layout) {
    Array<Integer> ret;
    LayoutDecision src = InitialLayoutDecision(layout.ndim());
    for (size_t i = 0; i < layout.ndim(); ++i) {
      ret.push_back(Integer(src->layout.IndexOf(layout[i])));
    }
    return ret;
  }

  IndexMap LayoutIndexMap(int ndim, const Layout& src_layout, const Layout& desired_layout) {
    tir::BijectiveLayout todesired(src_layout, desired_layout);
    Optional<IndexMap> inverse_index_map;

    Array<tvm::tir::Var> initial_indices;
    Array<PrimExpr> initial_indices_expr;
    initial_indices.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      auto var = tvm::tir::Var("i" + std::to_string(i), DataType::Int(32));
      initial_indices.push_back(var);
      initial_indices_expr.push_back(var);
    }
    Array<PrimExpr> desired_shape = todesired.ForwardIndex(initial_indices_expr);
    return IndexMap(initial_indices, desired_shape, std::move(inverse_index_map));
  }

  Expr RewriteExpr(const Expr& expr, const NLayout& to) {
    auto fvisitleaf = [&](const Expr& expr, std::array<NLayout, 2> layouts) -> Expr {
      NLayout from = layouts[0], to = layouts[1];
      if (NLayoutEqual()(from, to) || layouts[0].LeafValue()->layout.name() == "") return expr;
      // If not both from and to are unknown, then none of them can be unknown.
      ICHECK(!NLayoutEqual()(from, LayoutDecision::InitUnknownDim()) &&
             !NLayoutEqual()(to, LayoutDecision::InitUnknownDim()))
          << "Cannot convert when exactly one of the layouts is unknown";
      const auto* tensor = GetStructInfoAs<TensorStructInfoNode>(expr);
      ICHECK(tensor != nullptr) << "Expect a tensor, but got: " << expr;

      if (from.LeafValue()->layout.ndim() == to.LeafValue()->layout.ndim()) {
        Layout axes = TransposeLike(InitialLayoutDecision(tensor->ndim)->layout,
                                    from.LeafValue()->layout, to.LeafValue()->layout);
        return permute_dims(expr, LayoutToIntegers(axes));
      } else {
        auto index_map = LayoutIndexMap(from.LeafValue()->layout.ndim(), from.LeafValue()->layout,
                                        to.LeafValue()->layout);
        ObjectPtr<LayoutTransformAttrs> attrs = make_object<LayoutTransformAttrs>();
        Array<IntImm> axis_separator;
        Array<IntImm> input_axis_separator;
        attrs->index_map = std::move(Downcast<IndexMap>(LoadJSON(SaveJSON(index_map))));
        attrs->axis_separators = std::move(axis_separator);
        attrs->input_axis_separators = std::move(input_axis_separator);
        const Op& layout_transform_op_ = Op::Get("relax.layout_transform");
        auto ret_expr = Call(layout_transform_op_, {expr}, Attrs{std::move(attrs)}, {});
        return ret_expr;
      }
    };
    return TransformTupleLeaf<LayoutDecision>(
        VarReplacer::Replace(expr, var_remap_),
        std::array<NLayout, 2>({GetNLayout(var_layout_map_, expr), to}), fvisitleaf);
  }

  Array<Expr> RewriteArgs(const Array<Expr>& args, const Array<NLayout>& to) {
    // The `Array<Expr> args` array contains both tensor and
    // non-tensor arguments, where the `Array<NLayout> to` array only
    // contains tensor arguments.  The number of tensor arguments in
    // `args` should match the full extent of `to`.

    ICHECK_LE(to.size(), args.size());

    std::vector<Expr> new_args;
    for (size_t i = 0; i < args.size(); ++i) {
      Expr arg = args[i];
      if (i < to.size()) {
        arg = RewriteExpr(arg, to[i]);
      }
      new_args.push_back(arg);
    }

    return std::move(new_args);
  }

  void VisitBinding(const Binding& binding) final {
    // Emit the binding
    ExprMutator::VisitBinding(binding);
    // The layout is default to be initial if not rewritten.
    if (var_layout_map_.find(binding->var) == var_layout_map_.end()) {
      var_layout_map_[binding->var] = InitialNLayout(binding->var);
    }
  }

  Expr VisitVars_(const Var& var) {
    // We encounter a var use outside of inferrable regions, we rewrite it to initial layout.
    return RewriteExpr(var, InitialNLayout(var));
  }

  Expr VisitExpr_(const VarNode* op) final { return VisitVars_(GetRef<Var>(op)); }

  bool HasUnknownDimTensor(const NLayout& nlayout) {
    bool find = false;
    auto fvisit = [&](const LayoutDecision& layout) {
      find = find | (NLayoutEqual()(layout, LayoutDecision::InitUnknownDim()));
    };
    ForEachLeaf<LayoutDecision>(nlayout, fvisit);
    return find;
  }

  bool HasUnknownDimTensor(const Array<Expr>& args) {
    for (const auto& arg : args) {
      if (IsNestedTensor(arg)) {
        if (HasUnknownDimTensor(GetNLayout(var_layout_map_, arg))) {
          return true;
        }
      }
    }
    return false;
  }

  Optional<InferLayoutOutput> GetInferLayoutInfo(const CallNode* call_node,
                                                 const Map<String, Array<String>>& desired_layouts,
                                                 const VarLayoutMap& var_layout_map) {
    const OpNode* op_node = call_node->op.as<OpNode>();
    if (op_node == nullptr) return NullOpt;
    Op op = Downcast<Op>(GetRef<Op>(op_node));
    const auto attr_map = Op::GetAttrMap<FRelaxInferLayout>("FRelaxInferLayout");
    if (attr_map.count(op) && !HasUnknownDimTensor(call_node->args)) {
      // If the op has FRelaxInferLayout, and all the input tensors have known ndim
      FRelaxInferLayout f = attr_map[op];
      return f(GetRef<Call>(call_node), desired_layouts, var_layout_map);
    } else {
      // Otherwise, we use the default policy.
      return NullOpt;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    Optional<InferLayoutOutput> res =
        GetInferLayoutInfo(call_node, desired_layouts_, var_layout_map_);
    ObjectPtr<CallNode> new_call = make_object<CallNode>(*call_node);
    new_call->struct_info_ = NullOpt;
    if (!res.defined() ||
        (!IsNestedTensor(binding->var) && !binding->var->IsInstance<DataflowVarNode>())) {
      // Default policy: use the initial layout.
      // When we don't have the infer layout info, or it's a non-tensor global var binding.
      std::vector<NLayout> input_layout;
      for (const auto& arg : call_node->args) {
        input_layout.push_back(InitialNLayout(arg));
      }
      Array<Expr> new_args = RewriteArgs(call_node->args, std::move(input_layout));
      new_call->args = std::move(new_args);
      ReEmitBinding(binding, builder_->Normalize(Call(new_call)));
      // update the layout map
      var_layout_map_[binding->var] = InitialNLayout(binding->var);
    } else {
      // Convert the layout according to the inferred layout output.
      Array<Expr> new_args = RewriteArgs(call_node->args, res.value()->input_layouts);
      for (const auto& [i, arg] : res.value()->new_args) {
        new_args.Set(i->value, arg);
      }
      new_call->args = std::move(new_args);

      new_call->attrs = std::move(res.value()->new_attrs);
      Expr cur_call = builder_->Normalize(Call(new_call));
      if (binding->var->IsInstance<DataflowVarNode>()) {
        // Dataflow var, we emit the rewritten call.
        ReEmitBinding(binding, cur_call);
        // update the layout map
        var_layout_map_[binding->var] = res.value()->output_layouts[0];
      } else {
        // Global var (tensor), we rewrite it to initial layout
        ICHECK(IsNestedTensor(binding->var));
        if (!NLayoutEqual()(res.value()->output_layouts[0], InitialNLayout(binding->var))) {
          Var new_var = builder_->Emit(cur_call);
          var_layout_map_[new_var] = res.value()->output_layouts[0];
          cur_call = builder_->Normalize(RewriteExpr(new_var, InitialNLayout(binding->var)));
        }
        ReEmitBinding(binding, cur_call);
        // update the layout map
        var_layout_map_[binding->var] = InitialNLayout(binding->var);
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) final {
    std::vector<NLayout> input_layout;
    for (const auto& field : val->fields) {
      if (binding->var->IsInstance<DataflowVarNode>()) {
        // Df var: Use the current realized layout to group the tuple;
        input_layout.push_back(GetNLayout(var_layout_map_, field));
      } else {
        // Global var: Use the initial layout to group the tuple;
        input_layout.push_back(InitialNLayout(field));
      }
    }
    Array<Expr> new_fields = RewriteArgs(val->fields, std::move(input_layout));
    if (IsNestedTensor(binding->var)) {
      ReEmitBinding(binding, builder_->Normalize(Tuple(new_fields)));
      var_layout_map_[binding->var] = input_layout;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) final {
    NLayout input_layout = binding->var->IsInstance<DataflowVarNode>()
                               ? GetNLayout(var_layout_map_, val->tuple)
                               : InitialNLayout(val->tuple);
    ReEmitBinding(binding, builder_->Normalize(
                               TupleGetItem(RewriteExpr(val->tuple, input_layout), val->index)));
    // update the layout map
    var_layout_map_[binding->var] = input_layout.NestedArray()[val->index];
  }

  void VisitBinding_(const MatchCastNode* binding) final {
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      ExprMutator::VisitBinding_(binding);
      return;
    }
    NLayout from_layout = InitialNLayout(binding->value);
    NLayout input_layout = GetNLayout(var_layout_map_, binding->value);
    auto fvisitleaf = [&](const StructInfo& sinfo, std::array<NLayout, 2> layouts) -> StructInfo {
      NLayout from = layouts[0], to = layouts[1];
      if (NLayoutEqual()(from, to)) return sinfo;
      // If not both from and to are unknown, then none of them can be unknown.
      ICHECK(!NLayoutEqual()(from, LayoutDecision::InitUnknownDim()) &&
             !NLayoutEqual()(to, LayoutDecision::InitUnknownDim()))
          << "Cannot convert when exactly one of the layouts is unknown";
      const TensorStructInfoNode* tsinfo = sinfo.as<TensorStructInfoNode>();
      ICHECK(tsinfo != nullptr) << "We can not set layout for non-tensor struct";
      if (!tsinfo->shape.defined()) return sinfo;
      const ShapeExprNode* shape = tsinfo->shape.value().as<ShapeExprNode>();
      if (shape == nullptr) return sinfo;
      ICHECK_EQ(shape->values.size(), to.LeafValue()->layout.ndim());
      std::vector<PrimExpr> new_shape;
      for (size_t i = 0; i < shape->values.size(); ++i) {
        new_shape.push_back(
            shape->values[from.LeafValue()->layout.IndexOf(to.LeafValue()->layout[i])]);
      }
      VDevice vdev = tsinfo->vdevice.value_or(VDevice());
      return TensorStructInfo(ShapeExpr(new_shape), tsinfo->dtype, vdev, tsinfo->span);
    };
    StructInfo new_struct_info = TransformTupleLeaf<LayoutDecision>(
        binding->struct_info, std::array<NLayout, 2>({from_layout, input_layout}), fvisitleaf);
    // re-emit old binding if nothing changes
    if (new_struct_info.same_as(binding->struct_info)) {
      builder_->EmitNormalized(GetRef<MatchCast>(binding));
    } else {
      Var new_var =
          builder_->EmitMatchCast(RewriteExpr(binding->value, input_layout), new_struct_info);
      var_layout_map_[binding->var] = input_layout;
      this->var_remap_[binding->var->vid] = new_var;
    }
  }

  std::unordered_map<Var, NLayout> var_layout_map_;
  Map<String, Array<String>> desired_layouts_;
};  // namespace relax

DataflowBlock ConvertLayoutPass(const DataflowBlock& df_block,
                                Map<String, Array<String>> desired_layouts) {
  LayoutConvertMutator mutator(desired_layouts);
  return Downcast<DataflowBlock>(mutator.VisitBindingBlock(df_block));
}

namespace transform {

Pass ConvertLayout(Map<String, Array<String>> desired_layouts) {
  runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)> pass_func =
      [=](DataflowBlock df_block, IRModule m, PassContext pc) {
        return Downcast<DataflowBlock>(ConvertLayoutPass(df_block, desired_layouts));
      };
  return CreateDataflowBlockPass(pass_func, 0, "ConvertLayout", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ConvertLayout").set_body_typed(ConvertLayout);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
