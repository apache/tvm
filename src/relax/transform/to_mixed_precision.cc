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
 * \file src/relax/transform/to_mixed_precision.cc
 * \brief Automatic mixed precision pass.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include <array>
#include <cstdint>
#include <unordered_set>

#include "../op/nn/convolution.h"
#include "../op/tensor/datatype.h"
#include "../op/tensor/linear_algebra.h"
#include "infer_amp_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

using runtime::String;

int GetMixedPrecisionInfo(const CallNode* call_node) {
  const OpNode* op_node = call_node->op.as<OpNode>();
  if (op_node == nullptr) {
    return -1;
  }
  Op op = GetRef<Op>(op_node);
  auto attr_map = Op::GetAttrMap<TMixedPrecisionPolicy>("TMixedPrecisionPolicy");
  return attr_map.count(op) ? attr_map[op] : MixedPrecisionPolicyKind::kNever;
}

/*!
 * \brief Main logic to automatically cast fp32 input modules to fp16 for certain ops.
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
 * Depending on the properties of the op, we may have 3 different ways to rewrite it:
 *
 * 1. kAlways: Always cast the args to fp16
 *    Currently, this is only used for gemm and conv ops (to favor the use of TensorCore)
 *    We always cast the input args to fp16, and the dtype of the accumulator is configured
 *    by the global output_dtype parameter (default to fp32). We cast the output to fp16.
 *
 * 2. kFollow: If any of the args if fp32, cast all args to fp32. Otherwise, use fp16.
 *
 * 3. kNever: Never cast the args to fp16. Always cast all args to fp32 (the original dtype).
 *    Some ops, such as softmax, have numerical issues when using fp16. We will always use fp32
 *    to ensure the correctness.
 *
 * Note that in this case, we will actively cast the arg to fp16 only when it's used in kAlways.
 * This is to ensure that we have numerical stability to the best effort.
 *
 * DTypeDecisionCollector:
 *   Note that if some tensor is only used in kAlways ops, we can store it in fp16 without worsening
 *   numerical stability or using more storage. We use a backward propagation pass to detect such
 *   tensors. We will store the information of each var in the only_fp16_map_.
 *
 *   We reuse the NTtype struct to store the information of each var. There are 3 kinds of info:
 *     - Unknown (Float0): we never encounter a use of this tensor
 *     - Float16: we only encounter uses of this tensor in kAlways ops
 *     - Float32: we encounter some use of this tensor outside of kAlways ops
 *   The info value forms a semi-lattice, where Float8 is the top, Float16 is the middle, and
 *   Float32 is the bottom. The lower bound of two info values is the one with more bits.
 *
 * ToMixedPrecisionRewriter:
 *   We will then use a forward propagation pass to rewrite the program. Since we only keep one
 *   specific data type for each var, and we will cast the var to the required dtype locally when we
 *   encounter its use if needed. Note that we may cast the var to some certain dtype multiple
 *   times, but we decide not to store and reuse the casted copy due to the storage concern and to
 *   be more friendly to inlining and operator fusion. We will store the var to fp16 if it's only
 *   used in kAlways ops, otherwise we will store it as the natural output dtype of the op.
 *
 * The information of each op is registered in the
 * Op::GetAttr<FInferMixedPrecision>("FInferMixedPrecision"). The registered function has signature:
 * FInferMixedPrecision. We will call the registered function with the original call and the global
 * output_dtype parameter. The registered function will return the policy of the op, whether the op
 * can adjust the dtype of the accumulator, and the new call node with output_dtype set to the
 * global output_dtype parameter.
 *
 * Key design: wrap_param op
 *   We need to use fp16 parameters (which appear as constants in the program), but the type
 *   inference will fail if some parameters are fp16 and some are fp32 in the original module. To
 *   solve this, we introduce a new op wrap_param, which will wrap the original parameter and cast
 *   it to fp32 var.
 *
 *   When we encounter the var afterwards, we will directly replace it with the parameter. This
 *   information is tracked by the const_map_.
 */
class DTypeDecisionCollector : public ExprVisitor {
 public:
  explicit DTypeDecisionCollector(DataType output_dtype) : output_dtype_(output_dtype) {}

  static VarDTypeMap Collect(Function func, DataType output_dtype) {
    DTypeDecisionCollector collector(output_dtype);
    collector.VisitExpr(func);
    return std::move(collector.only_fp16_map_);
  }

 private:
  NType GetDType(const Var& var) {
    auto it = only_fp16_map_.find(var);
    if (it == only_fp16_map_.end()) {
      // we never encounter this var before
      NType unknown = NTypeFrom(var, unknown_);
      only_fp16_map_[var] = unknown;
      return unknown;
    }
    return it->second;
  }

  // merge the message for a var
  void UpdateVarDTypeMap(const Var& var, const NType& dtype) {
    auto it = only_fp16_map_.find(var);
    if (it == only_fp16_map_.end()) {
      only_fp16_map_[var] = dtype;
    } else {
      only_fp16_map_[var] = NTypeMerge(it->second, dtype);
    }
  }

  // merge the message for all vars in the expr list
  void RequireArgsToType(Array<Expr> args, Array<NType> to) {
    ICHECK(args.size() == to.size()) << "Invalid target dtypes";
    for (size_t i = 0; i < args.size(); ++i) {
      auto fvisitleaf = [&](const Expr& expr, NType to) {
        if (const auto* var = expr.as<VarNode>()) {
          UpdateVarDTypeMap(GetRef<Var>(var), to);
        } else if (expr->IsInstance<ConstantNode>()) {
          // Constant can be casted anyway, so we don't need to do anything here
          return;
        } else {
          LOG(FATAL) << "Unsupported argument type: " << expr->GetTypeKey();
        }
      };
      DecomposeNestedMsg(args[i], to[i], fvisitleaf);
    }
  }

  // merge the message for all vars in the expr list
  void RequireArgsToType(Array<Expr> args, DataType to) {
    std::vector<Expr> arg_arr;
    std::vector<NType> to_arr;
    for (const Expr& arg : args) {
      if (IsNestedTensor(arg)) {
        // only require the nested tensor args
        arg_arr.push_back(arg);
        to_arr.push_back(NTypeFrom(arg, to));
      }
    }
    RequireArgsToType(std::move(arg_arr), std::move(to_arr));
  }

  void VisitVars_(const VarNode* op) {
    Var var = GetRef<Var>(op);
    if (IsNestedTensor(var)) {
      // require the var to be fp32 (its original dtype)
      UpdateVarDTypeMap(var, NTypeFrom(var, fp32_));
      return;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) final { VisitVars_(op); }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    auto policy = GetMixedPrecisionInfo(call_node);
    if (policy == -1) {
      ExprVisitor::VisitBinding_(binding, call_node);
      return;
    }
    if (policy == kAlways) {
      // require inputs to be fp16
      RequireArgsToType(call_node->args, fp16_);
    } else if (policy == kFollow || policy == kNever) {
      // require inputs to be fp32 (the original dtype)
      RequireArgsToType(call_node->args, fp32_);
    } else {
      LOG(FATAL) << "Unsupported TMixedPrecisionPolicy: " << policy;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple_node) final {
    // require input fields to be the type of the lhs field respectively
    NType lhs_type = GetDType(binding->var);
    RequireArgsToType(tuple_node->fields, lhs_type.NestedArray());
  }

  void VisitBinding_(const VarBindingNode* binding,
                     const TupleGetItemNode* tuple_get_item_node) final {
    // require the i-th field rhs tuple to be the type of the lhs
    NType lhs_type = GetDType(binding->var);
    std::vector<NType> require_rhs;
    const TupleStructInfoNode* sinfo =
        tuple_get_item_node->tuple->struct_info_.as<TupleStructInfoNode>();
    ICHECK(sinfo != nullptr) << "TupleGetItemNode must have TupleStructInfo";
    for (size_t i = 0; i < sinfo->fields.size(); ++i) {
      if (i == static_cast<size_t>(tuple_get_item_node->index)) {
        require_rhs.push_back(lhs_type);
      } else {
        require_rhs.push_back(NTypeFrom(sinfo->fields[i], unknown_));
      }
    }
    RequireArgsToType({tuple_get_item_node->tuple}, {NType(require_rhs)});
  }

  // override the following methods to visit in backward order
  void VisitExpr_(const SeqExprNode* op) final {
    this->VisitSpan(op->span);
    this->VisitExpr(op->body);
    for (auto it = op->blocks.rbegin(); it != op->blocks.rend(); it++) {
      this->VisitBindingBlock(*it);
    }

    if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
      this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
    }
  }

  void VisitBindingBlock_(const BindingBlockNode* block) { return; }

  void VisitBindingBlock_(const DataflowBlockNode* block) {
    for (auto it = block->bindings.rbegin(); it != block->bindings.rend(); it++) {
      this->VisitBinding(*it);
    }
  }

  void VisitExpr_(const IfNode* op) final {
    this->VisitSpan(op->span);
    this->VisitExpr(op->true_branch);
    this->VisitExpr(op->false_branch);
    this->VisitExpr(op->cond);

    if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
      this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
    }
  }

  DataType unknown_ = DataType(DataType::TypeCode::kFloat, 0, 1);
  DataType fp16_ = DataType(DataType::TypeCode::kFloat, 16, 1);
  DataType fp32_ = DataType(DataType::TypeCode::kFloat, 32, 1);
  DataType output_dtype_;
  VarDTypeMap only_fp16_map_;
};

class ToMixedPrecisionRewriter : public ExprMutator {
 public:
  explicit ToMixedPrecisionRewriter(const VarDTypeMap* only_fp16_map, DataType output_dtype,
                                    const std::unordered_set<std::string>& fp16_input_names)
      : only_fp16_map_(only_fp16_map),
        output_dtype_(output_dtype),
        fp16_input_names_(fp16_input_names) {}

 private:
  Var GetRemapped(const Var& var) {
    auto it = var_remap_.find(var->vid);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      if (fp16_input_names_.count(var->name_hint())) {
        auto sinfo = GetStructInfo(var);
        if (auto tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
          VDevice vdev = VDevice();
          if (tensor_sinfo->vdevice.defined()) {
            vdev = tensor_sinfo->vdevice.value();
          }
          TensorStructInfo fp16_sinfo(tensor_sinfo->shape.value(), DataType::Float(16), vdev,
                                      tensor_sinfo->span);
          Var fp16_var(var->vid, fp16_sinfo, var->span);
          var_remap_[var->vid] = fp16_var;
          return fp16_var;
        }
      }
      return var;
    }
  }

  Array<Expr> RemapArgs(const Array<Expr>& args) {
    return args.Map([this](Expr arg) { return VarReplacer::Replace(arg, var_remap_); });
  }

  // Util function to rewrite the expr to the given dtype
  // rewrite each leaf tensor to the given dtype if necessary
  // Note that this function only accepts expr with nested tensor type
  Expr RewriteExpr(const Expr& expr, const NType& to) {
    auto fvisitleaf = [&](const Expr& expr, std::array<NType, 1> to) -> Expr {
      const auto* tensor = GetStructInfoAs<TensorStructInfoNode>(expr);
      ICHECK(tensor != nullptr) << "Only support rewriting tensor expr";
      // We only rewrite the expr if the dtype is not the same as the given dtype
      if (NTypeEqual()(to[0], NTypeFrom(expr))) return expr;
      // We only rewrite the expr if the dtype is fp16 or fp32, dtypes such as int32, float64 is not
      // supported to be rewritten
      if (tensor->dtype != fp16_ && tensor->dtype != fp32_) return expr;
      return astype(expr, DataType(String2DLDataType(to[0].LeafValue())));
    };
    return TransformTupleLeaf<String>(expr, std::array<NType, 1>({to}), fvisitleaf);
  }

  Array<Expr> RewriteArgs(const Array<Expr>& args, DataType to) {
    Array<Expr> new_args;
    for (const Expr& arg : args) {
      if (IsNestedTensor(arg)) {
        new_args.push_back(RewriteExpr(arg, NTypeFrom(arg, to)));
      } else {
        new_args.push_back(arg);
      }
    }
    return new_args;
  }

  template <typename T>
  bool CheckInFP16Range(const std::vector<uint8_t>& bytes, size_t num_elem) {
    const T* ptr = reinterpret_cast<const T*>(bytes.data());
    constexpr float fp16_max = 65504;
    for (size_t i = 0; i < num_elem; ++i) {
      if (std::abs(ptr[i]) > fp16_max) return false;
    }
    return true;
  }

  bool AllFP16Castable(const Array<Expr>& args) {
    auto is_fp16 = [](StructInfo sinfo) {
      if (auto tensor_sinfo = sinfo.as<TensorStructInfoNode>();
          tensor_sinfo && tensor_sinfo->dtype == DataType::Float(16)) {
        return true;
      }
      return false;
    };

    auto is_in_fp16_range = [this](const ConstantNode* constant) {
      const auto& data = constant->data;
      if (data->dtype.lanes > 1) {
        // Skip vectorized types for now.
        return false;
      }

      if (data.DataType() == DataType::Float(16)) {
        return true;
      }

      auto shape = data.Shape();
      size_t size_1d = 1;
      for (size_t i = 0; i < shape.size(); ++i) {
        size_1d *= shape[i];
      }
      auto elem_bytes = data->dtype.bits / 8;
      std::vector<uint8_t> bytes(size_1d * elem_bytes);
      data.CopyToBytes(bytes.data(), bytes.size());

      if (data.DataType() == DataType::Float(32)) {
        return CheckInFP16Range<float>(bytes, size_1d);
      } else if (data.DataType() == DataType::Float(64)) {
        return CheckInFP16Range<double>(bytes, size_1d);
      } else if (data.DataType() == DataType::Int(8)) {
        return CheckInFP16Range<std::int8_t>(bytes, size_1d);
      } else if (data.DataType() == DataType::Int(16)) {
        return CheckInFP16Range<std::int16_t>(bytes, size_1d);
      } else if (data.DataType() == DataType::Int(32)) {
        return CheckInFP16Range<std::int32_t>(bytes, size_1d);
      } else if (data.DataType() == DataType::Int(64)) {
        return CheckInFP16Range<std::int64_t>(bytes, size_1d);
      }
      return false;
    };

    for (const Expr& arg : args) {
      auto sinfo = GetStructInfo(arg);
      auto constant = arg.as<ConstantNode>();
      auto tuple = arg.as<TupleNode>();

      if (!IsNestedTensor(arg) || is_fp16(sinfo) || (constant && is_in_fp16_range(constant)) ||
          (tuple && AllFP16Castable(tuple->fields))) {
        continue;
      } else {
        return false;
      }
    }

    return true;
  }

  void CastIfFp16Only(const Var& var) {
    ICHECK(builder_->CurrentBlockIsDataFlow());
    // Get the current remapped var
    Var cur_var = GetRemapped(var);
    // Store the tensors that are fp16 only to fp16
    auto it = only_fp16_map_->find(var);
    if (it == only_fp16_map_->end()) return;
    // Get the to dtype, cast to fp16 if the var is fp16 only, otherwise do nothing
    auto fcombine = [](const String& from, const String& required) -> String {
      return required == "float16" ? required : from;
    };
    NType from = NTypeFrom(cur_var);
    NType to = CombineNestedMsg<String>(from, it->second, fcombine);
    Expr rewrite = RewriteExpr(cur_var, to);
    // If cur_var is not rewritten, we don't need to emit a new var
    if (!rewrite.same_as(cur_var)) {
      // Emit a new var, and update the var remap
      var_remap_[var->vid] = builder_->Emit(rewrite);
    }
  }

  Expr VisitVar_(const Var& var) {
    // We rewrite the remapped var to the original dtype
    auto it = var_remap_.find(var->vid);
    if (it != var_remap_.end()) {
      return RewriteExpr(it->second, NTypeFrom(var));
    }
    return var;
  }

  Expr VisitExpr_(const VarNode* op) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      return ExprMutator::VisitExpr_(op);
    }
    return VisitVar_(GetRef<Var>(op));
  }

  Var VisitVarDef(const Var& var) { return GetRemapped(var); }

  void VisitBinding(const Binding& binding) {
    ExprMutator::VisitBinding(binding);
    if (!builder_->CurrentBlockIsDataFlow()) return;
    CastIfFp16Only(binding->var);
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      ExprMutator::VisitBinding_(binding, call_node);
      return;
    }
    auto policy = GetMixedPrecisionInfo(call_node);
    if (policy == -1) {
      // not an op call
      ExprMutator::VisitBinding_(binding, call_node);
      return;
    }
    // var = Call(op)
    const auto* op_node = call_node->op.as<OpNode>();
    ICHECK(op_node != nullptr);
    Op op = GetRef<Op>(op_node);
    if (wrap_param_op.same_as(op)) {
      // wrap_param
      ReEmitBinding(binding, call_node->args[0]);
      return;
    }

    Call new_call = GetRef<Call>(call_node);

    // We first to remap the args to the current vars according to the var_remap_
    new_call.CopyOnWrite()->args = RemapArgs(new_call->args);

    // Then we rewrite the args according to the policy
    std::optional<DataType> opt_new_dtype = std::nullopt;

    if (policy == kAlways) {
      opt_new_dtype = fp16_;
      auto attr_map = Op::GetAttrMap<FInferMixedPrecision>("FInferMixedPrecision");
      ICHECK(attr_map.count(op));
      new_call = attr_map[op](new_call, output_dtype_);
    } else if (policy == kFollow) {
      opt_new_dtype = AllFP16Castable(new_call->args) ? fp16_ : fp32_;
    } else if (policy == kNever) {
      // An upstream operation may have changed the datatype of the
      // arguments.  Because this operation must be provided with
      // exactly the same dtype as it previously had, it may require a
      // cast back to the original datatype.

      if (!new_call->args.same_as(call_node->args)) {
        Array<Expr> new_typed_args;
        for (size_t i = 0; i < call_node->args.size(); i++) {
          auto arg = new_call->args[i];
          auto old_ntype = NTypeFrom(call_node->args[i]);
          new_typed_args.push_back(RewriteExpr(arg, old_ntype));
        }
        new_call.CopyOnWrite()->args = new_typed_args;
      }

    } else {
      LOG(FATAL) << "Unsupported TMixedPrecisionPolicy: " << policy;
    }

    Expr new_value = new_call;
    if (opt_new_dtype) {
      auto new_dtype = opt_new_dtype.value();
      new_call.CopyOnWrite()->args = RewriteArgs(new_call->args, new_dtype);
      new_call.CopyOnWrite()->struct_info_ = NullOpt;

      new_value = builder_->Normalize(Call(new_call));

      if (!binding->var->IsInstance<DataflowVarNode>()) {
        // Non-Dataflow var: store the tensors to the original dtype
        new_value = RewriteExpr(new_value, NTypeFrom(binding->var));
      } else if (policy == kAlways && binding->var->IsInstance<DataflowVarNode>()) {
        // kAlways: store the tensors to fp16
        // But non-dataflow vars will be stored to the original dtype anyway (see above)
        new_value = RewriteExpr(new_value, NTypeFrom(new_value, new_dtype));
      }
    }

    ReEmitBinding(binding, builder_->Normalize(new_value));
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple_node) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      ExprMutator::VisitBinding_(binding, tuple_node);
      return;
    }
    ObjectPtr<TupleNode> new_tuple = make_object<TupleNode>(*tuple_node);
    new_tuple->fields = std::move(RemapArgs(tuple_node->fields));
    new_tuple->struct_info_ = NullOpt;
    Expr new_value = builder_->Normalize(Tuple(new_tuple));
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      // Global var: store the tensors to the original dtype
      NType to = NTypeFrom(binding->var);
      new_value = RewriteExpr(new_value, to);
    }
    ReEmitBinding(binding, builder_->Normalize(new_value));
  }

  void VisitBinding_(const VarBindingNode* binding,
                     const TupleGetItemNode* tuple_get_item_node) final {
    if (!builder_->CurrentBlockIsDataFlow()) {
      // We don't need to rewrite the tuple_get_item in dataflow block
      ExprMutator::VisitBinding_(binding, tuple_get_item_node);
      return;
    }
    ObjectPtr<TupleGetItemNode> new_tuple_get_item =
        make_object<TupleGetItemNode>(*tuple_get_item_node);
    new_tuple_get_item->tuple = RemapArgs({tuple_get_item_node->tuple})[0];
    new_tuple_get_item->struct_info_ = NullOpt;
    Expr new_value = TupleGetItem(new_tuple_get_item);
    if (!binding->var->IsInstance<DataflowVarNode>()) {
      // Global var: store the tensors to the original dtype
      NType to = NTypeFrom(binding->var);
      new_value = RewriteExpr(new_value, to);
    }
    ReEmitBinding(binding, builder_->Normalize(new_value));
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) {
    builder_->BeginDataflowBlock();
    // prepare local versions of params here, if they are fp16 expected only
    for (auto param : params_) {
      CastIfFp16Only(param);
    }
    for (auto binding : block->bindings) {
      this->VisitBinding(binding);
    }
    for (auto param : params_) {
      // remove the local version of params
      auto it = var_remap_.find(param->vid);
      if (it != var_remap_.end()) {
        var_remap_.erase(it);
      }
    }
    return builder_->EndBlock();
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    params_ = op->params;
    return ExprMutator::VisitExpr_(op);
  }

  const VarDTypeMap* only_fp16_map_;

  DataType fp16_ = DataType(DataType::TypeCode::kFloat, 16, 1);
  DataType fp32_ = DataType(DataType::TypeCode::kFloat, 32, 1);
  DataType output_dtype_;
  Array<Var> params_;
  std::unordered_set<std::string> fp16_input_names_;

  const Op& wrap_param_op = Op::Get("relax.wrap_param");
};

Expr ToMixedPrecision(const Function& f, const DataType& out_dtype,
                      Optional<Array<String>> fp16_input_names) {
  VarDTypeMap only_fp16_map = std::move(DTypeDecisionCollector::Collect(f, out_dtype));
  std::unordered_set<std::string> fp16_input_names_set;
  if (fp16_input_names) {
    fp16_input_names_set.insert(fp16_input_names.value().begin(), fp16_input_names.value().end());
  }
  ToMixedPrecisionRewriter mutator(&only_fp16_map, out_dtype, fp16_input_names_set);
  return mutator(f);
}

namespace transform {

Pass ToMixedPrecision(const DataType& out_dtype, Optional<Array<String>> fp16_input_names) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ToMixedPrecision(f, out_dtype, fp16_input_names));
      };
  return CreateFunctionPass(pass_func, 0, "ToMixedPrecision", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ToMixedPrecision").set_body_typed(ToMixedPrecision);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
