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
 *
 * \file to_mixed_precision.cc
 * \brief Automatic mixed floating point precision for relay graphs. i.e. turn a graph into fp16.
 *
 */

#include <tvm/ir/attrs.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/object.h>

#include <utility>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

TVM_REGISTER_PASS_CONFIG_OPTION("relay.ToMixedPrecision.keep_orig_output_dtype", Bool);
// A callable which hashes std::pair
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    auto h1 = std::hash<T1>()(pair.first);
    auto h2 = std::hash<T2>()(pair.second);

    // Use boost's combine_hash strategy
    return h1 ^ (h1 + 0x9e3779b9 + (h2 << 6) + (h2 >> 2));
  }
};

// MIXED_PRECISION_ALWAYS ops should always be done in lower precision due to the speed and memory
// savings. MIXED_PRECISION_FOLLOW ops can be done in lower precision but don't have speedups to
// justify a cast. MIXED_PRECISION_NEVER colored ops should not be done in lower precision due to
// numerical reasons.
enum MixedTypeConversionCategory : int {
  MIXED_PRECISION_ALWAYS = 0,
  MIXED_PRECISION_FOLLOW = 1,
  MIXED_PRECISION_NEVER = 2
};

// A map of a parent node and a wanted dtype to existing nodes casted to the wanted dtype
using CachedCastNodes = std::unordered_map<std::pair<const ExprNode*, DataType>, Expr, pair_hash>;

// Return array is of type : [MixedTypeConversionCategory (int), String, String]
// The fields are          : [ConversionCategory, accumulation_datatype, output_datatype]
// Call is a call node, DataType is the mixed precision type
using FTVMMixedPrecisionConversionType = runtime::TypedPackedFunc<Array<ObjectRef>(
    const Call& call_node, const std::string& target_dtype_str)>;

/*! \brief This class transforms the given relay module into a version where
 * as many operations as possible operate in the target mixed precision dtype.
 *
 * Input : A Relay module with operations registered with FTVMMixedPrecisionConversionType
 *         functions. These describe when and how the operations will be transformed
 *         into the target precision dtype.
 *
 * Output : A Relay module with some operations transformed according to the below
 *          methodology.
 *
 * Methodology :
 *      1) Each relay Op is either of conversion category ALWAYS, FOLLOW, NEVER
 *         defined by the associated FTVMMixedPrecisionConversionType function.
 *         If an operation is not registered, it by default is assumed to be
 *         FOLLOW.
 *      2) ALWAYS operations always convert the input floating point args into
 *         the target mixed precision dtype. FOLLOW Ops will convert the input
 *         floating point args back into FP32 unless all floating point args
 *         are in the target mixed precision dtypes. NEVER ops will always cast
 *         inputs back into FP32.
 *      3) Each ALWAYS Op, and FOLLOW Op with mixed precision dtype arguments
 *         also have an associated accumulation_dtype and output_dtype which
 *         describe whether a larger dtype is used to accumulate the results
 *         of the operation. The output_dtype meanwhile describes the dtype
 *         most Ops should use from this accumulator.
 */
class MixedPrecisionPass : public MixedModeMutator {
 private:
  /*! \brief A cache of nodes + target dtype to a cast version of the node with target dtype. */
  CachedCastNodes cast_nodes_cache_;

  /*! \brief The target datatype we want to convert to e.g. FP16 */
  const DataType mixed_precision_type_;

  /*! \brief Map of Ops with no associated FTVMMixedPrecisionConversionType to the times they were
   * encountered. Used for emitting warnings on missing ops in the pass.
   */
  std::unordered_map<std::string, int> missing_ops_;
  const RelayExprNode* root_;
  std::vector<DataType> original_dtype_;
  bool keep_orig_output_dtype_;

  Attrs GetNewAttrs(const CallNode* call, const DataType& accumulation_dtype) const {
    /* If the accumulation dtype is in the attributes make a copy and mutate the field. */
    Attrs cur_attrs = call->attrs;
    if (cur_attrs.get() != nullptr) {
      // TODO(AndrewZhaoLuo): Figure out a better way to do this
      // modify output_dtype attributes (accumulation dtypes for ops)
      if (auto attrs = cur_attrs.as<Conv1DAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<Conv1DTransposeAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<Conv2DAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<Conv2DTransposeAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<Conv2DWinogradAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<Conv2DWinogradNNPACKWeightTransformAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<DeformableConv2DAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<Conv3DAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<Conv3DTransposeAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<Conv3DWinogradAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<DenseAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = cur_attrs.as<BatchMatmulAttrs>()) {
        return ModifyAttrsOutputDType(attrs, accumulation_dtype);
      }

      // modify dtype attributes (creating new tensors of type dtype)
      if (auto attrs = cur_attrs.as<InitOpAttrs>()) {
        return ModifyAttrsDType(attrs, accumulation_dtype);
      }
    }

    return cur_attrs;
  }

  template <typename T>
  Attrs ModifyAttrsOutputDType(const T* attrs, const DataType& accumulation_dtype) const {
    /*
     Helper template to modify relevant attributes with out_dtype type.
     These represent accumulation dtypes for some operations e.g.
     conv2d might take in fp16 and give a fp32 result.
     Attrs is const because we get it as a const.
     */
    DataType cur_type = (attrs->out_dtype);
    ObjectPtr<T> new_attrs = make_object<T>(*attrs);
    if (cur_type.is_float() || cur_type.is_bfloat16() || cur_type.is_void()) {
      new_attrs->out_dtype = accumulation_dtype;
    }
    return Attrs(new_attrs);
  }

  template <typename T>
  Attrs ModifyAttrsDType(const T* attrs, const DataType& accumulation_dtype) const {
    /*
     Helper template to modify relevant attributes with dtype type.
     This determines the output dtype for some ops. For example
     zeros creates a tensor of zeros of the specified dtype.
     Attrs is const because we get it as a const.
    */
    DataType cur_type = (attrs->dtype);
    ObjectPtr<T> new_attrs = make_object<T>(*attrs);
    if (cur_type.is_float() || cur_type.is_bfloat16() || cur_type.is_void()) {
      new_attrs->dtype = accumulation_dtype;
    }
    return Attrs(new_attrs);
  }

  Type GetType(const Expr& expr) const {
    // The expression has not been changed AND it's existing type
    // is known to still be valid. (See special handling for tuples etc
    // below for where we null out checked_type_ when we can not
    // sure it is still valid.
    Type checked_type = expr->checked_type_;
    if (checked_type.defined()) {
      return checked_type;
    }

    // This also populates the checked_type_ field for expr
    return transform::InferTypeLocal(expr);
  }

  bool IsMixedPrecisionType(const Type& t, bool ignore_non_float = false) const {
    /* Returns whether t is a type with only target mixed precision type elements.
       If ignore_non_float, then ignore non-floating types.
     */
    if (const TensorTypeNode* tensor_type = t.as<TensorTypeNode>()) {
      bool is_supported_floating_point_type =
          (tensor_type->dtype).is_float() || (tensor_type->dtype).is_bfloat16();
      return (ignore_non_float && !is_supported_floating_point_type) ||
             tensor_type->dtype == mixed_precision_type_;
    } else if (const TupleTypeNode* tuple_type = t.as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        if (!IsMixedPrecisionType(t, ignore_non_float)) return false;
      }
      return true;
    } else {
      LOG(FATAL) << "Unsupported type " << t << " we don't know how to handle";
      return false;
    }
  }

  Expr CachedCast(const Expr& expr, const DataType& expr_dtype, const DataType& wanted_dtype) {
    /* Cast tensor to the wanted datatype, returning a cached version if it's already been done. */

    // If this is not a floating point type, do not cast. E.g. it might be an integer
    if (!(expr_dtype.is_float() || expr_dtype.is_bfloat16())) {
      return expr;
    }

    if (expr_dtype == wanted_dtype) {
      return expr;
    }

    const ExprNode* expr_node = expr.as<ExprNode>();
    CHECK(expr_node) << "Non-expression node found in cast: " << expr;

    // Use cached result if possible.
    auto search = cast_nodes_cache_.find({expr_node, wanted_dtype});
    if (search != cast_nodes_cache_.end()) {
      return search->second;
    }

    Expr result = Cast(expr, wanted_dtype);
    cast_nodes_cache_[{expr_node, wanted_dtype}] = result;

    // Reverse the cache result, e.g. if we want to reverse the cast simply point to original node
    const ExprNode* new_expr_node = result.as<ExprNode>();
    cast_nodes_cache_[{new_expr_node, expr_dtype}] = expr;
    return result;
  }

  Expr CastArg(const Expr& expr, const Type& expr_type, const DataType& wanted_dtype) {
    /* Helper for casting arguments to call_nodes handling all relevant cases. */
    if (const TensorTypeNode* tensor_type = expr_type.as<TensorTypeNode>()) {
      return CachedCast(expr, tensor_type->dtype, wanted_dtype);
    } else if (const TupleTypeNode* tuple_type = expr_type.as<TupleTypeNode>()) {
      Array<Expr> new_expr;
      bool all_same = true;
      for (size_t i = 0; i < (tuple_type->fields).size(); i++) {
        Expr tuple_element = GetField(expr, i);
        Type tuple_element_dtype = (tuple_type->fields)[i];
        Expr casted_element = CastArg(tuple_element, tuple_element_dtype, wanted_dtype);
        new_expr.push_back(casted_element);
        all_same &= casted_element.same_as(tuple_element);
      }
      return all_same ? expr : Tuple(new_expr);
    }
    CHECK(0) << "Unsupported type " << expr_type << " we don't know how to cast for arguments!";
    return expr;
  }

  std::pair<Array<Expr>, Array<Type>> CastAllArgs(const Array<Expr>& cur_args,
                                                  const Array<Type>& cur_arg_types,
                                                  const DataType& wanted_dtype) {
    Array<Expr> new_args;
    Array<Type> new_arg_types;
    for (size_t i = 0; i < cur_args.size(); i++) {
      Expr cur_arg = cur_args[i];
      Type cur_arg_type = cur_arg_types[i];
      Expr new_arg = CastArg(cur_arg, cur_arg_type, wanted_dtype);
      Type new_arg_type = GetType(new_arg);
      new_args.push_back(new_arg);
      new_arg_types.push_back(new_arg_type);
    }
    return {new_args, new_arg_types};
  }

 public:
  using MixedModeMutator::VisitExpr_;

  explicit MixedPrecisionPass(Expr base, bool keep_orig_output_dtype,
                              DataType mixed_precision_type = DataType::Float(16))
      : MixedModeMutator(),
        mixed_precision_type_(mixed_precision_type),
        root_(Downcast<Function>(base)->body.get()),
        keep_orig_output_dtype_(keep_orig_output_dtype) {
    if (keep_orig_output_dtype_) {
      if (root_->IsInstance<tvm::relay::TupleNode>()) {
        const TupleTypeNode* tuple_type = (root_->checked_type_).as<TupleTypeNode>();
        for (Type t : tuple_type->fields) {
          const TensorTypeNode* tensor_type = t.as<TensorTypeNode>();
          original_dtype_.push_back(tensor_type->dtype);
        }
      } else if (root_->IsInstance<tvm::relay::CallNode>()) {
        original_dtype_.push_back((root_->checked_type_).as<TensorTypeNode>()->dtype);
      }
    }
    if (!(mixed_precision_type_.is_float() || mixed_precision_type_.is_bfloat16())) {
      LOG(FATAL) << "Only support IEEE floating point mixed precision types and bfloat16, but got "
                 << mixed_precision_type_;
    }
  }

  Expr Rewrite_(const CallNode* pre_call_node, const Expr& post) final {
    const CallNode* post_call_node = post.as<CallNode>();
    CHECK(post_call_node) << "Expected a CallNode, but got " << post;

    Expr cur_op = post_call_node->op;

    // TODO(AndrewZhaoLuo): Support ADTs
    // Relay's algebraic data types are not supported yet.
    ICHECK(!cur_op.as<GlobalVarNode>()       // used to declare functions for recursion
           && !cur_op.as<ConstructorNode>()  // constructing ADT types
           && !cur_op.as<VarNode>())         // used for calling recursive functions
        << "Algebraic Data Types (ADT) are not supported yet for mixed precision pass.";

    // Get info on the operation being called:
    // conversion category (int), accumulation dtype (str), output dtype (str)
    MixedTypeConversionCategory initial_category;
    DataType accumulation_dtype, output_dtype;
    if (cur_op.as<FunctionNode>()) {
      // Avoid messing with functions to avoid changing signature
      initial_category = MIXED_PRECISION_NEVER;
      accumulation_dtype = DataType::Float(32);
      output_dtype = DataType::Float(32);
    } else if (cur_op.as<OpNode>()) {
      static auto attr_map =
          Op::GetAttrMap<FTVMMixedPrecisionConversionType>("FTVMMixedPrecisionConversionType");
      Op op = Downcast<Op>(cur_op);
      if (attr_map.count(op)) {
        // Calculate the conversion category and dtypes from registered attribute.
        FTVMMixedPrecisionConversionType func = attr_map[op];
        Array<ObjectRef> op_descriptor =
            func(GetRef<Call>(pre_call_node), DLDataType2String(mixed_precision_type_));
        ICHECK(op_descriptor.size() == 3)
            << "got the wrong number of returned arguments (expected 3 got " << op_descriptor.size()
            << ") from FTVMMixedPrecisionConversionType for " << AsText(op, false);

        int64_t op_conversion_type = Downcast<Integer>(op_descriptor[0])->value;
        initial_category = static_cast<MixedTypeConversionCategory>(op_conversion_type);
        accumulation_dtype = DataType(String2DLDataType(Downcast<String>(op_descriptor[1])));
        output_dtype = DataType(String2DLDataType(Downcast<String>(op_descriptor[2])));
      } else {
        missing_ops_[op->name] += 1;

        // If not registered, by default assume is a generic FOLLOW operation.
        initial_category = MIXED_PRECISION_FOLLOW;
        accumulation_dtype = mixed_precision_type_;
        output_dtype = mixed_precision_type_;
      }
    } else {
      LOG(FATAL) << "Unsupported op type in CallNode: " << pre_call_node->op;
    }

    // First check if all the new mutated args are in lower precision form
    Array<Type> cur_arg_types;
    bool all_args_mixed_type_compatible = true;
    for (Expr arg : post_call_node->args) {
      Type cur_arg_type = GetType(arg);
      cur_arg_types.push_back(cur_arg_type);

      if (initial_category == MIXED_PRECISION_FOLLOW && all_args_mixed_type_compatible) {
        // We can cast Vars and Constants to the right types so don't care about the types.
        bool is_mixed_type_compatible = IsMixedPrecisionType(cur_arg_type, true) ||
                                        arg->IsInstance<VarNode>() ||
                                        arg->IsInstance<ConstantNode>();
        all_args_mixed_type_compatible &= is_mixed_type_compatible;
      }
    }

    // Determine the final category we want for conversion
    MixedTypeConversionCategory final_category = initial_category;
    if (initial_category == MIXED_PRECISION_FOLLOW) {
      final_category =
          all_args_mixed_type_compatible ? MIXED_PRECISION_ALWAYS : MIXED_PRECISION_NEVER;
    }

    // Create the new arguments to the call.
    DataType wanted_arg_dtypes =
        final_category == MIXED_PRECISION_ALWAYS ? mixed_precision_type_ : DataType::Float(32);
    auto call_args_and_types = CastAllArgs(post_call_node->args, cur_arg_types, wanted_arg_dtypes);
    Array<Expr> new_args = call_args_and_types.first;
    Array<Type> new_arg_types;

    if (pre_call_node->op.as<FunctionNode>()) {
      // Function Nodes don't store type info in the Call, it should be a []
      new_arg_types = pre_call_node->type_args;
    } else {
      new_arg_types = call_args_and_types.second;
    }

    // Finally create the new attributes.
    if (final_category == MIXED_PRECISION_ALWAYS) {
      Attrs new_attrs = GetNewAttrs(pre_call_node, accumulation_dtype);
      Expr output = Call(cur_op, new_args, new_attrs, new_arg_types, pre_call_node->span);
      if (accumulation_dtype != output_dtype) {
        output = CastArg(output, GetType(output), output_dtype);
      }
      if (pre_call_node == root_ && keep_orig_output_dtype_) {
        if (original_dtype_[0] != output_dtype) {
          output = CastArg(output, GetType(output), original_dtype_[0]);
        }
      }
      return output;
    }

    return Call(cur_op, new_args, pre_call_node->attrs, new_arg_types, pre_call_node->span);
  }

  Expr Rewrite_(const TupleGetItemNode* pre, const Expr& post) {
    // The old checked type in the expression may not be valid so clear it
    post->checked_type_ = Type(nullptr);
    return post;
  }

  Expr Rewrite_(const TupleNode* pre, const Expr& post) {
    // The old checked type in the expression may not be valid so clear it
    post->checked_type_ = Type(nullptr);
    if (pre == root_ && keep_orig_output_dtype_) {
      Array<Expr> new_expr;
      bool all_same = true;
      for (size_t i = 0; i < original_dtype_.size(); i++) {
        Expr output_element = GetField(post, i);
        Expr casted_element;
        auto output_element_type = transform::InferTypeLocal(output_element);
        casted_element = CastArg(output_element, output_element_type, original_dtype_[i]);
        new_expr.push_back(casted_element);
        all_same &= casted_element.same_as(output_element);
      }
      if (!all_same) {
        return Tuple(new_expr);
      }
    }
    return post;
  }

  Expr VisitExpr_(const FunctionNode* func) final {
    // Erase the ret_type annotation and let the normal pass recalculate
    const_cast<FunctionNode*>(func)->ret_type = Type(nullptr);
    return ExprMutator::VisitExpr_(func);
  }

  Expr VisitExpr_(const LetNode* op) final {
    // First convert as much of the bound computation to lower precision as possible
    Expr value = this->Mutate(op->value);

    // Then rewrite the var type and associated expression
    Var var = Downcast<Var>(this->Mutate(op->var));
    VarNode* mutable_var = const_cast<VarNode*>((op->var).as<VarNode>());
    mutable_var->type_annotation = GetType(value);
    mutable_var->checked_type_ = mutable_var->type_annotation;

    // Mutate body last as it may depend on previous results
    Expr body = this->Mutate(op->body);
    return Let(var, value, body, op->span);
  }

  // To access map of ops not registered for error reporting
  friend Expr ToMixedPrecision(const Expr& expr, bool keep_orig_output_dtype,
                               const DataType& mixed_precision_type, int missing_op_mode);
};

Expr ToMixedPrecision(const Expr& expr, bool keep_orig_output_dtype,
                      const DataType& mixed_precision_type, int missing_op_mode) {
  /*
  missing_op_mode:

  0: Does not allow any missing ops. Will throw errors and terminate the pass when encountering any.
  1: Allow missing ops but throw warnings.
  2: Allow missing ops and silently ignore them.
  */
  ICHECK(missing_op_mode >= 0 && missing_op_mode <= 2)
      << " missing_op_mode must be either 0, 1, or 2 got " << missing_op_mode;

  MixedPrecisionPass converter =
      MixedPrecisionPass(expr, keep_orig_output_dtype, mixed_precision_type);
  auto result = converter.Mutate(expr);

  for (auto it = converter.missing_ops_.begin();
       missing_op_mode != 2 && it != converter.missing_ops_.end(); it++) {
    std::string op_name = it->first;
    int appear_count = it->second;

    LOG(WARNING) << "Op \"" << op_name << "\" not registered "
                 << "FTVMMixedPrecisionConversionType appears " << appear_count
                 << " times in graph.";
  }

  if (converter.missing_ops_.size() != 0 && missing_op_mode == 0) {
    CHECK(0) << "Missing ops were found!";
  }
  return result;
}

namespace transform {

Pass ToMixedPrecision(DataType mixed_precision_type, int missing_op_mode) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        bool keep_orig_output_dtype = false;
        keep_orig_output_dtype = pc->GetConfig("relay.ToMixedPrecision.keep_orig_output_dtype",
                                               Bool(keep_orig_output_dtype))
                                     .value();
        return Downcast<Function>(
            ToMixedPrecision(f, keep_orig_output_dtype, mixed_precision_type, missing_op_mode));
      };
  return CreateFunctionPass(pass_func, 0, "ToMixedPrecision", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ToMixedPrecision").set_body_typed(ToMixedPrecision);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
