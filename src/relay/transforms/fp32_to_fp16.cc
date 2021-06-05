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
 * \file fp32_to_fp16.cc
 * \brief Rewrite a graph into an fp16 form.
 */
#include "fp32_to_fp16.h"

#include <tvm/ir/attrs.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/object.h>

#include <utility>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

// A callable which hashes std::pair
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    auto h1 = std::hash<T1>()(pair.first);
    auto h2 = std::hash<T2>()(pair.second);

    return h1 ^ (h2 << 1);
  }
};

// A map of a parent node and a wanted dtype to existing nodes casted to the wanted dtype
using CachedCastNodes = std::unordered_map<std::pair<const ExprNode*, DataType>, Expr, pair_hash>;

// A function which maps CallNodes to their initial conversion color
using ColorFunc = std::function<FP16ConversionCategory(const CallNode*)>;

// A function which maps green CallNodes to wanted accumulation and output dtypes
using OutputDtypeFunc = std::function<FP16OpDType(const CallNode*)>;

class FP16GraphCreator : public ExprMutator {
 private:
  CachedCastNodes cast_nodes_cache;
  const ColorFunc colorer;
  const OutputDtypeFunc output_dtype_func;

  Attrs GetNewAttrs(const CallNode* call, const DataType& accumulation_dtype) const {
    /* If the accumulation dtype is in the attributes make a copy and mutate the field. */
    Attrs new_attrs = Attrs(call->attrs);
    if (new_attrs.get() != nullptr) {
      // TODO(AndrewZhaoLuo): Figure out a better way to do this
      // modify output_dtype attributes (accumulation dtypes for ops)
      if (auto attrs = new_attrs.as<Conv1DAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv1DTransposeAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv2DAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv2DTransposeAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv2DWinogradAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv2DWinogradNNPACKWeightTransformAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<DeformableConv2DAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv3DAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv3DTransposeAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<Conv3DWinogradAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<DenseAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      } else if (auto attrs = new_attrs.as<BatchMatmulAttrs>()) {
        ModifyAttrsOutputDType(attrs, accumulation_dtype);
      }

      // modify dtype attributes (creating new tensors of type dtype)
      if (auto attrs = new_attrs.as<InitOpAttrs>()) {
        ModifyAttrsDType(attrs, accumulation_dtype);
      }
    }

    return new_attrs;
  }

  template <typename T>
  void ModifyAttrsOutputDType(const T* attrs, const DataType& accumulation_dtype) const {
    /*
     Helper template to modify relevant attributes with out_dtype type.
     These represent accumulation dtypes for some operations e.g.
     conv2d might take in fp16 and give a fp32 result.
     Attrs is const because we get it as a const.
     */
    T* mutable_attrs = const_cast<T*>(attrs);
    mutable_attrs->out_dtype = accumulation_dtype;
  }

  template <typename T>
  void ModifyAttrsDType(const T* attrs, const DataType& accumulation_dtype) const {
    /*
     Helper template to modify relevant attributes with dtype type.
     This determines the output dtype for some ops. For example
     zeros creates a tensor of zeros of the specified dtype.
     Attrs is const because we get it as a const.
    */
    T* mutable_attrs = const_cast<T*>(attrs);
    mutable_attrs->dtype = accumulation_dtype;
  }

  Type GetType(const Expr& expr) const {
    auto mod = IRModule::FromExpr(expr);
    mod = transform::InferType()(mod);
    if (expr.as<FunctionNode>()) {
      return mod->Lookup("main")->checked_type();
    } else {
      return mod->Lookup("main").as<FunctionNode>()->body->checked_type();
    }
  }

  bool IsFP16Type(const Type& t, bool ignore_non_float = false) const {
    /* Returns whether t is a type with only fp16 elements.
       If ignore_non_float, then ignore non-floating types.
     */
    if (const TensorTypeNode* tensor_type = t.as<TensorTypeNode>()) {
      return (!ignore_non_float || (tensor_type->dtype).is_float()) &&
             tensor_type->dtype == DataType::Float(16);
    } else if (const TupleTypeNode* tuple_type = t.as<TupleTypeNode>()) {
      for (Type t : tuple_type->fields) {
        if (!IsFP16Type(t, ignore_non_float)) return false;
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
    if (!expr_dtype.is_float()) {
      return expr;
    }

    const ExprNode* expr_node = expr.as<ExprNode>();
    if (!expr_node) {
      LOG(FATAL) << "Non-expression node found in cast: " << expr;
    }

    // Use cached result if possible.
    auto search = cast_nodes_cache.find({expr_node, wanted_dtype});
    if (search != cast_nodes_cache.end()) {
      return search->second;
    }

    Expr result = expr_dtype == wanted_dtype ? expr : Cast(expr, wanted_dtype);
    cast_nodes_cache[{expr_node, wanted_dtype}] = result;

    // Reverse the cache result, e.g. if we want to reverse the cast simply point to original node
    const ExprNode* new_expr_node = result.as<ExprNode>();
    cast_nodes_cache[{new_expr_node, expr_dtype}] = expr;
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
    } else {
      LOG(FATAL) << "Unsupported type " << expr_type << " we don't know how to cast for arguments!";
      return expr;
    }
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
  explicit FP16GraphCreator(ColorFunc colorer, OutputDtypeFunc output_dtype_func)
      : ExprMutator(), colorer(colorer), output_dtype_func(output_dtype_func) {}

  Expr VisitExpr_(const CallNode* call_node) {
    FP16ConversionCategory initial_color = colorer(call_node);
    auto new_op = this->Mutate(call_node->op);

    // Mutate arguments to FP16 form first if possible and keep track of whether all floating point
    // tensors are in FP16 form already. This is useful for propagating color.
    Array<Expr> new_args;
    Array<Type> new_arg_types;
    bool all_args_fp16_compatible = true;
    for (Expr arg : call_node->args) {
      Expr new_arg = this->Mutate(arg);
      Type new_arg_type = GetType(new_arg);
      new_args.push_back(new_arg);
      new_arg_types.push_back(new_arg_type);

      if (all_args_fp16_compatible) {
        // We can cast Vars and Constants to the right types so don't care about the types.
        bool is_fp16_compatible = IsFP16Type(new_arg_type, true) || arg->IsInstance<VarNode>() ||
                                  arg->IsInstance<ConstantNode>();
        all_args_fp16_compatible &= is_fp16_compatible;
      }
    }

    // Determine the final color.
    FP16ConversionCategory final_color;
    if (initial_color == GRAY) {
      final_color = all_args_fp16_compatible ? GREEN : RED;
    } else {
      final_color = initial_color;
    }

    // Create the new arguments to the call.
    DataType wanted_arg_dtypes = final_color == GREEN ? DataType::Float(16) : DataType::Float(32);
    auto call_args_and_types = CastAllArgs(new_args, new_arg_types, wanted_arg_dtypes);

    Array<Expr> call_args = call_args_and_types.first;
    Array<Type> call_arg_types;
    if (call_node->op.as<FunctionNode>()) {
      // Function Nodes don't store type info in the Call, it should be a []
      call_arg_types = call_node->type_args;
    } else {
      call_arg_types = call_args_and_types.second;
    }

    // Finally create the new attributes.
    if (final_color == GREEN) {
      FP16OpDType output_dtypes = output_dtype_func(call_node);

      Attrs new_attrs = GetNewAttrs(call_node, output_dtypes.accumulation_dtype);
      Expr output = Call(new_op, call_args, new_attrs, call_arg_types, call_node->span);
      if (output_dtypes.accumulation_dtype != output_dtypes.output_dtype) {
        output = CastArg(output, GetType(output), output_dtypes.output_dtype);
      }
      return output;
    } else {
      return Call(new_op, call_args, call_node->attrs, call_arg_types, call_node->span);
    }
  }

  Expr VisitExpr_(const FunctionNode* func) final {
    // Erase the ret_type annotation and let the normal pass recalculate
    const_cast<FunctionNode*>(func)->ret_type = Type(nullptr);
    return ExprMutator::VisitExpr_(func);
  }

  Expr VisitExpr_(const LetNode* op) final {
    // First convert as much of the bound computation to FP16 as possible
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
};

Expr RewriteFp16Graph(const Expr& expr, const ColorFunc& colorer,
                      const OutputDtypeFunc& output_dtype_func) {
  FP16GraphCreator converter = FP16GraphCreator(colorer, output_dtype_func);
  return converter.Mutate(expr);
}

namespace transform {

Pass RewriteFP16() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            RewriteFp16Graph(f, DefaultFP16Colorer(), DefaultFP16OpDefinition()));
      };
  return CreateFunctionPass(pass_func, 10, "RewriteFp16", {});
}

TVM_REGISTER_GLOBAL("relay._transform.RewriteFP16").set_body_typed(RewriteFP16);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
