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
 * \file lazy_gradient_init.cc
 *
 * \brief Lazily instantiate 0-filled or 1-filled tensors.
 * This pass should be used after reverse-mode ad so that gradient tensors
 * are not instantiated until after the forward pass.
 *
 * This pass delays or removes memory allocation by converting tensors into
 * GradCell, an algebraic data type defined in gradient.rly.
 *
 * This will delay or decrease memory usage. All calls to
 * ones, ones_like, zeros, zeros_like will call the One or Zero constructor
 * of GradCell, which will not instantiate in memory until needed. All other cases result
 * in using the Raw constructor which means the tensor is instantiated in memory.
 *
 * It also overloads + and * operation which can increase performance when doing
 * operations involving tensors with values of only 0 or 1.
 *
 * Note: this pass can only be used with functions where the input/output types are
 * a combination of TupleTypes and TensorTypes
 *
 * This pass optimizes 6 ops:
 * - add
 * - multiply
 * - ones
 * - ones_like
 * - zeros
 * - zeros_like
 *
 * This pass makes use of three visitor. The most important one visits the entire function,
 * one is used for wrap inputs and one to unwrap outputs.
 *
 * For example:
 * fn: TensorType[(10,10), float32] -> TensorType[(10,10), float32]
 *
 * After this pass
 * fn: GradCell[TensorType[(10,10), float32]] -> GradCell[TensorType[(10,10), float32]]
 *
 * Thus, it is necessary to wrap this outer function so that the input/output types remain the same
 */

#include <tvm/ir/type_functor.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/feature.h>
#include <tvm/relay/transform.h>

#include "let_list.h"

namespace tvm {
namespace relay {

class LazyGradientInitializer : public ExprMutator, public TypeMutator {
 public:
  explicit LazyGradientInitializer(IRModule module) : module_(module) {
    module_->ImportFromStd("gradient.rly");
  }

  Expr WrapExpr(const Var& var, const Type& type, LetList* ll) {
    if (type.as<TensorTypeNode>()) {
      return Call(module_->GetConstructor("GradCell", "Raw"), {var}, Attrs(), {type});
    } else if (auto* type_anno = type.as<TupleTypeNode>()) {
      tvm::Array<Expr> fields;
      for (size_t i = 0; i < type_anno->fields.size(); i++) {
        const Type& t = type_anno->fields[i];
        fields.push_back(WrapExpr(ll->Push(TupleGetItem(var, i)), t, ll));
      }
      Expr tuple = Tuple(fields);
      return tuple;
    }

    return var;
  }

  Expr UnwrapExpr(const Var& var, const Type& type, LetList* ll) {
    if (auto* type_call = type.as<TypeCallNode>()) {
      if (type_call->func.same_as(module_->GetGlobalTypeVar("GradCell"))) {
        return Call(module_->GetGlobalVar("FromGradCell"), {var});
      }
      return var;
    } else if (auto* type_anno = type.as<TupleTypeNode>()) {
      tvm::Array<Expr> fields;
      for (size_t i = 0; i < type_anno->fields.size(); i++) {
        const Type& t = type_anno->fields[i];
        fields.push_back(UnwrapExpr(ll->Push(TupleGetItem(var, i)), t, ll));
      }
      Expr tuple = Tuple(fields);
      return tuple;
    }

    return var;
  }

  // Turn off memo for constant node.
  Expr VisitExpr(const Expr& e) final {
    if (e.as<ConstantNode>()) {
      return ExprFunctor::VisitExpr(e);
    } else {
      return ExprMutator::VisitExpr(e);
    }
  }

  /*!
   * \brief apply LazyGradientInit transformation and wrap function
   * so that function type stays the same
   *
   * input/output types should only be a combination of TupleTypes and TensorTypes
   */
  Expr Transform(const Expr& e) {
    auto* f = e.as<FunctionNode>();
    auto* transformed = this->Mutate(e).as<FunctionNode>();

    ICHECK(f);
    ICHECK(transformed);

    if (e.same_as(GetRef<Function>(transformed))) {
      return GetRef<Function>(transformed);
    }

    auto tensorOutput = LetList::With([&](LetList* ll) {
      // wrap inputs of Tensor type using InputVisitor class
      tvm::Array<Expr> args;
      for (const Var& var : f->params) {
        args.push_back(WrapExpr(var, var->checked_type(), ll));
      }
      Expr transformedExpr = Call(GetRef<Function>(transformed), args);
      // unwrap outputs of GradCell type into Tensor type using OutputVisitor class
      return UnwrapExpr(ll->Push(transformedExpr), transformed->ret_type, ll);
    });
    return Function(f->params, tensorOutput, f->ret_type, Array<TypeVar>());
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    return Call(module_->GetConstructor("GradCell", "Raw"), {GetRef<Constant>(op)}, Attrs(),
                {op->checked_type()});
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    if (auto op = call_node->op.as<Op>()) {
      Expr op_expr = op.value();

      if (op_expr == Op::Get("add")) {
        return CallGradCellFunction(call_node, module_->GetGlobalVar("AddGradCell"));
      }

      if (op_expr == Op::Get("multiply")) {
        return CallGradCellFunction(call_node, module_->GetGlobalVar("MultiplyGradCell"));
      }

      if (op_expr == Op::Get("ones") || op_expr == Op::Get("zeros")) {
        // ones and zeros need TensorType input
        Expr result = CallPrimitiveOp(call_node);
        Expr func = Function({}, result, {call_node->checked_type()}, Array<TypeVar>());
        // call appropriate GradCell constructor
        std::string constructor_name = op_expr == Op::Get("ones") ? "One" : "Zero";
        return Call(module_->GetConstructor("GradCell", constructor_name), {func}, Attrs(),
                    {call_node->checked_type()});
      }

      if (op_expr == Op::Get("ones_like") || op_expr == Op::Get("zeros_like")) {
        // ones_like and zeros_like need TensorType input
        Expr result = CallPrimitiveOp(call_node);
        // fn() -> T, function returns result of operation
        Expr func = Function({}, result, {call_node->checked_type()}, Array<TypeVar>());
        // call appropriate GradCell constructor
        std::string constructor_name = op_expr == Op::Get("ones_like") ? "One" : "Zero";
        return Call(module_->GetConstructor("GradCell", "One"), {func}, Attrs(),
                    {call_node->checked_type()});
      }

      // handle all other ops
      Expr result = CallPrimitiveOp(call_node);
      // wrap result with Raw constructor
      return Call(module_->GetConstructor("GradCell", "Raw"), {result}, Attrs(),
                  {call_node->checked_type()});
    }
    // not an op
    return ExprMutator::VisitExpr_(call_node);
  }

  Type VisitType(const Type& t) final { return TypeMutator::VisitType(t); }

  Type VisitType_(const TensorTypeNode* op) {
    GlobalTypeVar gradCell = module_->GetGlobalTypeVar("GradCell");
    tvm::Array<Type> args;
    args.push_back(GetRef<TensorType>(op));
    return TypeCall(gradCell, args);
  }

 private:
  // Module
  IRModule module_;

  /*!
   * \brief Convert call_node to add/multiply op to use overloaded functions for GradCell type
   */
  Expr CallGradCellFunction(const CallNode* call_node, GlobalVar overloaded_op) {
    // can only use overloaded functions if 2 arguments of same type
    if (call_node->args.size() != 2 ||
        !tvm::StructuralEqual()(call_node->args[0]->checked_type(),
                                call_node->args[1]->checked_type())) {
      Expr result = CallPrimitiveOp(call_node);
      return Call(module_->GetConstructor("GradCell", "Raw"), {result}, Attrs(),
                  {call_node->checked_type()});
    }

    tvm::Array<Expr> args;
    // create "fallback" function for overloaded function
    Type paramType = call_node->args[0]->checked_type();
    tvm::Array<Var> params = {Var("lhs", paramType), Var("rhs", paramType)};
    // use primitive op in this case
    Expr callOp = Call(call_node->op, {params[0], params[1]});
    Expr func = Function(params, callOp, paramType, Array<TypeVar>());

    // pass "fallback" function and tensors as arguments
    args.push_back(func);
    for (Expr expr : call_node->args) {
      args.push_back(VisitExpr(expr));
    }
    // return new call to overloaded function
    return Call(overloaded_op, args, Attrs(), {paramType});
  }

  /*!
   * \brief Convert calls to other ops by converting args into TensorType
   * \return call expr returning result of op
   */
  Expr CallPrimitiveOp(const CallNode* call_node) {
    const auto fromFunc = module_->GetGlobalVar("FromGradCell");
    tvm::Array<Expr> args;
    // use FromGradCell to convert args to Tensor
    for (Expr expr : call_node->args) {
      args.push_back(Call(fromFunc, {VisitExpr(expr)}, Attrs(), {expr->checked_type()}));
    }
    // result of operation
    return Call(call_node->op, args, call_node->attrs);
  }
};

Expr LazyGradientInit(const Expr& e, IRModule mod) {
  CheckFeature(e, mod, FeatureSet::All() - fGraph);
  auto ret = LazyGradientInitializer(mod).Transform(e);
  CheckFeature(ret, mod, FeatureSet::All() - fGraph);
  return ret;
}

namespace transform {
Pass LazyGradientInit() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(LazyGradientInit(f, m));
      };
  return CreateFunctionPass(pass_func, 2, "LazyGradientInit", {});
}

TVM_REGISTER_GLOBAL("relay._transform.LazyGradientInit").set_body_typed(LazyGradientInit);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
