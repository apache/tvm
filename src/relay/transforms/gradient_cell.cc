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
 * \file gradient_cell.cc
 *
 * \brief Convert all tensors to a Gradient Cell
 * 
 * This pass delays or removes memory allocation by converting tensors into 
 * GradCell, an algebraic data type defined in gradient.rly
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

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/transform.h>
#include "let_list.h"

namespace tvm {
namespace relay {

/*!
* \brief Get constructor of GradCell TypeDef with name_hint
*
* module must have TypeDefinition of GradCell (defined in gradient.rly)
*/
Constructor getGradCellConstructor(IRModule module, std::string name_hint) {
  TypeData gradCell = module->LookupTypeDef("GradCell");
  for (Constructor c : gradCell->constructors) {
    if (name_hint.compare(c->name_hint) == 0) {
      return c;
    }
  }

  LOG(FATAL) << "Constructor " << name_hint << "not found in GradCell typedata.";
  throw std::runtime_error("Constructor not found in GradCell typedata");
}

/*!
* \brief Visitor to wrap inputs
*/
class InputVisitor: public ExprFunctor<Expr(const Expr&, const Type&)> {
 public:
  explicit InputVisitor(IRModule module): module_(module) {}

  Expr wrapExpr(const Expr expr, const Type& type) {
    if (type.as<TensorTypeNode>()) {
      return CallNode::make(getGradCellConstructor(module_, "Raw"),
                          {expr}, Attrs(), {type});
    } else if (auto* type_anno = type.as<TupleTypeNode>()) {
      tvm::Array<Expr> fields;
      for (size_t i = 0; i < type_anno->fields.size(); i++) {
        const Type& t = type_anno->fields[i];
        fields.push_back(this->VisitExpr(TupleGetItemNode::make(expr, i), t));
      }
      Expr tuple = TupleNode::make(fields);
      return tuple;
    }

    return expr;
  }

  Expr VisitExpr_(const VarNode* op, const Type& t) final {
    std::cout << op->type_annotation << std::endl;
    return wrapExpr(GetRef<Var>(op), op->type_annotation);
  }

  Expr VisitExpr_(const TupleGetItemNode* op, const Type& t) final {
    return wrapExpr(GetRef<TupleGetItem>(op), t);
  }
 private:
  IRModule module_;
};

/*!
* \brief Visitor to unwrap output
*/
class OutputVisitor: public ExprFunctor<Expr(const Expr&, const Type&)> {
 public:
  explicit OutputVisitor(IRModule module): module_(module) {}

  Expr unwrapExpr(const Expr expr, const Type& type) {
    if (auto* type_call = type.as<TypeCallNode>()) {
      if (type_call->func.same_as(module_->GetGlobalTypeVar("GradCell"))) {
        return CallNode::make(module_->GetGlobalVar("FromGradCell"), {expr});
      }
      return expr;
    } else if (auto* type_anno = type.as<TupleTypeNode>()) {
      tvm::Array<Expr> fields;
      for (size_t i = 0; i < type_anno->fields.size(); i++) {
        const Type& t = type_anno->fields[i];
        fields.push_back(this->VisitExpr(TupleGetItemNode::make(expr, i), t));
      }
      Expr tuple = TupleNode::make(fields);
      return tuple;
    }

    return expr;
  }

  Expr VisitExpr_(const CallNode* op, const Type& t) final {
    return unwrapExpr(GetRef<Call>(op), t);
  }

  Expr VisitExpr_(const TupleGetItemNode* op, const Type& t) final {
    return unwrapExpr(GetRef<TupleGetItem>(op), t);
  }
 private:
  IRModule module_;
};

class GradientCellTransform: public ExprMutator, public TypeMutator {
 public:
  explicit GradientCellTransform(IRModule module):
    module_(module) {
      module_->ImportFromStd("gradient.rly");
    }

  /*!
  * \brief apply GradientCell transformation and wrap function
  * so that function type stays the same
  * 
  * input/output types should only be a combination of TupleTypes and TensorTypes
  */
  Expr transform(const Expr& e) {
    auto* f = (e).as<FunctionNode>();
    auto* transformed = this->Mutate(e).as<FunctionNode>();

    if (e.same_as(GetRef<Function>(transformed))) {
      return GetRef<Function>(transformed);
    }

    // wrap inputs of Tensor type using InputVisitor class
    tvm::Array<Expr> args;
    for (Var var : f->params) {
      Expr wrappedInput = InputVisitor(module_).VisitExpr(var, var->checked_type());
      args.push_back(wrappedInput);
    }
    Expr transformedExpr = CallNode::make(GetRef<Function>(transformed), args);

    // unwrap outputs of GradCell type into Tensor type using OutputVisitor class
    Expr tensorOutput = OutputVisitor(module_).VisitExpr(transformedExpr, transformed->ret_type);
    return Function(f->params, tensorOutput, f->ret_type, Array<TypeVar>());
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    return CallNode::make(getGradCellConstructor(module_, "Raw"),
                          {GetRef<Constant>(op)}, Attrs(), {op->checked_type()});
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    // optimize operators
    if (auto* op = (call_node->op).as<OpNode>()) {
      if (op->name.compare("add") == 0 && call_node->args.size() == 2 &&
          AlphaEqual(call_node->args[0]->checked_type(), call_node->args[1]->checked_type())) {
        // case: "add" between two tensors of the same size
        const auto addFunc = module_->GetGlobalVar("AddGradCell");
        tvm::Array<Expr> args;
        // create add function
        Type paramType = call_node->args[0]->checked_type();
        tvm::Array<Var> params = {VarNode::make("lhs", paramType),
                                  VarNode::make("rhs", paramType)};
        Expr callAdd = CallNode::make(Op::Get("add"), {params[0], params[1]});
        Expr addTensorsFunc = Function(params, callAdd, paramType,
                                                Array<TypeVar>());

        // pass add function and tensors into arguments
        args.push_back(addTensorsFunc);
        for (Expr expr : call_node->args) {
          args.push_back(VisitExpr(expr));
        }
        return CallNode::make(addFunc, args, Attrs(), {paramType});
      } else if (op->name.compare("multiply") == 0 && call_node->args.size() == 2 &&
          AlphaEqual(call_node->args[0]->checked_type(), call_node->args[1]->checked_type())) {
        // case: "multiply" between two tensors of the same size
        const auto multFunc = module_->GetGlobalVar("MultiplyGradCell");
        // create multiply function
        tvm::Array<Expr> args;
        Type paramType = call_node->args[0]->checked_type();
        tvm::Array<Var> params = {VarNode::make("lhs", paramType),
                                  VarNode::make("rhs", paramType)};
        Expr callMultiply = CallNode::make(Op::Get("multiply"),
                                          {params[0], params[1]});
        Expr multTensorsFunc = Function(params, callMultiply, paramType,
                                                  Array<TypeVar>());

        // pass multiply function and tensors into arguments
        args.push_back(multTensorsFunc);
        for (Expr expr : call_node->args) {
          args.push_back(VisitExpr(expr));
        }
        return CallNode::make(multFunc, args, Attrs(), {paramType});
      } else if (op->name.compare("ones") == 0) {
        // ones operator, use One constructor of GradCell
        Expr func = Function({}, {ExprMutator::VisitExpr_(call_node)},
                                        {call_node->checked_type()}, {});
        return CallNode::make(getGradCellConstructor(module_, "One"),
                              {func}, Attrs(), {call_node->checked_type()});
      } else if (op->name.compare("zeros") == 0) {
        // zeros operator, use Zero constructor of GradCell
        Expr func = Function({}, {ExprMutator::VisitExpr_(call_node)},
                                        {call_node->checked_type()}, {});
        return CallNode::make(getGradCellConstructor(module_, "Zero"),
                              {func}, Attrs(), {call_node->checked_type()});
      }

      // handle other ops + zeros_like + ones_like
      // we put zeros_like and ones_like here to make use of
      // code converting the arguments of CallNode into Tensor
      const auto fromFunc = module_->GetGlobalVar("FromGradCell");
      tvm::Array<Expr> args;
      // use FromGradCell to convert args to Tensor
      for (Expr expr : call_node->args) {
        args.push_back(CallNode::make(fromFunc,
                                      {VisitExpr(expr)}, Attrs(), {expr->checked_type()}));
      }

      const Expr tensorRes = CallNode::make(call_node->op, args);

      if (op->name.compare("ones_like") == 0) {
        Expr onesFunction = Function({}, tensorRes,
                              {call_node->checked_type()}, Array<TypeVar>());
        return CallNode::make(getGradCellConstructor(module_, "One"),
                              {onesFunction}, Attrs(), {call_node->checked_type()});
      } else if (op->name.compare("zeros_like") == 0) {
        Expr zerosFunction = Function({}, tensorRes,
                              {call_node->checked_type()}, Array<TypeVar>());
        return CallNode::make(getGradCellConstructor(module_, "Zero"),
                              {zerosFunction}, Attrs(), {call_node->checked_type()});
      }
      return CallNode::make(getGradCellConstructor(module_, "Raw"), {tensorRes},
                            Attrs(), {call_node->checked_type()});
    }
    // call-> op is not a relay op
    return ExprMutator::VisitExpr_(call_node);
  }

  Type VisitType(const Type& t) final {
    return TypeMutator::VisitType(t);
  }

  Type VisitType_(const TensorTypeNode* op) {
    GlobalTypeVar gradCell = module_->GetGlobalTypeVar("GradCell");
    tvm::Array<Type> args;
    args.push_back(GetRef<TensorType>(op));
    return TypeCall(gradCell, args);
  }

 private:
  // Module
  IRModule module_;
};

Expr GradientCell(const Expr& e, IRModule mod) {
  return GradientCellTransform(mod).transform(e);
}

namespace transform {
Pass GradientCell() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(GradientCell(f, m));
    };
    return CreateFunctionPass(pass_func, 2, "GradientCell", {});
}

TVM_REGISTER_GLOBAL("relay._transform.GradientCell")
.set_body_typed(GradientCell);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
