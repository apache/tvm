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
 * This algorithm is implemented by one visitor
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/transform.h>
#include "let_list.h"

namespace tvm {
namespace relay {

class GradientCellTransform: public ExprMutator, public TypeMutator {
  public:
    explicit GradientCellTransform(IRModule module):
      module_(module)
      {}

    Expr VisitExpr_(const ConstantNode* op) final {
      return CallNode::make(getGradCellConstructor("Raw"), {GetRef<Constant>(op)}, Attrs(), {op->checked_type()});
    }

    Expr VisitExpr_(const CallNode* call_node) final {
      if (auto* op = (call_node->op).as<OpNode>()) {
        if (op->name.compare("add") == 0 && call_node->args.size() == 2 &&
            AlphaEqual(call_node->args[0]->checked_type(), call_node->args[1]->checked_type())) {
          const auto addFunc = module_->GetGlobalVar("AddGradCell");
          tvm::Array<Expr> args;

          Type paramType = call_node->args[0]->checked_type();

          tvm::Array<Var> params = {VarNode::make("lhs", paramType), VarNode::make("rhs", paramType)};
          Expr callAdd = CallNode::make(Op::Get("add"), {params[0], params[1]});

          Expr addTensorsFunc = FunctionNode::make(params, callAdd, paramType, Array<TypeVar>(), Attrs());

          args.push_back(addTensorsFunc);
          for (Expr expr: call_node->args) {
            args.push_back(VisitExpr(expr));
          }
          return CallNode::make(addFunc, args, Attrs(), {paramType});
        } else if (op->name.compare("multiply") == 0 && call_node->args.size() == 2 &&
            AlphaEqual(call_node->args[0]->checked_type(), call_node->args[1]->checked_type())) {
          const auto multFunc = module_->GetGlobalVar("MultiplyGradCell");
          tvm::Array<Expr> args;

          Type paramType = call_node->args[0]->checked_type();

          tvm::Array<Var> params = {VarNode::make("lhs", paramType), VarNode::make("rhs", paramType)};
          Expr callMultiply = CallNode::make(Op::Get("multiply"), {params[0], params[1]});

          Expr multTensorsFunc = FunctionNode::make(params, callMultiply, paramType, Array<TypeVar>(), Attrs());

          args.push_back(multTensorsFunc);
          for (Expr expr: call_node->args) {
            args.push_back(VisitExpr(expr));
          }
          return CallNode::make(multFunc, args, Attrs(), {paramType});
        } else if (op->name.compare("ones") == 0) {
          Expr func = FunctionNode::make({}, {ExprMutator::VisitExpr_(call_node)}, {call_node->checked_type()}, {}, Attrs());
          return CallNode::make(getGradCellConstructor("One"), {func}, Attrs(), {call_node->checked_type()});
        } else if (op->name.compare("zeros") == 0) {
          Expr func = FunctionNode::make({}, {ExprMutator::VisitExpr_(call_node)}, {call_node->checked_type()}, {}, Attrs());
          return CallNode::make(getGradCellConstructor("Zero"), {func}, Attrs(), {call_node->checked_type()});
        } 

        const auto fromFunc = module_->GetGlobalVar("FromGradCell");
        tvm::Array<Expr> args;
        // use FromGradCell to convert args to Tensor
        for (Expr expr: call_node->args) {
          args.push_back(CallNode::make(fromFunc, {VisitExpr(expr)}, Attrs(), {expr->checked_type()}));
        }

        const Expr tensorRes = CallNode::make(call_node->op, args);

        if (op->name.compare("ones_like") == 0) {
          Expr onesFunction = FunctionNode::make({}, tensorRes, {call_node->checked_type()}, Array<TypeVar>(), Attrs());
          return CallNode::make(getGradCellConstructor("One"), {onesFunction}, Attrs(), {call_node->checked_type()});
        } else if (op->name.compare("zeros_like") == 0) {
          Expr zerosFunction = FunctionNode::make({}, tensorRes, {call_node->checked_type()}, Array<TypeVar>(), Attrs());
          return CallNode::make(getGradCellConstructor("Zero"), {zerosFunction}, Attrs(), {call_node->checked_type()});
        }
        return CallNode::make(getGradCellConstructor("Raw"), {tensorRes}, Attrs(), {call_node->checked_type()});
      }

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
    // Constructors of gradCell datatype
    std::unordered_map<std::string, Constructor> gradCellConstructors;

    Constructor getGradCellConstructor(std::string name_hint) {
      TypeData gradCell = module_->LookupTypeDef("GradCell");
      for (Constructor c: gradCell->constructors) {
        if (name_hint.compare(c->name_hint) == 0) {
          return c;
        }
      }

      CHECK(false) << "Constructor " << name_hint << "not found in GradCell datatype.";
    }
};

Expr GradientCell(const Expr& e, IRModule mod) {
  return GradientCellTransform(mod).Mutate(e);
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

} //namespace transform

} //namespace relay
} //namespace tvm
