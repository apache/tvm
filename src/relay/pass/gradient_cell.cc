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
 * \file gradient_node.cc
 *
 * \brief Convert all tensors to a Gradient Cell
 *
 * The algorithm is implemented by two visitor:
 * CalcDep turn an expr into a dependency graph of expr,
 * GenLet turn the dependency graph into a let list, taking only the used value.
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

    Expr VisitExpr_(const CallNode* call_node) final {
      if (auto* op = (call_node->op).as<OpNode>()) {
        if (op->name.compare("add") == 0) {
          const BaseFunc addFunc = module_->Lookup("AddGradCell");
          tvm::Array<Expr> args;

          args.push_back(Op::Get("add"));
          for (Expr expr: call_node->args) {
            args.push_back(expr);
          }

          return CallNode::make(addFunc, args);
        } else if (op->name.compare("multiply") == 0) {
          const BaseFunc multFunc = module_->Lookup("MultiplyGradCell");
          tvm::Array<Expr> args;

          args.push_back(Op::Get("multiply"));
          for (Expr expr: call_node->args) {
            args.push_back(expr);
          }

          return CallNode::make(multFunc, args);
        }
        const BaseFunc fromFunc = module_->Lookup("FromGradCell");
        GlobalTypeVar gradCellType = module_->GetGlobalTypeVar("GradCell");
        tvm::Array<Expr> args;
        // use FromGradCell to convert args to Tensor
        for (Expr expr: call_node->args) {
          tvm::Array<Expr> fromGradArgs;
          fromGradArgs.push_back(expr);
          args.push_back(CallNode::make(fromFunc, fromGradArgs));
        }
        
        return CallNode::make(call_node->op, args);
      }

      return GetRef<Call>(call_node);
    }

    Type VisitType(const Type& t) final {
      std::cout << "visittype called" << std::endl;
      return TypeMutator::VisitType(t);
    }

    Type VisitType_(const TensorTypeNode* op) {
      std::cout << "TypeTensor" << std::endl;
      GlobalTypeVar gradCell = module_->GetGlobalTypeVar("GradCell");
      tvm::Array<Type> args;
      args.push_back(GetRef<TensorType>(op));
      
      return TypeCall(gradCell, args);
    }
    
  private:
    // Module
    IRModule module_;

    // memo which Expr visited 
    std::unordered_set<Type, ObjectHash, ObjectEqual> visited_;  
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