/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/ir/expr.cc
 * \brief The expression AST nodes of Relay.
 */
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

Constant ConstantNode::make(runtime::NDArray data) {
  NodePtr<ConstantNode> n = make_node<ConstantNode>();
  n->data = std::move(data);
  return Constant(n);
}

TVM_REGISTER_API("relay._make.Constant")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = ConstantNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ConstantNode>([](const ConstantNode *node, tvm::IRPrinter *p) {
    p->stream << "Constant(TODO)";
  });

TensorType ConstantNode::tensor_type() const {
  auto dtype = TVMType2Type(data->dtype);

  Array<tvm::Expr> shape;
  for (int i = 0; i < data->ndim; i++) {
    shape.push_back(tvm::ir::IntImm::make(HalideIR::Int(64), data->shape[i]));
  }

  return TensorTypeNode::make(shape, dtype);
}

Tuple TupleNode::make(tvm::Array<relay::Expr> fields) {
  NodePtr<TupleNode> n = make_node<TupleNode>();
  n->fields = std::move(fields);
  return Tuple(n);
}

TVM_REGISTER_API("relay._make.Tuple")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = TupleNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TupleNode>([](const TupleNode *node, tvm::IRPrinter *p) {
    p->stream << "Tuple(" << node->fields << ")";
  });

Var VarNode::make(std::string name_hint) {
  NodePtr<VarNode> n = make_node<VarNode>();
  n->name_hint = std::move(name_hint);
  return Var(n);
}

TVM_REGISTER_API("relay._make.Var")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = VarNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<VarNode>([](const VarNode *node, tvm::IRPrinter *p) {
    p->stream << "Var(" << node->name_hint << ")";
  });

GlobalVar GlobalVarNode::make(std::string name_hint) {
  NodePtr<GlobalVarNode> n = make_node<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  return GlobalVar(n);
}

TVM_REGISTER_API("relay._make.GlobalVar")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = GlobalVarNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<GlobalVarNode>([](const GlobalVarNode *node, tvm::IRPrinter *p) {
    p->stream << "GlobalVar(" << node->name_hint << ")";
  });

Param ParamNode::make(Var var, Type type) {
  NodePtr<ParamNode> n = make_node<ParamNode>();
  n->var = std::move(var);
  n->type = std::move(type);
  return Param(n);
}

TVM_REGISTER_API("relay._make.Param")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = ParamNode::make(args[0], args[1]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ParamNode>([](const ParamNode *node, tvm::IRPrinter *p) {
    p->stream << "Param(" << node->var << ", " << node->type << ")";
});

Function FunctionNode::make(tvm::Array<Param> params, Type ret_type, Expr body,
                            tvm::Array<TypeParam> type_params) {
  NodePtr<FunctionNode> n = make_node<FunctionNode>();
  n->params = std::move(params);
  n->ret_type = std::move(ret_type);
  n->body = std::move(body);
  n->type_params = std::move(type_params);
  return Function(n);
}

Type FunctionNode::fn_type() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(param->type);
  }

  return FuncTypeNode::make(param_types, this->ret_type, this->type_params, {});
}

TVM_REGISTER_API("relay._make.Function")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = FunctionNode::make(args[0], args[1], args[2], args[3]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionNode>([](const FunctionNode *node,
                                   tvm::IRPrinter *p) {
      p->stream << "FunctionNode(" << node->params << ", " << node->ret_type
                << ", " << node->body << ", " << node->type_params << ")";
});

Call CallNode::make(Expr op, Array<Expr> args, Attrs attrs,
                    Array<Type> type_args) {
  NodePtr<CallNode> n = make_node<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  return Call(n);
}

TVM_REGISTER_API("relay._make.Call")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = CallNode::make(args[0], args[1], args[2], args[3]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<CallNode>([](const CallNode *node, tvm::IRPrinter *p) {
  p->stream << "CallNode(" << node->op << ", " << node->args << ", "
    << node->attrs << ", " << node->type_args << ")";
});

Let LetNode::make(Var var, Expr value, Expr body, Type value_type) {
  NodePtr<LetNode> n = make_node<LetNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->body = std::move(body);
  n->value_type = std::move(value_type);
  return Let(n);
}

TVM_REGISTER_API("relay._make.Let")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = LetNode::make(args[0], args[1], args[2], args[3]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<LetNode>([](const LetNode *node, tvm::IRPrinter *p) {
  p->stream << "LetNode(" << node->var << ", " << node->value
    << ", " << node->body << ", " << node->value_type << ")";
});

If IfNode::make(Expr cond, Expr true_branch, Expr false_branch) {
  NodePtr<IfNode> n = make_node<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  return If(n);
}

TVM_REGISTER_API("relay._make.If").set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = IfNode::make(args[0], args[1], args[2]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<IfNode>([](const IfNode *node, tvm::IRPrinter *p) {
  p->stream << "IfNode(" << node->cond << ", " << node->true_branch
            << ", " << node->false_branch << ")";
});

}  // namespace relay
}  // namespace tvm
