/*!
 *  Copyright (c) 2018 by Contributors
 * \file type.cc
 * \brief The type system AST nodes of Relay.
 */
#include "tvm/relay/type.h"
#include "tvm/ir_functor.h"


namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

TensorType TensorTypeNode::make(Array<ShapeExpr> shape, DataType dtype) {
  std::shared_ptr<TensorTypeNode> n = std::make_shared<TensorTypeNode>();
  n->shape = std::move(shape);
  n->dtype = std::move(dtype);
  return TensorType(n);
}

TensorType TensorTypeNode::Int(int bits, int lanes) {
  return TensorTypeNode::make({}, HalideIR::Int(bits, lanes));
}

TensorType TensorTypeNode::UInt(int bits, int lanes) {
  return TensorTypeNode::make({}, HalideIR::UInt(bits, lanes));
}

TensorType TensorTypeNode::Float(int bits, int lanes) {
  return TensorTypeNode::make({}, HalideIR::Float(bits, lanes));
}

TensorType TensorTypeNode::Bool(int lanes) {
  return TensorTypeNode::make({}, HalideIR::Bool(lanes));
}

TVM_REGISTER_API("relay._make.TensorType")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Array<ShapeExpr> shape = args[0];
      *ret = TensorTypeNode::make(shape, args[1]);
    });


TVM_REGISTER_API("relay._make.IntType")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = TensorTypeNode::Int(args[0], args[1]);
    });

TVM_REGISTER_API("relay._make.UIntType")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = TensorTypeNode::UInt(args[0], args[1]);
    });

TVM_REGISTER_API("relay._make.BoolType")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = TensorTypeNode::Bool(args[0]);
    });

TVM_REGISTER_API("relay._make.FloatType")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = TensorTypeNode::Float(args[0], args[1]);
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<TensorTypeNode>([](const TensorTypeNode *node,
                                     tvm::IRPrinter *p) {
      p->stream << "TensorTypeNode(" << node->dtype << ", " << node->shape
                << ")";
    });

TypeParam TypeParamNode::make(std::string name, TypeParamNode::Kind kind) {
  std::shared_ptr<TypeParamNode> n = std::make_shared<TypeParamNode>();
  n->var = tvm::Var(name);
  n->kind = std::move(kind);
  return TypeParam(n);
}

TVM_REGISTER_API("relay._make.TypeParam")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      int kind = args[1];
      *ret =
          TypeParamNode::make(args[0], static_cast<TypeParamNode::Kind>(kind));
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<TypeParamNode>([](const TypeParamNode *node,
                                    tvm::IRPrinter *p) {
      p->stream << "TypeParamNode(" << node->var->name_hint << ", "
                << node->kind << ")";
    });


FuncType FuncTypeNode::make(tvm::Array<Type> arg_types, Type ret_type,
                             tvm::Array<TypeParam> type_params,
                             tvm::Array<TypeConstraint> type_constraints) {
  std::shared_ptr<FuncTypeNode> n = std::make_shared<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->type_constraints = std::move(type_constraints);
  return FuncType(n);
}

TVM_REGISTER_API("relay._make.FuncType")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = FuncTypeNode::make(args[0], args[1], args[2], args[3]);
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<FuncTypeNode>([](const FuncTypeNode *node,
                                   tvm::IRPrinter *p) {
      p->stream << "FuncTypeNode(" << node->type_params << ", "
                << node->arg_types << ", " << node->ret_type << ", "
                << node->type_constraints << ")";
    });

TypeFunction TypeFunctionNode::make(std::string name, int num_args) {
  std::shared_ptr<TypeFunctionNode> n = std::make_shared<TypeFunctionNode>();
  n->name = std::move(name);
  n->num_args = std::move(num_args);
  return TypeFunction(n);
}

TVM_REGISTER_API("relay._make.TypeFunction")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = TypeFunctionNode::make(args[0], args[1]);
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<TypeFunctionNode>([](const TypeFunctionNode *node,
                                   tvm::IRPrinter *p) {
      p->stream << "TypeFunctionNode(" << node->name << ", " << node->num_args << ")";
    });

TypeCall TypeCallNode::make(Type func, Array<Type> args) {
  std::shared_ptr<TypeCallNode> n = std::make_shared<TypeCallNode>();
  n->func = std::move(func);
  n->args = std::move(args);
  return TypeCall(n);
}

TVM_REGISTER_API("relay._make.TypeCall")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = TypeCallNode::make(args[0], args[1]);
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<TypeCallNode>([](const TypeCallNode *node,
                                   tvm::IRPrinter *p) {
      p->stream << "TypeCallNode(" << node->func << ", " << node->args << ")";
    });


}  // namespace relay
}  // namespace tvm
