/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/ir/type.cc
 * \brief The type system AST nodes of Relay.
 */
#include <tvm/relay/type.h>

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

TensorType TensorTypeNode::make(Array<IndexExpr> shape, DataType dtype) {
  NodePtr<TensorTypeNode> n = make_node<TensorTypeNode>();
  n->shape = std::move(shape);
  n->dtype = std::move(dtype);
  return TensorType(n);
}

TensorType TensorTypeNode::Scalar(DataType dtype) {
  return TensorTypeNode::make({}, dtype);
}

TVM_REGISTER_API("relay._make.TensorType")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  Array<IndexExpr> shape = args[0];
  *ret = TensorTypeNode::make(shape, args[1]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TensorTypeNode>([](const TensorTypeNode *node,
                                     tvm::IRPrinter *p) {
  p->stream << "TensorTypeNode(" << node->dtype << ", " << node->shape << ")";
});

TypeParam TypeParamNode::make(std::string name, TypeParamNode::Kind kind) {
  NodePtr<TypeParamNode> n = make_node<TypeParamNode>();
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

IncompleteType IncompleteTypeNode::make(TypeParamNode::Kind kind) {
  auto n = make_node<IncompleteTypeNode>();
  n->kind = std::move(kind);
  return IncompleteType(n);
}

TVM_REGISTER_API("relay._make.IncompleteType")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    int kind = args[0];
    *ret = IncompleteTypeNode::make(static_cast<TypeParamNode::Kind>(kind));
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<IncompleteTypeNode>(
    [](const IncompleteTypeNode* node,
       tvm::IRPrinter* p) {
      p->stream << "IncompleteTypeNode(" << node->kind << ", " << node << ")";
    });

FuncType FuncTypeNode::make(tvm::Array<Type> arg_types,
                            Type ret_type,
                            tvm::Array<TypeParam> type_params,
                            tvm::Array<TypeConstraint> type_constraints) {
  NodePtr<FuncTypeNode> n = make_node<FuncTypeNode>();
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

TypeRelation TypeRelationNode::make(TypeRelationFn func,
                                    Array<Type> args,
                                    int num_inputs,
                                    Attrs attrs) {
  NodePtr<TypeRelationNode> n = make_node<TypeRelationNode>();
  n->func = std::move(func);
  n->args = std::move(args);
  n->num_inputs = num_inputs;
  n->attrs = std::move(attrs);
  return TypeRelation(n);
}

TVM_REGISTER_API("relay._make.TypeRelation")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = TypeRelationNode::make(args[0], args[1], args[2], args[3]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TypeRelationNode>([](const TypeRelationNode *node, tvm::IRPrinter *p) {
    p->stream << "TypeRelationNode("
              << node->func->name
              << ", " << node->args << ")";
});

TupleType TupleTypeNode::make(Array<Type> fields) {
  NodePtr<TupleTypeNode> n = make_node<TupleTypeNode>();
  n->fields = std::move(fields);
  return TupleType(n);
}

TVM_REGISTER_API("relay._make.TupleType")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = TupleTypeNode::make(args[0]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TupleTypeNode>([](const TupleTypeNode *node,
                                    tvm::IRPrinter *p) {
  p->stream << "TupleTypeNode(" << node->fields << ")";
});

}  // namespace relay
}  // namespace tvm
