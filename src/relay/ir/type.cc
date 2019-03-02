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

IndexExpr TensorTypeNode::Size() const {
  if (shape.size() == 0) {
    return make_const(Int(64), 1);
  }

  IndexExpr size = shape[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    size *= shape[i];
  }
  return size;
}

TVM_REGISTER_NODE_TYPE(TensorTypeNode);

TVM_REGISTER_API("relay._make.TensorType")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Array<IndexExpr> shape = args[0];
  *ret = TensorTypeNode::make(shape, args[1]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TensorTypeNode>([](const TensorTypeNode* node,
                                 tvm::IRPrinter* p) {
  p->stream << "TensorType(" << node->shape << ", " << node->dtype << ")";
});

TypeVar TypeVarNode::make(std::string name, Kind kind) {
  NodePtr<TypeVarNode> n = make_node<TypeVarNode>();
  n->var = tvm::Var(name);
  n->kind = std::move(kind);
  return TypeVar(n);
}

TVM_REGISTER_NODE_TYPE(TypeVarNode);

TVM_REGISTER_API("relay._make.TypeVar")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  int kind = args[1];
  *ret =
    TypeVarNode::make(args[0], static_cast<Kind>(kind));
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TypeVarNode>([](const TypeVarNode* node,
                                    tvm::IRPrinter* p) {
  p->stream << "TypeVarNode(" << node->var->name_hint << ", "
    << node->kind << ")";
});

GlobalTypeVar GlobalTypeVarNode::make(std::string name, Kind kind) {
  NodePtr<GlobalTypeVarNode> n = make_node<GlobalTypeVarNode>();
  n->var = tvm::Var(name);
  n->kind = std::move(kind);
  return GlobalTypeVar(n);
}

TVM_REGISTER_NODE_TYPE(GlobalTypeVarNode);

TVM_REGISTER_API("relay._make.GlobalTypeVar")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  int kind = args[1];
  *ret = GlobalTypeVarNode::make(args[0], static_cast<Kind>(kind));
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<GlobalTypeVarNode>([](const GlobalTypeVarNode *node,
                                    tvm::IRPrinter *p) {
  p->stream << "GlobalTypeVarNode(" << node->var->name_hint << ", "
            << node->kind << ")";
});

TypeCall TypeCallNode::make(Type func, tvm::Array<Type> args) {
  NodePtr<TypeCallNode> n = make_node<TypeCallNode>();
  n->func = std::move(func);
  n->args = std::move(args);
  return TypeCall(n);
}

TVM_REGISTER_NODE_TYPE(TypeCallNode);

TVM_REGISTER_API("relay._make.TypeCall")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = TypeCallNode::make(args[0], args[1]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TypeCallNode>([](const TypeCallNode* node,
                               tvm::IRPrinter* p) {
  p->stream << "TypeCallNode(" << node->func << ", "
            << node->args << ")";
});

IncompleteType IncompleteTypeNode::make(Kind kind) {
  auto n = make_node<IncompleteTypeNode>();
  n->kind = std::move(kind);
  return IncompleteType(n);
}

TVM_REGISTER_NODE_TYPE(IncompleteTypeNode);

TVM_REGISTER_API("relay._make.IncompleteType")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    int kind = args[0];
    *ret = IncompleteTypeNode::make(static_cast<Kind>(kind));
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<IncompleteTypeNode>(
    [](const IncompleteTypeNode* node,
       tvm::IRPrinter* p) {
      p->stream << "IncompleteTypeNode(" << node->kind << ", " << node << ")";
    });

FuncType FuncTypeNode::make(tvm::Array<Type> arg_types,
                            Type ret_type,
                            tvm::Array<TypeVar> type_params,
                            tvm::Array<TypeConstraint> type_constraints) {
  NodePtr<FuncTypeNode> n = make_node<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->type_constraints = std::move(type_constraints);
  return FuncType(n);
}

TVM_REGISTER_NODE_TYPE(FuncTypeNode);

TVM_REGISTER_API("relay._make.FuncType")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = FuncTypeNode::make(args[0], args[1], args[2], args[3]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FuncTypeNode>([](const FuncTypeNode* node,
                                   tvm::IRPrinter* p) {
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

TVM_REGISTER_NODE_TYPE(TypeRelationNode);

TVM_REGISTER_API("relay._make.TypeRelation")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = TypeRelationNode::make(args[0], args[1], args[2], args[3]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TypeRelationNode>([](const TypeRelationNode* node, tvm::IRPrinter* p) {
    p->stream << "TypeRelationNode("
              << node->func->name
              << ", " << node->args << ")";
});

TupleType TupleTypeNode::make(Array<Type> fields) {
  NodePtr<TupleTypeNode> n = make_node<TupleTypeNode>();
  n->fields = std::move(fields);
  return TupleType(n);
}

TVM_REGISTER_NODE_TYPE(TupleTypeNode);

TVM_REGISTER_API("relay._make.TupleType")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = TupleTypeNode::make(args[0]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TupleTypeNode>([](const TupleTypeNode* node,
                                tvm::IRPrinter* p) {
  p->stream << "TupleTypeNode(" << node->fields << ")";
});

RefType RefTypeNode::make(Type value) {
  NodePtr<RefTypeNode> n = make_node<RefTypeNode>();
  n->value = std::move(value);
  return RefType(n);
}

TVM_REGISTER_API("relay._make.RefType")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = RefTypeNode::make(args[0]);
});

TVM_REGISTER_NODE_TYPE(RefTypeNode);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<RefTypeNode>([](const RefTypeNode* node,
                              tvm::IRPrinter* p) {
  p->stream << "RefTypeNode(" << node->value << ")";
});

}  // namespace relay
}  // namespace tvm
