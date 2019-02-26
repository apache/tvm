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

TVM_REGISTER_NODE_TYPE(ConstantNode);

TVM_REGISTER_API("relay._make.Constant")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = ConstantNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ConstantNode>([](const ConstantNode* node, tvm::IRPrinter* p) {
    const PackedFunc* fprint = Registry::Get("relay._constant_repr");
    CHECK(fprint) << "unable to find printing function for constants";
    std::string data = (*fprint)(GetRef<Constant>(node));
    p->stream << "Constant(" << data << ")";
  });

TensorType ConstantNode::tensor_type() const {
  auto dtype = TVMType2Type(data->dtype);
  Array<tvm::Expr> shape;
  for (int i = 0; i < data->ndim; i++) {
    CHECK_LE(data->shape[i], std::numeric_limits<int32_t>::max());
    CHECK_GE(data->shape[i], std::numeric_limits<int32_t>::min());
    shape.push_back(
        tvm::ir::IntImm::make(Int(32), data->shape[i]));
  }

  return TensorTypeNode::make(shape, dtype);
}

Tuple TupleNode::make(tvm::Array<relay::Expr> fields) {
  NodePtr<TupleNode> n = make_node<TupleNode>();
  n->fields = std::move(fields);
  return Tuple(n);
}

TVM_REGISTER_NODE_TYPE(TupleNode);

TVM_REGISTER_API("relay._make.Tuple")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = TupleNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TupleNode>([](const TupleNode* node, tvm::IRPrinter* p) {
    p->stream << "Tuple(" << node->fields << ")";
  });


Var VarNode::make(Id vid, Type type_annotation) {
  NodePtr<VarNode> n = make_node<VarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  return Var(n);
}

Var VarNode::make(std::string name_hint, Type type_annotation) {
  NodePtr<IdNode> n = make_node<IdNode>();
  n->name_hint = std::move(name_hint);
  return VarNode::make(Id(n), type_annotation);
}

TVM_REGISTER_NODE_TYPE(VarNode);

TVM_REGISTER_API("relay._make.Var")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = VarNode::make(args[0].operator std::string(), args[1]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<VarNode>([](const VarNode* node, tvm::IRPrinter* p) {
    p->stream << "Var(" << node->name_hint();
    if (node->type_annotation.defined()) {
      p->stream << ", ty=";
      p->print(node->type_annotation);
    }
    p->stream << ")";
  });

GlobalVar GlobalVarNode::make(std::string name_hint) {
  NodePtr<GlobalVarNode> n = make_node<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  return GlobalVar(n);
}

TVM_REGISTER_NODE_TYPE(GlobalVarNode);

TVM_REGISTER_API("relay._make.GlobalVar")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = GlobalVarNode::make(args[0]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<GlobalVarNode>([](const GlobalVarNode* node, tvm::IRPrinter* p) {
    p->stream << "GlobalVar(" << node->name_hint << ")";
  });


Function FunctionNode::make(tvm::Array<Var> params,
                            Expr body,
                            Type ret_type,
                            tvm::Array<TypeVar> type_params,
                            tvm::Attrs attrs) {
  NodePtr<FunctionNode> n = make_node<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->attrs = std::move(attrs);
  return Function(n);
}

FuncType FunctionNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    Type param_type = (param->type_annotation.defined()) ? param->type_annotation
      : IncompleteTypeNode::make(Kind::kType);
    param_types.push_back(param_type);
  }

  Type ret_type = (this->ret_type.defined()) ? this->ret_type
    : IncompleteTypeNode::make(Kind::kType);
  return FuncTypeNode::make(param_types, ret_type, this->type_params, {});
}

bool FunctionNode::IsPrimitive() const {
  NodeRef res = FunctionGetAttr(GetRef<Function>(this), "Primitive");
  const ir::IntImm* pval = res.as<ir::IntImm>();
  return pval && pval->value != 0;
}

NodeRef FunctionGetAttr(const Function& func, const std::string& key) {
  if (!func->attrs.defined()) { return NodeRef(); }

  const DictAttrsNode* dict_attrs = func->attrs.as<DictAttrsNode>();
  CHECK(dict_attrs);
  auto it = dict_attrs->dict.find(key);
  if (it != dict_attrs->dict.end()) {
    return (*it).second;
  } else {
    return NodeRef();
  }
}

Function FunctionSetAttr(const Function& func, const std::string& key, const NodeRef& data) {
  const DictAttrsNode* dattrs = func->attrs.as<DictAttrsNode>();
  Attrs func_attrs;
  if (dattrs) {
    Map<std::string, NodeRef> dict = dattrs->dict;
    dict.Set(key, data);
    func_attrs = DictAttrsNode::make(dict);
  } else {
    Map<std::string, NodeRef> dict = {{key, data}};
    func_attrs = DictAttrsNode::make(dict);
  }

  return FunctionNode::make(
    func->params,
    func->body,
    func->ret_type,
    func->type_params,
    func_attrs);
}

TVM_REGISTER_NODE_TYPE(FunctionNode);

TVM_REGISTER_API("relay._make.Function")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = FunctionNode::make(args[0], args[1], args[2], args[3], args[4]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<FunctionNode>([](const FunctionNode* node,
                                   tvm::IRPrinter* p) {
      p->stream << "FunctionNode(" << node->params << ", " << node->ret_type
                << ", " << node->body << ", " << node->type_params << ", "
                << node->attrs << ")";
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

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_REGISTER_API("relay._make.Call")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = CallNode::make(args[0], args[1], args[2], args[3]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<CallNode>([](const CallNode* node, tvm::IRPrinter* p) {
  p->stream << "CallNode(" << node->op << ", " << node->args << ", "
    << node->attrs << ", " << node->type_args << ")";
});

Let LetNode::make(Var var, Expr value, Expr body) {
  NodePtr<LetNode> n = make_node<LetNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->body = std::move(body);
  return Let(n);
}

TVM_REGISTER_NODE_TYPE(LetNode);

TVM_REGISTER_API("relay._make.Let")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = LetNode::make(args[0], args[1], args[2]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<LetNode>([](const LetNode* node, tvm::IRPrinter* p) {
  p->stream << "LetNode(" << node->var << ", " << node->value
            << ", " << node->body << ")";
});

If IfNode::make(Expr cond, Expr true_branch, Expr false_branch) {
  NodePtr<IfNode> n = make_node<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  return If(n);
}

TVM_REGISTER_NODE_TYPE(IfNode);

TVM_REGISTER_API("relay._make.If").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = IfNode::make(args[0], args[1], args[2]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<IfNode>([](const IfNode* node, tvm::IRPrinter* p) {
  p->stream << "IfNode(" << node->cond << ", " << node->true_branch
            << ", " << node->false_branch << ")";
});

TupleGetItem TupleGetItemNode::make(Expr tuple, int index) {
  NodePtr<TupleGetItemNode> n = make_node<TupleGetItemNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  return TupleGetItem(n);
}

TVM_REGISTER_NODE_TYPE(TupleGetItemNode);

TVM_REGISTER_API("relay._make.TupleGetItem").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = TupleGetItemNode::make(args[0], args[1]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TupleGetItemNode>([](const TupleGetItemNode* node, tvm::IRPrinter* p) {
  p->stream << "TupleGetItemNode(" << node->tuple << ", " << node->index << ")";
});

RefCreate RefCreateNode::make(Expr value) {
  NodePtr<RefCreateNode> n = make_node<RefCreateNode>();
  n->value = std::move(value);
  return RefCreate(n);
}

TVM_REGISTER_API("relay._make.RefCreate").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = RefCreateNode::make(args[0]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<RefCreateNode>([](const RefCreateNode* node, tvm::IRPrinter* p) {
  p->stream << "RefCreateNode(" << node->value << ")";
});

RefRead RefReadNode::make(Expr ref) {
  NodePtr<RefReadNode> n = make_node<RefReadNode>();
  n->ref = std::move(ref);
  return RefRead(n);
}

TVM_REGISTER_API("relay._make.RefRead")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = RefReadNode::make(args[0]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<RefReadNode>([](const RefReadNode* node, tvm::IRPrinter* p) {
  p->stream << "RefReadNode(" << node->ref << ")";
});

RefWrite RefWriteNode::make(Expr ref, Expr value) {
  NodePtr<RefWriteNode> n = make_node<RefWriteNode>();
  n->ref = std::move(ref);
  n->value = std::move(value);
  return RefWrite(n);
}

TVM_REGISTER_API("relay._make.RefWrite")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = RefWriteNode::make(args[0], args[1]);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<RefWriteNode>([](const RefWriteNode* node, tvm::IRPrinter* p) {
  p->stream << "RefWriteNode(" << node->ref << ", " << node->value << ")";
});

TVM_REGISTER_API("relay._expr.TempExprRealize")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  TempExpr temp = args[0];
  *ret = temp->Realize();
});

}  // namespace relay
}  // namespace tvm
