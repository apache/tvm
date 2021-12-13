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
 * \file src/relay/ir/expr.cc
 * \brief The expression AST nodes of Relay.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/target/se_scope.h>

namespace tvm {

SEScope RelayExprNode::virtual_device() const {
  if (virtual_device_.defined()) {
    return Downcast<SEScope>(this->virtual_device_);
  }
  return SEScope::FullyUnconstrained();
}

namespace relay {

using tvm::ReprPrinter;
using namespace tvm::runtime;

Constant::Constant(runtime::NDArray data, Span span) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ConstantNode);

TVM_REGISTER_GLOBAL("relay.ir.Constant").set_body_typed([](runtime::NDArray data) {
  return Constant(data);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstantNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ConstantNode*>(ref.get());
      const PackedFunc* fprint = Registry::Get("relay._constant_repr");
      ICHECK(fprint) << "unable to find printing function for constants";
      std::string data = (*fprint)(GetRef<Constant>(node));
      p->stream << "Constant(" << data << ")";
    });

TensorType ConstantNode::tensor_type() const {
  auto dtype = DataType(data->dtype);
  Array<tvm::PrimExpr> shape;
  for (int i = 0; i < data->ndim; i++) {
    ICHECK_LE(data->shape[i], std::numeric_limits<int32_t>::max());
    ICHECK_GE(data->shape[i], std::numeric_limits<int32_t>::min());
    shape.push_back(tvm::IntImm(DataType::Int(32), data->shape[i]));
  }

  return TensorType(shape, dtype);
}

Tuple::Tuple(tvm::Array<relay::Expr> fields, Span span) {
  ObjectPtr<TupleNode> n = make_object<TupleNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleNode);

TVM_REGISTER_GLOBAL("relay.ir.Tuple").set_body_typed([](tvm::Array<relay::Expr> fields, Span span) {
  return Tuple(fields, span);
});
Tuple WithFields(Tuple tuple, Optional<Array<Expr>> opt_fields,
                 Optional<SEScope> opt_virtual_device, Optional<Span> opt_span) {
  Array<Expr> fields = opt_fields.value_or(tuple->fields);
  SEScope virtual_device = opt_virtual_device.value_or(tuple->virtual_device());
  Span span = opt_span.value_or(tuple->span);

  bool all_fields_unchanged = true;
  if (fields.size() == tuple->fields.size()) {
    for (size_t i = 0; i < fields.size(); i++) {
      all_fields_unchanged &= fields[i].same_as(tuple->fields[i]);
    }
  } else {
    all_fields_unchanged = false;
  }

  all_fields_unchanged = all_fields_unchanged && span.same_as(tuple->span);
  if (!all_fields_unchanged) {
    TupleNode* cow_tuple_node = tuple.CopyOnWrite();
    cow_tuple_node->fields = fields;
    cow_tuple_node->virtual_device_ = virtual_device;
    cow_tuple_node->span = span;
  }
  return std::move(tuple);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleNode*>(ref.get());
      p->stream << "Tuple(" << node->fields << ")";
    });

Var::Var(Id vid, Type type_annotation, Span span) {
  ObjectPtr<VarNode> n = make_object<VarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

Var WithFields(Var var, Optional<Id> opt_vid, Optional<Type> opt_type_annotation,
               Optional<SEScope> opt_virtual_device, Optional<Span> opt_span) {
  Id vid = opt_vid.value_or(var->vid);
  Type type_annotation = opt_type_annotation.value_or(var->type_annotation);
  SEScope virtual_device = opt_virtual_device.value_or(var->virtual_device());
  Span span = opt_span.value_or(var->span);

  bool unchanged = vid.same_as(var->vid) && type_annotation.same_as(var->type_annotation) &&
                   span.same_as(var->span);

  if (!unchanged) {
    VarNode* cow_var_node = var.CopyOnWrite();
    cow_var_node->vid = vid;
    cow_var_node->type_annotation = type_annotation;
    cow_var_node->virtual_device_ = virtual_device;
    cow_var_node->span = span;
  }
  return std::move(var);
}

TVM_REGISTER_NODE_TYPE(VarNode);

TVM_REGISTER_GLOBAL("relay.ir.Var").set_body_typed([](String str, Type type_annotation) {
  return Var(str, type_annotation);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<VarNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const VarNode*>(ref.get());
      p->stream << "Var(" << node->name_hint();
      if (node->type_annotation.defined()) {
        p->stream << ", ty=";
        p->Print(node->type_annotation);
      }
      p->stream << ")";
    });

Call::Call(Expr op, Array<Expr> args, Attrs attrs, Array<Type> type_args, Span span) {
  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

Call WithFields(Call call, Optional<Expr> opt_op, Optional<Array<Expr>> opt_args,
                Optional<Attrs> opt_attrs, Optional<Array<Type>> opt_type_args,
                Optional<SEScope> opt_virtual_device, Optional<Span> opt_span) {
  Expr op = opt_op.value_or(call->op);
  Array<Expr> args = opt_args.value_or(call->args);
  Attrs attrs = opt_attrs.value_or(call->attrs);
  Array<Type> type_args = opt_type_args.value_or(call->type_args);
  SEScope virtual_device = opt_virtual_device.value_or(call->virtual_device());
  Span span = opt_span.value_or(call->span);

  bool unchanged = op.same_as(call->op) && attrs.same_as(call->attrs) && span.same_as(call->span);

  // Check that the args are unchanged
  if (unchanged) {
    bool all_args_unchanged = true;
    if (args.size() == call->args.size()) {
      for (size_t i = 0; i < args.size(); i++) {
        all_args_unchanged &= args[i].same_as(call->args[i]);
      }
    } else {
      all_args_unchanged = false;
    }
    unchanged &= all_args_unchanged;
  }

  // Check that the type_args are unchanged
  if (unchanged) {
    bool all_type_args_unchanged = true;
    if (type_args.size() == call->type_args.size()) {
      for (size_t i = 0; i < type_args.size(); i++) {
        all_type_args_unchanged &= type_args[i].same_as(call->type_args[i]);
      }
    } else {
      all_type_args_unchanged = false;
    }

    unchanged &= all_type_args_unchanged;
  }

  if (!unchanged) {
    CallNode* cow_call_node = call.CopyOnWrite();
    cow_call_node->op = op;
    cow_call_node->args = args;
    cow_call_node->attrs = attrs;
    cow_call_node->type_args = type_args;
    cow_call_node->virtual_device_ = virtual_device;
    cow_call_node->span = span;
  }
  return std::move(call);
}

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_REGISTER_GLOBAL("relay.ir.Call")
    .set_body_typed([](Expr op, Array<Expr> args, Attrs attrs, Array<Type> type_args, Span span) {
      return Call(op, args, attrs, type_args, span);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CallNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CallNode*>(ref.get());
      p->stream << "CallNode(" << node->op << ", " << node->args << ", " << node->attrs << ", "
                << node->type_args << ")";
    });

Let::Let(Var var, Expr value, Expr body, Span span) {
  ObjectPtr<LetNode> n = make_object<LetNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->body = std::move(body);
  n->span = std::move(span);
  data_ = std::move(n);
}

Let WithFields(Let let, Optional<Var> opt_var, Optional<Expr> opt_value, Optional<Expr> opt_body,
               Optional<SEScope> opt_virtual_device, Optional<Span> opt_span) {
  Var var = opt_var.value_or(let->var);
  Expr value = opt_value.value_or(let->value);
  Expr body = opt_body.value_or(let->body);
  SEScope virtual_device = opt_virtual_device.value_or(let->virtual_device());
  Span span = opt_span.value_or(let->span);

  bool unchanged = var.same_as(let->var) && value.same_as(let->value) && body.same_as(let->body) &&
                   span.same_as(let->span);

  if (!unchanged) {
    LetNode* cow_let_node = let.CopyOnWrite();
    cow_let_node->var = var;
    cow_let_node->value = value;
    cow_let_node->body = body;
    cow_let_node->virtual_device_ = virtual_device;
    cow_let_node->span = span;
  }
  return std::move(let);
}

TVM_REGISTER_NODE_TYPE(LetNode);

TVM_REGISTER_GLOBAL("relay.ir.Let").set_body_typed([](Var var, Expr value, Expr body) {
  return Let(var, value, body);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const LetNode*>(ref.get());
      p->stream << "LetNode(" << node->var << ", " << node->value << ", " << node->body << ")";
    });

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ObjectPtr<IfNode> n = make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  n->span = std::move(span);
  data_ = std::move(n);
}

If WithFields(If if_expr, Optional<Expr> opt_cond, Optional<Expr> opt_true_branch,
              Optional<Expr> opt_false_branch, Optional<SEScope> opt_virtual_device,
              Optional<Span> opt_span) {
  Expr cond = opt_cond.value_or(if_expr->cond);
  Expr true_branch = opt_true_branch.value_or(if_expr->true_branch);
  Expr false_branch = opt_false_branch.value_or(if_expr->false_branch);
  SEScope virtual_device = opt_virtual_device.value_or(if_expr->virtual_device());
  Span span = opt_span.value_or(if_expr->span);

  bool unchanged = cond.same_as(if_expr->cond) && true_branch.same_as(if_expr->true_branch) &&
                   false_branch.same_as(if_expr->false_branch) && span.same_as(if_expr->span);

  if (!unchanged) {
    IfNode* cow_if_node = if_expr.CopyOnWrite();
    cow_if_node->cond = cond;
    cow_if_node->true_branch = true_branch;
    cow_if_node->false_branch = false_branch;
    cow_if_node->virtual_device_ = virtual_device;
    cow_if_node->span = span;
  }
  return std::move(if_expr);
}

TVM_REGISTER_NODE_TYPE(IfNode);

TVM_REGISTER_GLOBAL("relay.ir.If")
    .set_body_typed([](Expr cond, Expr true_branch, Expr false_branch) {
      return If(cond, true_branch, false_branch);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IfNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IfNode*>(ref.get());
      p->stream << "IfNode(" << node->cond << ", " << node->true_branch << ", "
                << node->false_branch << ")";
    });

TupleGetItem::TupleGetItem(Expr tuple, int index, Span span) {
  ObjectPtr<TupleGetItemNode> n = make_object<TupleGetItemNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleGetItem WithFields(TupleGetItem tuple_get_item, Optional<Expr> opt_tuple,
                        Optional<Integer> opt_index, Optional<SEScope> opt_virtual_device,
                        Optional<Span> opt_span) {
  Expr tuple = opt_tuple.value_or(tuple_get_item->tuple);
  Integer index = opt_index.value_or(tuple_get_item->index);
  SEScope virtual_device = opt_virtual_device.value_or(tuple->virtual_device());
  Span span = opt_span.value_or(tuple_get_item->span);

  bool unchanged = tuple.same_as(tuple_get_item->tuple) && (index == tuple_get_item->index) &&
                   span.same_as(tuple_get_item->span);
  if (!unchanged) {
    TupleGetItemNode* cow_tuple_get_item_node = tuple_get_item.CopyOnWrite();
    cow_tuple_get_item_node->tuple = tuple;
    cow_tuple_get_item_node->index = index;
    cow_tuple_get_item_node->span = span;
    cow_tuple_get_item_node->virtual_device_ = virtual_device;
  }
  return std::move(tuple_get_item);
}

TVM_REGISTER_NODE_TYPE(TupleGetItemNode);

TVM_REGISTER_GLOBAL("relay.ir.TupleGetItem").set_body_typed([](Expr tuple, int index) {
  return TupleGetItem(tuple, index);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleGetItemNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleGetItemNode*>(ref.get());
      p->stream << "TupleGetItemNode(" << node->tuple << ", " << node->index << ")";
    });

RefCreate::RefCreate(Expr value, Span span) {
  ObjectPtr<RefCreateNode> n = make_object<RefCreateNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  data_ = std::move(n);
}

RefCreate WithFields(RefCreate ref_create, Optional<Expr> opt_value,
                     Optional<SEScope> opt_virtual_device, Optional<Span> opt_span) {
  Expr value = opt_value.value_or(ref_create->value);
  SEScope virtual_device = opt_virtual_device.value_or(ref_create->virtual_device());
  Span span = opt_span.value_or(ref_create->span);

  bool unchanged = value.same_as(ref_create->value) && span.same_as(ref_create->span);
  if (!unchanged) {
    RefCreateNode* cow_ref_create_node = ref_create.CopyOnWrite();
    cow_ref_create_node->value = value;
    cow_ref_create_node->virtual_device_ = virtual_device;
    cow_ref_create_node->span = span;
  }
  return std::move(ref_create);
}

TVM_REGISTER_NODE_TYPE(RefCreateNode);

TVM_REGISTER_GLOBAL("relay.ir.RefCreate").set_body_typed([](Expr value) {
  return RefCreate(value);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RefCreateNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RefCreateNode*>(ref.get());
      p->stream << "RefCreateNode(" << node->value << ")";
    });

RefRead::RefRead(Expr ref, Span span) {
  ObjectPtr<RefReadNode> n = make_object<RefReadNode>();
  n->ref = std::move(ref);
  n->span = std::move(span);
  data_ = std::move(n);
}

RefRead WithFields(RefRead ref_read, Optional<Expr> opt_ref, Optional<SEScope> opt_virtual_device,
                   Optional<Span> opt_span) {
  Expr ref = opt_ref.value_or(ref_read->ref);
  SEScope virtual_device = opt_virtual_device.value_or(ref_read->virtual_device());
  Span span = opt_span.value_or(ref_read->span);

  bool unchanged = ref.same_as(ref_read->ref) && span.same_as(ref_read->span);
  if (!unchanged) {
    RefReadNode* cow_ref_read_node = ref_read.CopyOnWrite();
    cow_ref_read_node->ref = ref;
    cow_ref_read_node->virtual_device_ = virtual_device;
    cow_ref_read_node->span = span;
  }
  return std::move(ref_read);
}

TVM_REGISTER_NODE_TYPE(RefReadNode);

TVM_REGISTER_GLOBAL("relay.ir.RefRead").set_body_typed([](Expr ref) { return RefRead(ref); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RefReadNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RefReadNode*>(ref.get());
      p->stream << "RefReadNode(" << node->ref << ")";
    });

RefWrite::RefWrite(Expr ref, Expr value, Span span) {
  ObjectPtr<RefWriteNode> n = make_object<RefWriteNode>();
  n->ref = std::move(ref);
  n->value = std::move(value);
  n->span = std::move(span);
  data_ = std::move(n);
}

RefWrite WithFields(RefWrite ref_write, Optional<Expr> opt_ref, Optional<Expr> opt_value,
                    Optional<SEScope> opt_virtual_device, Optional<Span> opt_span) {
  Expr ref = opt_ref.value_or(ref_write->ref);
  Expr value = opt_value.value_or(ref_write->value);
  SEScope virtual_device = opt_virtual_device.value_or(ref_write->virtual_device());
  Span span = opt_span.value_or(ref_write->span);

  bool unchanged = ref.same_as(ref_write->ref) && value.same_as(ref_write->value) &&
                   span.same_as(ref_write->span);
  if (!unchanged) {
    RefWriteNode* cow_ref_write_node = ref_write.CopyOnWrite();
    cow_ref_write_node->ref = ref;
    cow_ref_write_node->value = value;
    cow_ref_write_node->virtual_device_ = virtual_device;
    cow_ref_write_node->span = span;
  }
  return std::move(ref_write);
}

TVM_REGISTER_NODE_TYPE(RefWriteNode);

TVM_REGISTER_GLOBAL("relay.ir.RefWrite").set_body_typed([](Expr ref, Expr value) {
  return RefWrite(ref, value);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RefWriteNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RefWriteNode*>(ref.get());
      p->stream << "RefWriteNode(" << node->ref << ", " << node->value << ")";
    });

TVM_REGISTER_GLOBAL("relay.ir.TempExprRealize").set_body_typed([](TempExpr temp) {
  return temp->Realize();
});

TVM_REGISTER_GLOBAL("relay.ir.Any").set_body_typed([]() { return Any(); });

/*
 * Non-recursive traversal with dismantling unused call nodes,
 * a derivative from ExpandDataflow method
 */
inline void Dismantle(const Expr& expr) {
  std::stack<std::pair<Expr, bool>> stack;
  auto fpush_to_stack = [&stack](const Expr& expr) {
    // do not visit nodes with more than 2 refs (one can be in stack)
    if (expr.use_count() < 3) {
      stack.push({expr, false});
    }
  };
  fpush_to_stack(expr);
  while (stack.size() > 0) {
    const auto& node = stack.top().first;
    if (stack.top().second) {
      // dismantle node
      // +1 ref in stack/deque;
      if (node.use_count() < 3) {
        if (auto* op = const_cast<CallNode*>(node.as<CallNode>())) {
          op->args = Array<Expr>();
        }
        if (auto* op = const_cast<LetNode*>(node.as<LetNode>())) {
          op->body = Expr();
        }
      }
      // eject
      stack.pop();
    } else {
      stack.top().second = true;

      // special handling
      if (const CallNode* op = node.as<CallNode>()) {
        // do not process args if used elsewhere
        if (op->args.use_count() < 2) {
          for (auto it = op->args.rbegin(); it != op->args.rend(); ++it) {
            fpush_to_stack(*it);
          }
        }
      } else if (const TupleNode* op = node.as<TupleNode>()) {
        // do not process fields if used elsewhere
        if (op->fields.use_count() < 2) {
          for (auto it = op->fields.rbegin(); it != op->fields.rend(); ++it) {
            fpush_to_stack(*it);
          }
        }
      } else if (const TupleGetItemNode* op = node.as<TupleGetItemNode>()) {
        // do not process tuple if used elsewhere
        if (op->tuple.use_count() < 2) {
          fpush_to_stack(op->tuple);
        }
      } else if (const LetNode* op = node.as<LetNode>()) {
        // do not process let if used elsewhere
        if (op->body.use_count() < 2) {
          fpush_to_stack(op->body);
        }
      }
    }
  }
}

/*
 * Non-recursive destructor
 */
Call::~Call() {
  // attempt to dismantle if referenced one or zero times
  if (this->use_count() < 2) {
    if (this->as<CallNode>() && this->as<CallNode>()->args.size()) {
      Dismantle(*this);
    }
  }
}

/*
 * CallNode's deleter
 */
void CallNode::Deleter_(Object* ptr) {
  auto p = reinterpret_cast<CallNode*>(ptr);
  // resore original deleter
  p->deleter_ = p->saved_deleter_;
  // create Call reference in order to invoke ~Call
  auto c = GetRef<Call>(p);
}

/*
 * Non-recursive destructor
 */
Let::~Let() {
  // attempt to dismantle if referenced one or zero times
  if (this->use_count() < 2) {
    if (this->as<LetNode>() && this->as<LetNode>()->body.defined()) {
      Dismantle(*this);
    }
  }
}

/*
 * LetNode's deleter
 */
void LetNode::Deleter_(Object* ptr) {
  auto p = reinterpret_cast<LetNode*>(ptr);
  // resore original deleter
  p->deleter_ = p->saved_deleter_;
  // create Let reference in order to invoke ~Let
  auto c = GetRef<Let>(p);
}

}  // namespace relay
}  // namespace tvm
