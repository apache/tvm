/*!
 *  Copyright (c) 2018 by Contributors
 * \file attrs.cc
 */
#include <tvm/attrs.h>

namespace tvm {

void DictAttrsNode::VisitAttrs(AttrVisitor* v)  {
  v->Visit("__dict__", &dict);
}

void DictAttrsNode::InitByPackedArgs(
    const runtime::TVMArgs& args, bool allow_unknown) {
  for (int i = 0; i < args.size(); i += 2) {
    std::string key = args[i];
    runtime::TVMArgValue val = args[i + 1];
    if (val.type_code() == kNodeHandle) {
      dict.Set(key, val.operator NodeRef());
    } else if (val.type_code() == kStr) {
      dict.Set(key, Expr(val.operator std::string()));
    } else {
      dict.Set(key, val.operator Expr());
    }
  }
}

Array<AttrFieldInfo> DictAttrsNode::ListFieldInfo() const {
  return {};
}

Attrs DictAttrsNode::make(Map<std::string, NodeRef> dict) {
  NodePtr<DictAttrsNode> n = make_node<DictAttrsNode>();
  n->dict = std::move(dict);
  return Attrs(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<DictAttrsNode>([](const DictAttrsNode *op, IRPrinter *p) {
    p->stream << op->dict;
});

TVM_REGISTER_NODE_TYPE(DictAttrsNode);

TVM_REGISTER_NODE_TYPE(AttrFieldInfoNode);

}  // namespace tvm
