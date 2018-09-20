/*!
 *  Copyright (c) 2018 by Contributors
 * \file base.cc
 * \brief The core base types for Relay.
 */
#include <tvm/api_registry.h>
#include <tvm/relay/base.h>

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace tvm::runtime;

NodePtr<SourceNameNode> GetSourceNameNode(const std::string& name) {
  // always return pointer as the reference can change as map re-allocate.
  // or use another level of indirection by creating a unique_ptr
  static std::unordered_map<std::string, NodePtr<SourceNameNode> > source_map;

  auto sn = source_map.find(name);
  if (sn == source_map.end()) {
    NodePtr<SourceNameNode> n = make_node<SourceNameNode>();
    n->name = std::move(name);
    source_map[name] = n;
    return n;
  } else {
    return sn->second;
  }
}

SourceName SourceName::Get(const std::string& name) {
  return SourceName(GetSourceNameNode(name));
}

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<SourceNameNode>([](const SourceNameNode *node, tvm::IRPrinter *p) {
    p->stream << "SourceName(" << node->name << ", " << node << ")";
  });

TVM_REGISTER_NODE_TYPE(SourceNameNode)
.set_creator(GetSourceNameNode)
.set_global_key([](const Node* n) {
    return static_cast<const SourceNameNode*>(n)->name;
  });

Span SpanNode::make(SourceName source, int lineno, int col_offset) {
  auto n = make_node<SpanNode>();
  n->source = std::move(source);
  n->lineno = lineno;
  n->col_offset = col_offset;
  return Span(n);
}

TVM_REGISTER_API("relay._make.Span")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = SpanNode::make(args[0], args[1], args[2]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<SpanNode>([](const SpanNode *node, tvm::IRPrinter *p) {
    p->stream << "SpanNode(" << node->source << ", " << node->lineno << ", "
              << node->col_offset << ")";
  });

}  // namespace relay
}  // namespace tvm
