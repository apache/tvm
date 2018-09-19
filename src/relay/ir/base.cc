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

SourceName SourceNameNode::make(std::string name) {
  std::shared_ptr<SourceNameNode> n = std::make_shared<SourceNameNode>();
  n->name = std::move(name);
  return SourceName(n);
}

std::shared_ptr<SourceNameNode> CreateSourceName(const std::string& name) {
  SourceName sn = SourceName::Get(name);
  CHECK(!sn.defined()) << "Cannot find source name \'" << name << '\'';
  std::shared_ptr<Node> node = sn.node_;
  return std::dynamic_pointer_cast<SourceNameNode>(node);
}

const SourceName& SourceName::Get(const std::string& name) {
  static std::unordered_map<std::string, SourceName> source_map;

  auto sn = source_map.find(name);
  if (sn == source_map.end()) {
    auto source_name = SourceNameNode::make(name);
    source_map.insert({name, source_name});
    return source_map.at(name);
  } else {
    return sn->second;
  }
}

TVM_REGISTER_API("relay._make.SourceName")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue *ret) {
      *ret = SourceNameNode::make(args[0]);
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<SourceNameNode>([](const SourceNameNode *node, tvm::IRPrinter *p) {
      p->stream << "SourceNameNode(" << node->name << ", " << node << ")";
    });

TVM_REGISTER_NODE_TYPE(SourceNameNode)
.set_creator(CreateSourceName)
.set_global_key([](const Node* n) {
    return static_cast<const SourceNameNode*>(n)->name;
  });

Span SpanNode::make(SourceName source, int lineno, int col_offset) {
  std::shared_ptr<SpanNode> n = std::make_shared<SpanNode>();
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
