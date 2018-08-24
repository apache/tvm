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
.set_dispatch<SourceNameNode>([](const SourceNameNode *node, tvm::IRPrinter *p) {
    p->stream << node->name;
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<SpanNode>([](const SpanNode *node, tvm::IRPrinter *p) {
    p->stream << "SpanNode(" << node->source << ", " << node->lineno << ", "
              << node->col_offset << ")";
  });

}  // namespace relay
}  // namespace tvm
