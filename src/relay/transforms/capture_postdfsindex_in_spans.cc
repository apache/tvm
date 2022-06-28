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
 * \file tvm/relay/transform/capture_index_in_spans.cc
 * \brief A pass to set spans to capture the post-dfs index of every node.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../ir/indexed_graph.h"

namespace tvm {
namespace relay {
namespace transform {

namespace {

/*! \brief Update all the spans to capture their post-dfs index. */
class SpansRewriter : public ExprRewriter {
 public:
  explicit SpansRewriter(const IndexedGraph<Expr>* indexed_graph)
      : source_name_(SourceName::Get("index")), indexed_graph_(indexed_graph) {}

 private:
  Expr Rewrite_(const VarNode* var_node, const Expr& post) final {
    return WithFields(Downcast<Var>(post), {}, {}, {}, MakeSpan(GetRef<Var>(var_node)));
  }

  Expr Rewrite_(const GlobalVarNode* global_var_node, const Expr& post) final {
    return WithFields(Downcast<GlobalVar>(post), {}, {}, {},
                      MakeSpan(GetRef<GlobalVar>(global_var_node)));
  }

  Expr Rewrite_(const ConstantNode* constant_node, const Expr& post) final {
    return WithFields(Downcast<Constant>(post), {}, {}, MakeSpan(GetRef<Constant>(constant_node)));
  }

  Expr Rewrite_(const TupleNode* tuple_node, const Expr& post) final {
    return WithFields(Downcast<Tuple>(post), {}, {}, MakeSpan(GetRef<Tuple>(tuple_node)));
  }

  Expr Rewrite_(const FunctionNode* function_node, const Expr& post) final {
    return WithFields(Downcast<Function>(post), {}, {}, {}, {}, {}, {},
                      MakeSpan(GetRef<Function>(function_node)));
  }

  Expr Rewrite_(const CallNode* call_node, const Expr& post) final {
    return WithFields(Downcast<Call>(post), {}, {}, {}, {}, {}, MakeSpan(GetRef<Call>(call_node)));
  }

  Expr Rewrite_(const LetNode* let_node, const Expr& post) final {
    return WithFields(Downcast<Let>(post), {}, {}, {}, {}, MakeSpan(GetRef<Let>(let_node)));
  }

  Expr Rewrite_(const IfNode* if_node, const Expr& post) final {
    return WithFields(Downcast<If>(post), {}, {}, {}, {}, MakeSpan(GetRef<If>(if_node)));
  }

  // OpNodes are not rewritten.

  Expr Rewrite_(const TupleGetItemNode* tuple_get_item_node, const Expr& post) final {
    return WithFields(Downcast<TupleGetItem>(post), {}, {}, {},
                      MakeSpan(GetRef<TupleGetItem>(tuple_get_item_node)));
  }

  Expr Rewrite_(const RefCreateNode* ref_create_node, const Expr& post) final {
    return WithFields(Downcast<RefCreate>(post), {}, {},
                      MakeSpan(GetRef<RefCreate>(ref_create_node)));
  }

  Expr Rewrite_(const RefReadNode* ref_read_node, const Expr& post) final {
    return WithFields(Downcast<RefRead>(post), {}, {}, MakeSpan(GetRef<RefRead>(ref_read_node)));
  }

  Expr Rewrite_(const RefWriteNode* ref_write_node, const Expr& post) final {
    return WithFields(Downcast<RefWrite>(post), {}, {}, {},
                      MakeSpan(GetRef<RefWrite>(ref_write_node)));
  }

  // ConstructorNodes are  not rewritten.

  Expr Rewrite_(const MatchNode* match_node, const Expr& post) final {
    return WithFields(Downcast<Match>(post), {}, {}, {}, MakeSpan(GetRef<Match>(match_node)));
  }

  Span MakeSpan(const Expr& expr) {
    auto node = indexed_graph_->item_to_node(expr);
    int node_index = static_cast<int>(node->index_);
    int dominator_index =
        node->dominator_parent_ ? static_cast<int>(node->dominator_parent_->index_) : -1;
    Span span(source_name_, /*line=*/node_index, /*end_line=*/node_index,
              /*column=*/dominator_index, /*end_column=*/dominator_index);
    return span;
  }

  SourceName source_name_;
  const IndexedGraph<Expr>* indexed_graph_;
};

}  // namespace

tvm::transform::Pass CapturePostDfsIndexInSpans() {
  auto pass_func = [](Function f, IRModule m, transform::PassContext ctxt) {
    std::unique_ptr<IndexedGraph<Expr>> indexed_graph = CreateIndexedGraph(f);
    SpansRewriter rewriter(indexed_graph.get());
    return Downcast<Function>(PostOrderRewrite(f, &rewriter));
  };
  return CreateFunctionPass(pass_func, 0, "CapturePostDfsIndexInSpans", {});
}

TVM_REGISTER_GLOBAL("relay._transform.CapturePostDfsIndexInSpans")
    .set_body_typed(CapturePostDfsIndexInSpans);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
