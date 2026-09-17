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
#include <gtest/gtest.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/op.h>

#include <algorithm>
#include <string>
#include <vector>

namespace tvm {
namespace s_tir {
namespace {

using namespace tirx;

TEST(STIRFunctor, LegacyInheritedDispatchAndContainsNode) {
  class Dispatch : public StmtFunctor<int(const Stmt&, int)> {
   public:
    using StmtFunctor::VisitStmt_;
    int VisitStmt_(const SBlockNode*, int value) final { return value + 1; }
    int VisitStmt_(const SBlockRealizeNode*, int value) final { return value + 2; }
    int VisitStmt_(const EvaluateNode*, int value) final { return value + 3; }
  } dispatch;
  Stmt body = Evaluate(0);
  SBlock block({}, {}, {}, "block", body);
  Stmt realize = SBlockRealize({}, IntImm::Bool(true), block);
  EXPECT_EQ(dispatch(block, 10), 11);
  EXPECT_EQ(dispatch(realize, 10), 12);
  EXPECT_EQ(dispatch(body, 10), 13);
  EXPECT_TRUE(ContainsNode<SBlockNode>(realize));
  EXPECT_TRUE(ContainsNode<EvaluateNode>(realize));
  EXPECT_FALSE(ContainsNode<ForNode>(realize));
}

TEST(STIRFunctor, NativeBlockOverrideReusesInheritedCoreHooks) {
  class Visitor : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;
    ffi::Optional<VisitInterrupt> Visit_(const SBlockNode* op) final {
      ++blocks;
      return StmtExprVisitor::Visit_(op);
    }
    ffi::Optional<VisitInterrupt> Visit_(const EvaluateNode* op) final {
      ++evaluates;
      return StmtExprVisitor::Visit_(op);
    }
    int blocks = 0;
    int evaluates = 0;
  };
  SBlock block({}, {}, {}, "block", Evaluate(0));
  auto visitor = ffi::make_object<Visitor>();
  visitor->Visit(SBlockRealize({}, IntImm::Bool(true), block));
  EXPECT_EQ(visitor->blocks, 1);
  EXPECT_EQ(visitor->evaluates, 1);
}

TEST(STIRFunctor, NativeVisitPreservesBlockOrderAndBinders) {
  PrimVar index("index"), extent("extent"), annotation("annotation");
  BufferVar buffer = decl_buffer({16});
  BufferRegion region(buffer, {Range::FromMinExtent(0, 16)});
  IterVar iter(Range::FromMinExtent(0, extent), index, IterVarType::kDataPar);
  SBlock block({iter}, {region}, {}, "block", Evaluate(index), std::nullopt, {buffer}, {},
               {{"annotation", annotation}});
  class Visitor : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;
    ffi::Optional<VisitInterrupt> Visit_(const VarNode* var) final {
      vars.push_back(var);
      if (var->ty.as<BufferTypeNode>()) {
        buffer_regions.push_back(def_region_kind());
      }
      return std::nullopt;
    }
    std::vector<const VarNode*> vars;
    std::vector<TVMFFIDefRegionKind> buffer_regions;
  };
  auto visitor = ffi::make_object<Visitor>();
  visitor->Visit(block);
  EXPECT_EQ(std::count(visitor->vars.begin(), visitor->vars.end(), index.get()), 1);
  EXPECT_EQ(std::count(visitor->vars.begin(), visitor->vars.end(), annotation.get()), 0);
  ASSERT_EQ(visitor->buffer_regions.size(), 2);
  EXPECT_EQ(visitor->buffer_regions[0], kTVMFFIDefRegionKindSimple);
  EXPECT_EQ(visitor->buffer_regions[1], kTVMFFIDefRegionKindNone);

  // A full structural walk retains its distinct binder/annotation traversal.
  int structural_index = 0;
  int structural_annotation = 0;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      block, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        structural_index += var.same_as(index);
        structural_annotation += var.same_as(annotation);
        return ffi::WalkResult::Advance();
      });
  EXPECT_EQ(structural_index, 2);
  EXPECT_EQ(structural_annotation, 1);
}

TEST(STIRFunctor, NativeMutationKeepsAnnotationsAndSharedIteratorBinders) {
  PrimVar index("index"), extent("extent");
  PrimExpr expression = extent + 1;
  IterVar iter(Range::FromMinExtent(0, expression), index, IterVarType::kDataPar);
  SBlock block({iter}, {}, {}, "block", Evaluate(expression), std::nullopt, {}, {},
               {{"annotation", expression}});
  SBlock retained = block;
  class Mutator : public StmtExprMutator {
   public:
    using StmtExprMutator::Mutate_;
    UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* op, InplaceMode) final { return op->a; }
  };
  auto mutator = ffi::make_object<Mutator>();
  Stmt result = mutator->Mutate(block, InplaceMode::kAllow).ValueOrUnchanged(block);
  const auto* changed = result.as<SBlockNode>();
  ASSERT_NE(changed, nullptr);
  EXPECT_NE(changed, retained.get());
  EXPECT_TRUE(changed->iter_vars[0]->var.same_as(index));
  EXPECT_TRUE(changed->iter_vars[0]->dom->extent.same_as(extent));
  EXPECT_TRUE(changed->body.as<EvaluateNode>()->value.same_as(extent));
  EXPECT_TRUE(changed->annotations.at("annotation").cast<PrimExpr>().same_as(expression));
  EXPECT_TRUE(retained->iter_vars[0]->dom->extent.same_as(expression));
  EXPECT_TRUE(iter->dom->extent.same_as(expression));

  // A sole owning block can update in place while its shared iterator is copied.
  SBlock unique({iter}, {}, {}, "unique", Evaluate(expression));
  const auto* original = unique.get();
  auto update = mutator->Mutate(unique, InplaceMode::kAllow);
  EXPECT_TRUE(update.IsUnchanged());
  EXPECT_EQ(unique.get(), original);
  EXPECT_TRUE(unique->iter_vars[0]->dom->extent.same_as(extent));
  EXPECT_TRUE(iter->dom->extent.same_as(expression));
}

template <typename Base>
void CheckMutationRemapsBufferDefinitionsAndUses() {
  PrimVar extent("extent");
  BufferVar allocated = decl_buffer({extent + 1}, PrimType::Int(32));
  BufferVar matched = decl_buffer({extent + 1}, PrimType::Int(32));
  BufferRegion region(allocated, {Range::FromMinExtent(0, extent + 1)});
  MatchBufferRegion match(matched, region);
  Stmt body = SeqStmt({BufferStore(allocated, 0, {0}), BufferStore(matched, 0, {0})});
  SBlock block({}, {region}, {region}, "block", body, std::nullopt, {allocated}, {match});
  class Mutator : public Base {
   public:
    using Base::Mutate_;
    UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* op, InplaceMode) final { return op->a; }
  };
  auto mutator = ffi::make_object<Mutator>();
  Stmt result = mutator->Mutate(block).ValueOrUnchanged(block);
  const auto* changed = result.as<SBlockNode>();
  ASSERT_NE(changed, nullptr);
  BufferVar new_allocated = changed->alloc_buffers[0];
  BufferVar new_matched = changed->match_buffers[0]->buffer;
  EXPECT_FALSE(new_allocated.same_as(allocated));
  EXPECT_FALSE(new_matched.same_as(matched));
  EXPECT_TRUE(new_allocated->shape[0].same_as(extent));
  EXPECT_TRUE(new_matched->shape[0].same_as(extent));
  EXPECT_TRUE(changed->reads[0]->buffer.same_as(new_allocated));
  EXPECT_TRUE(changed->writes[0]->buffer.same_as(new_allocated));
  EXPECT_TRUE(changed->match_buffers[0]->source->buffer.same_as(new_allocated));
  const auto* statements = changed->body.as<SeqStmtNode>();
  ASSERT_NE(statements, nullptr);
  EXPECT_TRUE(statements->seq[0].as<BufferStoreNode>()->buffer.same_as(new_allocated));
  EXPECT_TRUE(statements->seq[1].as<BufferStoreNode>()->buffer.same_as(new_matched));
  EXPECT_TRUE(block->alloc_buffers[0].same_as(allocated));
  EXPECT_TRUE(block->match_buffers[0]->buffer.same_as(matched));
}

TEST(STIRFunctor, NativeMutationRemapsBufferDefinitionsAndUses) {
  CheckMutationRemapsBufferDefinitionsAndUses<StmtExprMutator>();
}

TEST(STIRFunctor, GenericTIRXMutationRemapsBufferDefinitionsAndUses) {
  CheckMutationRemapsBufferDefinitionsAndUses<tirx::StmtExprMutator>();
}

TEST(STIRFunctor, GenericTIRXVisitorUsesRegisteredNativePolicy) {
  PrimVar index("index"), annotation("annotation");
  BufferVar buffer = decl_buffer({16});
  BufferRegion region(buffer, {Range::FromMinExtent(0, 16)});
  IterVar iter(Range::FromMinExtent(0, 16), index, IterVarType::kDataPar);
  SBlock block({iter}, {region}, {}, "block", Evaluate(index), std::nullopt, {buffer}, {},
               {{"annotation", annotation}});
  class Visitor : public tirx::StmtExprVisitor {
   public:
    using tirx::StmtExprVisitor::Visit_;
    ffi::Optional<VisitInterrupt> Visit_(const VarNode* var) final {
      vars.push_back(var);
      if (var->ty.as<BufferTypeNode>()) {
        buffer_regions.push_back(def_region_kind());
      }
      return std::nullopt;
    }
    std::vector<const VarNode*> vars;
    std::vector<TVMFFIDefRegionKind> buffer_regions;
  };
  auto visitor = ffi::make_object<Visitor>();
  visitor->Visit(block);
  EXPECT_EQ(std::count(visitor->vars.begin(), visitor->vars.end(), index.get()), 1);
  EXPECT_EQ(std::count(visitor->vars.begin(), visitor->vars.end(), annotation.get()), 0);
  ASSERT_EQ(visitor->buffer_regions.size(), 2);
  EXPECT_EQ(visitor->buffer_regions[0], kTVMFFIDefRegionKindSimple);
  EXPECT_EQ(visitor->buffer_regions[1], kTVMFFIDefRegionKindNone);
}

TEST(STIRFunctor, GenericTIRXMutationPreservesBindersAndAnnotations) {
  PrimVar index("index"), replacement("replacement"), extent("extent");
  PrimExpr expression = extent + 1;
  IterVar iter(Range::FromMinExtent(0, expression), index, IterVarType::kDataPar);
  BufferVar buffer = decl_buffer({expression}, PrimType::Int(32));
  SBlock block({iter}, {}, {}, "block", BufferStore(buffer, index, {0}), std::nullopt, {buffer}, {},
               {{"annotation", expression}});
  class Mutator : public tirx::StmtExprMutator {
   public:
    using tirx::StmtExprMutator::Mutate_;
    UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* op, InplaceMode) final { return op->a; }
  };
  auto mutator = ffi::make_object<Mutator>();
  mutator->VarRemapSet(index, replacement);
  SBlock retained = block;
  Stmt result = mutator->Mutate(block, InplaceMode::kAllow).ValueOrUnchanged(block);
  const auto* changed = result.as<SBlockNode>();
  ASSERT_NE(changed, nullptr);
  EXPECT_TRUE(changed->iter_vars[0]->var.same_as(index));
  EXPECT_TRUE(changed->iter_vars[0]->dom->extent.same_as(extent));
  EXPECT_TRUE(changed->annotations.at("annotation").cast<PrimExpr>().same_as(expression));
  EXPECT_TRUE(changed->body.as<BufferStoreNode>()->value.same_as(replacement));
  EXPECT_TRUE(changed->body.as<BufferStoreNode>()->buffer.same_as(changed->alloc_buffers[0]));
  EXPECT_TRUE(changed->alloc_buffers[0]->shape[0].same_as(extent));
  EXPECT_TRUE(block->iter_vars[0]->var.same_as(index));
  EXPECT_TRUE(block->annotations.at("annotation").cast<PrimExpr>().same_as(expression));

  SBlock unique({}, {}, {}, "unique", Evaluate(expression), std::nullopt, {}, {},
                {{"annotation", expression}});
  const auto* original = unique.get();
  auto update = mutator->Mutate(unique, InplaceMode::kAllow);
  EXPECT_TRUE(update.IsUnchanged());
  EXPECT_EQ(unique.get(), original);
  EXPECT_TRUE(unique->body.as<EvaluateNode>()->value.same_as(extent));
  EXPECT_TRUE(unique->annotations.at("annotation").cast<PrimExpr>().same_as(expression));
}

TEST(STIRFunctor, GenericTIRXPolicyPreservesInterruptAndErrorIdentity) {
  PrimVar annotation("annotation"), body("body");
  SBlock block({}, {}, {}, "block", Evaluate(body), std::nullopt, {}, {},
               {{"annotation", annotation}});
  class Visitor : public tirx::StmtExprVisitor {
   public:
    using tirx::StmtExprVisitor::Visit_;
    ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
      ++count;
      return VisitInterrupt(ffi::GetRef<Var>(op));
    }
    int count = 0;
  };
  auto visitor = ffi::make_object<Visitor>();
  auto interrupt = visitor->Visit(block);
  ASSERT_TRUE(interrupt.has_value());
  EXPECT_TRUE(interrupt.value()->value.cast<Var>().same_as(body));
  EXPECT_EQ(visitor->count, 1);

  class Mutator : public tirx::StmtExprMutator {
   public:
    using tirx::StmtExprMutator::Mutate_;
    ffi::Error error{"ValueError", "block child mutation error", ""};
    UnchangedOr<Expr> Mutate_(const VarNode*, InplaceMode) final { throw error; }
  };
  auto mutator = ffi::make_object<Mutator>();
  auto result = mutator->MutateExpected(block);
  ASSERT_TRUE(result.is_err());
  EXPECT_TRUE(result.error().same_as(mutator->error));
  auto context = ffi::VisitErrorContext::TryGetFromError(result.error());
  ASSERT_TRUE(context.has_value());
  EXPECT_TRUE(context.value()->reverse_visit_pattern.back().same_as(block));
}

TEST(STIRFunctor, NativeInterruptStopsBeforeBlockBody) {
  PrimVar stop("stop"), body("body");
  IterVar iter(Range::FromMinExtent(0, 16), PrimVar("index"), IterVarType::kDataPar);
  SBlock block({iter}, {}, {}, "block", Evaluate(body));
  Stmt realize = SBlockRealize({stop}, IntImm::Bool(true), block);
  class Visitor : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;
    ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
      ++count;
      return VisitInterrupt(ffi::GetRef<Var>(op));
    }
    int count = 0;
  };
  auto visitor = ffi::make_object<Visitor>();
  auto interrupt = visitor->Visit(realize);
  ASSERT_TRUE(interrupt.has_value());
  EXPECT_TRUE(interrupt.value()->value.cast<Var>().same_as(stop));
  EXPECT_EQ(visitor->count, 1);
}

}  // namespace
}  // namespace s_tir
}  // namespace tvm
