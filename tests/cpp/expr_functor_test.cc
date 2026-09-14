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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace tvm {
namespace {

class PairExprNode : public ExprNode {
 public:
  Expr left;
  Expr right;
  static void RegisterReflection() {
    ffi::reflection::ObjectDef<PairExprNode>()
        .def_ro("left", &PairExprNode::left)
        .def_ro("right", &PairExprNode::right);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.NativePairExpr", PairExprNode, ExprNode);
};
class PairExpr : public Expr {
 public:
  PairExpr(Expr left, Expr right) {
    auto node = ffi::make_object<PairExprNode>();
    node->left = std::move(left);
    node->right = std::move(right);
    data_ = std::move(node);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PairExpr, Expr, PairExprNode);
};
TVM_FFI_STATIC_INIT_BLOCK() { PairExprNode::RegisterReflection(); }

class Collect : public ExprVisitor {
 public:
  using ExprVisitor::Visit_;
  std::vector<int64_t> values;
  Expected<ffi::Optional<VisitInterrupt>> Visit_(const IntImmNode* node) override {
    values.push_back(node->value);
    return std::nullopt;
  }
};

TEST(ExprVisitor, StructuralFallback) {
  auto visitor = ffi::make_object<Collect>();
  PairExpr pair(IntImm::Int32(1), IntImm::Int32(2));
  auto result = visitor->VisitExpected(pair);
  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  EXPECT_EQ(visitor->values, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(pair->left.as<IntImmNode>()->value, 1);
  EXPECT_EQ(pair->right.as<IntImmNode>()->value, 2);
}

class Rewrite : public ExprMutator {
 public:
  using ExprMutator::Mutate_;
  Expected<UnchangedOr<ffi::Any>> Mutate_(const IntImmNode* node,
                                          ffi::InplaceMode inplace_mode) override {
    return ffi::Any(IntImm::Int32(node->value + 1));
  }
};

TEST(ExprMutator, StructuralFallback) {
  auto mutator = ffi::make_object<Rewrite>();
  PairExpr pair(IntImm::Int32(1), IntImm::Int32(2));
  auto result = mutator->MutateExpected(pair).value();
  ASSERT_FALSE(result.IsUnchanged());
  auto replacement = std::move(result).ValueUnchecked();
  const auto* changed = replacement.as<PairExprNode>();
  ASSERT_NE(changed, nullptr);
  EXPECT_EQ(changed->left.as<IntImmNode>()->value, 2);
  EXPECT_EQ(changed->right.as<IntImmNode>()->value, 3);
  EXPECT_EQ(pair->left.as<IntImmNode>()->value, 1);
  EXPECT_EQ(pair->right.as<IntImmNode>()->value, 2);
}

using ffi::InplaceMode;

class RecordVisits : public Collect {
 public:
  using Collect::VisitExpected;
  std::vector<const ExprNode*> entries;
  bool interrupt = false;
  bool fail = false;

  Expected<ffi::Optional<VisitInterrupt>> VisitExpected(ffi::AnyView value) noexcept override {
    if (const auto* expr = value.as<ExprNode>()) entries.push_back(expr);
    return Collect::VisitExpected(value);
  }
  Expected<ffi::Optional<VisitInterrupt>> Visit_(const IntImmNode* node) override {
    if (fail) throw ffi::Error("ValueError", "visit failed", "");
    if (interrupt) return VisitInterrupt(ffi::Any(17));
    return Collect::Visit_(node);
  }
};

TEST(ExprVisitor, VirtualEntryAndQualifiedParent) {
  auto visitor = ffi::make_object<RecordVisits>();
  for (Expr root : {Expr(Tuple({IntImm::Int32(1), IntImm::Int32(2)})),
                    Expr(PairExpr(IntImm::Int32(1), IntImm::Int32(2)))}) {
    visitor->entries.clear();
    EXPECT_TRUE(visitor->VisitExpected(root).is_ok());
    ASSERT_EQ(visitor->entries.size(), 3);
    EXPECT_EQ(visitor->entries.front(), root.get());
    visitor->entries.clear();
    EXPECT_TRUE(visitor->Collect::VisitExpected(root).is_ok());
    EXPECT_EQ(visitor->entries.size(), 2);
    EXPECT_NE(visitor->entries.front(), root.get());
    visitor->entries.clear();
    ffi::StructuralVisitorObj* raw = visitor.get();
    EXPECT_TRUE(raw->VisitExpected(root).is_ok());
    EXPECT_EQ(visitor->entries.size(), 3);
    EXPECT_EQ(visitor->entries.front(), root.get());
  }
  visitor->interrupt = true;
  PairExpr root(IntImm::Int32(1), IntImm::Int32(2));
  visitor->entries.clear();
  auto stopped = visitor->VisitExpected(root);
  ASSERT_TRUE(stopped.is_ok());
  ASSERT_TRUE(stopped.value().has_value());
  EXPECT_EQ(stopped.value().value()->value.cast<int>(), 17);
  EXPECT_EQ(visitor->entries.size(), 2);
}

class RecordMutations : public Rewrite {
 public:
  using Rewrite::MutateExpected;
  std::vector<const ExprNode*> entries;
  std::vector<InplaceMode> leaf_modes;
  std::vector<InplaceMode> entry_modes;
  bool fail = false;

  Expected<UnchangedOr<ffi::Any>> MutateExpected(
      ffi::AnyView value, InplaceMode mode = InplaceMode::kDisallow) noexcept override {
    if (const auto* expr = value.as<ExprNode>()) {
      entries.push_back(expr);
      entry_modes.push_back(mode);
    }
    return Rewrite::MutateExpected(value, mode);
  }
  Expected<UnchangedOr<ffi::Any>> Mutate_(const IntImmNode* node, InplaceMode mode) override {
    leaf_modes.push_back(mode);
    if (fail) return ffi::Unexpected(ffi::Error("ValueError", "mutation failed", ""));
    return Rewrite::Mutate_(node, mode);
  }
};

TEST(ExprMutator, PermissionAndOwnership) {
  auto mutator = ffi::make_object<RecordMutations>();
  for (InplaceMode mode : {InplaceMode::kDisallow, InplaceMode::kAllow}) {
    for (bool alias_root : {false, true}) {
      Expr root = Tuple(ffi::Array<Expr>{Tuple({IntImm::Int32(1)})});
      ASSERT_NE(root.as<TupleNode>()->fields[0].as<TupleNode>(), nullptr);
      Expr alias = alias_root ? root : Expr();
      const auto* original = root.get();
      const auto* child = root.as<TupleNode>()->fields[0].get();
      mutator->leaf_modes.clear();
      mutator->entry_modes.clear();
      auto result = mutator->MutateExpected(root, mode).value();
      Expr changed = std::move(result).ValueOrUnchanged(ffi::AnyView(root)).cast<Expr>();
      EXPECT_EQ(mutator->entry_modes.front(), mode);
      bool reuse = mode == InplaceMode::kAllow && !alias_root;
      EXPECT_EQ(changed.get() == original, reuse);
      EXPECT_EQ(changed.as<TupleNode>()->fields[0].get() == child, reuse);
      EXPECT_EQ(mutator->leaf_modes,
                std::vector<InplaceMode>{reuse ? InplaceMode::kAllow : InplaceMode::kDisallow});
      if (!reuse) {
        EXPECT_EQ(
            root.as<TupleNode>()->fields[0].as<TupleNode>()->fields[0].as<IntImmNode>()->value, 1);
      }
      EXPECT_EQ(
          changed.as<TupleNode>()->fields[0].as<TupleNode>()->fields[0].as<IntImmNode>()->value, 2);
    }
  }
  Expr child = Tuple({IntImm::Int32(1)});
  Expr root = Tuple({child});
  const auto* original = root.get();
  mutator->leaf_modes.clear();
  auto result = mutator->MutateExpected(root, InplaceMode::kAllow).value();
  EXPECT_TRUE(result.IsUnchanged());
  EXPECT_EQ(root.get(), original);
  EXPECT_NE(root.as<TupleNode>()->fields[0].get(), child.get());
  EXPECT_EQ(child.as<TupleNode>()->fields[0].as<IntImmNode>()->value, 1);
  EXPECT_EQ(mutator->leaf_modes, std::vector<InplaceMode>{InplaceMode::kDisallow});
  // Reflected fallback remains copy-on-write, including below a unique native root.
  root = Tuple({PairExpr(IntImm::Int32(1), IntImm::Int32(2))});
  mutator->leaf_modes.clear();
  EXPECT_TRUE(mutator->MutateExpected(root, InplaceMode::kAllow).value().IsUnchanged());
  EXPECT_EQ(mutator->leaf_modes,
            (std::vector<InplaceMode>{InplaceMode::kDisallow, InplaceMode::kDisallow}));
}

TEST(ExprMutator, VirtualEntryAndQualifiedParent) {
  auto mutator = ffi::make_object<RecordMutations>();
  for (Expr root : {Expr(Tuple({IntImm::Int32(1), IntImm::Int32(2)})),
                    Expr(PairExpr(IntImm::Int32(1), IntImm::Int32(2)))}) {
    mutator->entries.clear();
    auto result = mutator->MutateExpected(root);
    ASSERT_TRUE(result.is_ok());
    EXPECT_EQ(mutator->entries.size(), 3);
    EXPECT_EQ(mutator->entries.front(), root.get());
    mutator->entries.clear();
    EXPECT_TRUE(mutator->Rewrite::MutateExpected(root).is_ok());
    EXPECT_EQ(mutator->entries.size(), 2);
    EXPECT_NE(mutator->entries.front(), root.get());
    mutator->entries.clear();
    ffi::StructuralMutatorObj* raw = mutator.get();
    EXPECT_TRUE(raw->MutateExpected(root).is_ok());
    EXPECT_EQ(mutator->entries.size(), 3);
    EXPECT_EQ(mutator->entries.front(), root.get());
  }
  // The FFI base selects both raw mode slots before the native virtual entry.
  for (InplaceMode mode : {InplaceMode::kDisallow, InplaceMode::kAllow}) {
    Expr leaf = IntImm::Int32(1);
    mutator->entries.clear();
    mutator->leaf_modes.clear();
    ffi::StructuralMutatorObj* raw = mutator.get();
    EXPECT_TRUE(raw->MutateExpected(leaf, mode).is_ok());
    EXPECT_EQ(mutator->entries, std::vector<const ExprNode*>{leaf.get()});
    EXPECT_EQ(mutator->leaf_modes, std::vector<InplaceMode>{mode});
  }
}

TEST(ExprMutator, TrustedDefaultAfterAcquiringReference) {
  class DefaultAfterRef : public Rewrite {
   public:
    using Rewrite::Mutate_;
    Expected<UnchangedOr<ffi::Any>> Mutate_(const TupleNode* node, InplaceMode mode) override {
      auto owning = ffi::GetRef<Tuple>(node);
      EXPECT_FALSE(node->unique());
      return ffi::StructuralMutatorObj::DefaultMutateExpected(owning, mode);
    }
  };
  auto mutator = ffi::make_object<DefaultAfterRef>();
  Expr root = Tuple({IntImm::Int32(1)});
  const auto* original = root.get();
  auto result = mutator->MutateExpected(root, InplaceMode::kAllow).value();
  Expr changed = std::move(result).ValueOrUnchanged(ffi::AnyView(root)).cast<Expr>();
  EXPECT_EQ(changed.get(), original);
  EXPECT_EQ(changed.as<TupleNode>()->fields[0].as<IntImmNode>()->value, 2);
}

TEST(ExprMutator, NullAndInlineResults) {
  class NullEntry : public ExprMutator {
   public:
    using ExprMutator::MutateExpected;
    int calls = 0;
    bool replace = false;
    bool fail = false;
    Expected<UnchangedOr<ffi::Any>> MutateExpected(
        ffi::AnyView value, InplaceMode mode = InplaceMode::kDisallow) noexcept override {
      ++calls;
      if (fail) return ffi::Unexpected(ffi::Error("ValueError", "null failed", ""));
      if (replace) return ffi::Any(IntImm::Int32(7));
      return ExprMutator::MutateExpected(value, mode);
    }
  };
  auto mutator = ffi::make_object<NullEntry>();
  auto check_none = [](Expected<UnchangedOr<ffi::Any>> result) {
    ASSERT_TRUE(result.is_ok());
    auto none = std::move(result).value();
    ASSERT_FALSE(none.IsUnchanged());
    EXPECT_EQ(std::move(none).ValueUnchecked().type_index(), ffi::TypeIndex::kTVMFFINone);
  };
  check_none(mutator->MutateExpected(Expr()));
  check_none(mutator->MutateExpected(ffi::ObjectRef()));
  check_none(mutator->MutateExpected(ffi::AnyView(nullptr)));
  EXPECT_EQ(mutator->calls, 3);
  auto integer = mutator->MutateExpected(ffi::AnyView(9)).value();
  ASSERT_FALSE(integer.IsUnchanged());
  EXPECT_EQ(std::move(integer).ValueUnchecked().cast<int>(), 9);
  mutator->replace = true;
  EXPECT_EQ(mutator->MutateExpected(Expr()).value().ValueUnchecked().as<IntImmNode>()->value, 7);
  EXPECT_EQ(
      mutator->MutateExpected(ffi::ObjectRef()).value().ValueUnchecked().as<IntImmNode>()->value,
      7);
  mutator->fail = true;
  EXPECT_TRUE(mutator->MutateExpected(Expr()).is_err());
  EXPECT_TRUE(mutator->MutateExpected(ffi::ObjectRef()).is_err());
  auto identity = ffi::make_object<ExprMutator>();
  Expr leaf = IntImm::Int32(1);
  EXPECT_TRUE(identity->MutateExpected(leaf).value().IsUnchanged());
}

TEST(ExprVisitor, ErrorContextThroughRawBridge) {
  auto visitor = ffi::make_object<RecordVisits>();
  auto mutator = ffi::make_object<RecordMutations>();
  visitor->fail = true;
  mutator->fail = true;
  for (Expr root : {Expr(IntImm::Int32(1)), Expr(PairExpr(IntImm::Int32(1), IntImm::Int32(2)))}) {
    ffi::StructuralVisitorObj* raw_visitor = visitor.get();
    ffi::StructuralMutatorObj* raw_mutator = mutator.get();
    auto visit = raw_visitor->VisitExpected(root);
    auto mutate = raw_mutator->MutateExpected(root);
    ASSERT_TRUE(visit.is_err());
    ASSERT_TRUE(mutate.is_err());
    for (const ffi::Error& error : {visit.error(), mutate.error()}) {
      auto context = ffi::VisitErrorContext::TryGetFromError(error);
      ASSERT_TRUE(context.has_value());
      const auto& path = context.value()->reverse_visit_pattern;
      ASSERT_EQ(path.size(), root.as<PairExprNode>() ? 2 : 1);
      EXPECT_TRUE(path[path.size() - 1].same_as(root));
    }
  }
}

TEST(ExprMutator, RawAbiAndManagedDestruction) {
  using RawMutate = TVMFFIAny (*)(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept;
  using Table = ffi::StructuralMutatorVTable;
  using RawVisit = TVMFFIAny (*)(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept;
  using VisitorTable = ffi::StructuralVisitorVTable;
  static_assert(std::is_same_v<ffi::FStructuralVisit, RawVisit>);
  static_assert(std::is_same_v<decltype(VisitorTable::visit), RawVisit>);
  static_assert(std::is_standard_layout_v<VisitorTable>);
  static_assert(offsetof(VisitorTable, visit) == 0);
  static_assert(sizeof(VisitorTable) == sizeof(RawVisit));
  static_assert(std::is_same_v<ffi::FStructuralMutate, RawMutate>);
  static_assert(std::is_same_v<decltype(Table::mutate), RawMutate>);
  static_assert(std::is_same_v<decltype(Table::maybe_inplace_mutate), RawMutate>);
  static_assert(std::is_same_v<decltype(Table::var_remap_get), ffi::FStructuralVarRemapGet>);
  static_assert(std::is_same_v<decltype(Table::var_remap_set), ffi::FStructuralVarRemapSet>);
  static_assert(std::is_standard_layout_v<Table>);
  static_assert(offsetof(Table, mutate) == 0);
  static_assert(offsetof(Table, maybe_inplace_mutate) == sizeof(RawMutate));
  static_assert(offsetof(Table, var_remap_get) == 2 * sizeof(RawMutate));
  static_assert(offsetof(Table, var_remap_set) ==
                2 * sizeof(RawMutate) + sizeof(ffi::FStructuralVarRemapGet));
  static_assert(sizeof(Table) == 2 * sizeof(RawMutate) + sizeof(ffi::FStructuralVarRemapGet) +
                                     sizeof(ffi::FStructuralVarRemapSet));
  static_assert(!std::is_polymorphic_v<ffi::StructuralMutatorObj>);
  static_assert(!std::is_polymorphic_v<ffi::StructuralVisitorObj>);
  class OwnedMutator : public ExprMutator {
   public:
    explicit OwnedMutator(bool* destroyed) : destroyed_(destroyed) {}
    ~OwnedMutator() { *destroyed_ = true; }
    bool* destroyed_;
  };
  class OwnedVisitor : public ExprVisitor {
   public:
    explicit OwnedVisitor(bool* destroyed) : destroyed_(destroyed) {}
    ~OwnedVisitor() { *destroyed_ = true; }
    bool* destroyed_;
  };
  bool mutator_destroyed = false;
  bool visitor_destroyed = false;
  {
    ffi::ObjectPtr<ffi::StructuralMutatorObj> mutator =
        ffi::make_object<OwnedMutator>(&mutator_destroyed);
    ffi::ObjectPtr<ffi::StructuralVisitorObj> visitor =
        ffi::make_object<OwnedVisitor>(&visitor_destroyed);
    EXPECT_TRUE(mutator->MutateExpected(ffi::AnyView(1)).is_ok());
    EXPECT_TRUE(visitor->VisitExpected(ffi::AnyView(nullptr)).is_ok());
  }
  EXPECT_TRUE(mutator_destroyed);
  EXPECT_TRUE(visitor_destroyed);
}

}  // namespace
}  // namespace tvm
