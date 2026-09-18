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
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include <memory>

namespace tvm {
namespace tirx {
namespace {

class ExtensionStmtNode : public StmtNode {
 public:
  static constexpr uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("test.ExtensionStmt", ExtensionStmtNode, StmtNode);
};

class ChildExtensionStmtNode : public ExtensionStmtNode {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.ChildExtensionStmt", ChildExtensionStmtNode,
                                    ExtensionStmtNode);
};

using ForwardFunctor = StmtFunctor<int(const Stmt&, std::unique_ptr<int>, int&)>;

class ExtendedFunctor : public ForwardFunctor {
 public:
  using ForwardFunctor::Dispatch_;
  ExtendedFunctor() : ForwardFunctor(Table()) {}

  int Dispatch_(const EvaluateNode*, std::unique_ptr<int> value, int& count) override {
    ++count;
    return *value;
  }
  virtual int Dispatch_(const ExtensionStmtNode*, std::unique_ptr<int> value, int& count) {
    ++count;
    return *value + 10;
  }
  int DispatchDefault_(const ffi::Object*, std::unique_ptr<int> value, int& count) override {
    ++count;
    return -*value;
  }

 private:
  static const VTable* Table() {
    static const VTable table = [] {
      VTable table;
      ForwardFunctor::InitVTable(&table);
      SetDispatch<ExtendedFunctor, ExtensionStmtNode>(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
};

TEST(StmtFunctor, ExtendedTableForwardingAndAncestorDispatch) {
  ExtendedFunctor functor;
  int calls = 0;
  auto value = std::make_unique<int>(3);
  EXPECT_EQ(functor(Evaluate(prim::IntImm::Int32(1)), std::move(value), calls), 3);
  EXPECT_EQ(value, nullptr);
  EXPECT_EQ(calls, 1);

  Stmt extension(ffi::make_object<ExtensionStmtNode>());
  EXPECT_EQ(functor.Dispatch(extension, std::make_unique<int>(4), calls), 14);
  Stmt child(ffi::make_object<ChildExtensionStmtNode>());
  EXPECT_EQ(functor(child, std::make_unique<int>(5), calls), 15);
  EXPECT_EQ(calls, 3);

  // An inherited default hook receives the same move-only and reference arguments.
  EXPECT_EQ(functor(Break(Span()), std::make_unique<int>(6), calls), -6);
  EXPECT_EQ(calls, 4);
  // Extending one finalized table does not alter the base signature's default table.
  ForwardFunctor base;
  EXPECT_THROW(base(extension, std::make_unique<int>(1), calls), ffi::Error);
  EXPECT_EQ(calls, 4);
}

TEST(StmtFunctor, EntryPointLifetimeAndNoRecursion) {
  class Functor : public StmtFunctor<int(const Stmt&)> {
   public:
    explicit Functor(bool* destroyed) : destroyed_(destroyed) {}
    ~Functor() override { *destroyed_ = true; }
    int Dispatch(const Stmt& node) override { return 10 + StmtFunctor::Dispatch(node); }
    int Dispatch_(const SeqStmtNode*) override { return 2; }

   private:
    bool* destroyed_;
  };
  bool destroyed = false;
  {
    std::unique_ptr<StmtFunctor<int(const Stmt&)>> functor = std::make_unique<Functor>(&destroyed);
    // Evaluate would throw if the dispatcher recursed into the sequence.
    EXPECT_EQ(
        (*functor)(SeqStmt({Evaluate(prim::IntImm::Int32(1)), Evaluate(prim::IntImm::Int32(2))})),
        12);
    EXPECT_THROW((*functor)(Evaluate(prim::IntImm::Int32(1))), ffi::Error);
    EXPECT_THROW((*functor)(Stmt(nullptr)), ffi::Error);
  }
  EXPECT_TRUE(destroyed);
}

TEST(StmtFunctor, ReferenceResult) {
  class Functor : public StmtFunctor<int&(const Stmt&, int&)> {
   public:
    int& Dispatch_(const EvaluateNode*, int& value) override { return value; }
  };
  Functor functor;
  int value = 3;
  functor(Evaluate(prim::IntImm::Int32(1)), value) = 7;
  EXPECT_EQ(value, 7);
}

}  // namespace
}  // namespace tirx
}  // namespace tvm
