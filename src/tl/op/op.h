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
 * \file tl/op/op.h
 * \brief Tile library operations.
 *
 */

#ifndef TVM_TL_OP_OP_H_
#define TVM_TL_OP_OP_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tir;

using OpBuilderFunc = TypedPackedFunc<void*(const Array<PrimExpr>&, const Map<Var, Buffer>&)>;

#define TIR_REGISTER_TL_OP(Entry, OpName)                                                \
  const Op& Entry::Get() {                                                               \
    static const Op& op = Op::Get("tl." #OpName);                                        \
    return op;                                                                           \
  }                                                                                      \
  TVM_REGISTER_OP("tl." #OpName)                                                         \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)                       \
      .set_attr<OpBuilderFunc>("TLOpBuilder",                                            \
                               [](const Array<PrimExpr>& a, const Map<Var, Buffer>& b) { \
                                 return (void*)(new Entry(a, b));                        \
                               })

enum class InferLevel {
  kFree = 0,
  kCommon = 1,
  kStrict = 2,
};

using AddWorkspaceCallback = std::function<PrimExpr(int, DataType)>;
using LayoutMap = Map<Buffer, Layout>;

struct LowerArgs {
  Target target;
  size_t block_size;
  Var thread_var;
  AddWorkspaceCallback AddWorkspace;
  LayoutMap layout_map;
};

struct LayoutInferArgs {
  Target target;
  size_t block_size;
  LayoutMap layout_map;
};

struct CanonializeArgs {};

class Operator {
 public:
  virtual Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const;
  virtual Stmt Canonialize(const CanonializeArgs& T, arith::Analyzer* analyzer) const;
  virtual LayoutMap InferLayout(const LayoutInferArgs& T, InferLevel level);
  virtual ~Operator() = default;
};

class RegionOp : public Operator {
 public:
  RegionOp(const Array<PrimExpr>& args, const Map<Var, Buffer>& vmap);
  static const Op& Get();

  const Buffer& GetBuffer() const { return buffer_; }
  const Array<Range>& GetRanges() const { return ranges_; }
  int GetAccessMask() const { return access_mask_; }

 private:
  Buffer buffer_;
  Array<Range> ranges_;
  int access_mask_;
};

Var GetVarFromAccessPtr(const PrimExpr& expr);

std::unique_ptr<Operator> ParseOperator(Call call, const Map<Var, Buffer>& vmap);
std::unique_ptr<Operator> ParseOperator(Stmt stmt, const Map<Var, Buffer>& vmap);

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_OP_OP_H_
