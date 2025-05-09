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
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include "tvm/arith/analyzer.h"

namespace tvm {
namespace tir {

int32_t DataType2Int(const tvm::DataType& dtype) {
  static_assert(sizeof(DLDataType) == sizeof(int32_t), "Incorrect size of DLDataType");
  union {
    DLDataType src;
    int32_t dst;
  } converter;
  converter.src.code = dtype.code();
  converter.src.bits = dtype.bits();
  converter.src.lanes = dtype.lanes();
  return converter.dst;
}

String Int2DataTypeStr(int32_t dtype) {
  union {
    DLDataType dst;
    int32_t src;
  } converter;
  converter.src = dtype;
  static std::string type_code_tab[] = {"int", "uint", "float", "handle", "bfloat"};
  std::ostringstream os;
  os << type_code_tab[converter.dst.code];
  os << static_cast<int>(converter.dst.bits);
  if (converter.dst.lanes != 1) {
    os << "x" << static_cast<int>(converter.dst.lanes);
  }
  return os.str();
}

struct TResult {
  TResult() = default;

  void Add(const tvm::DataType& dtype) { data_[DataType2Int(dtype)] += 1; }

  TResult operator+=(const TResult& rhs) {
    for (const auto& kv : rhs.data_) {
      data_[kv.first] += kv.second;
    }
    return *this;
  }

  TResult operator*=(int64_t rhs) {
    for (auto& kv : data_) {
      kv.second *= rhs;
    }
    return *this;
  }

  TResult MaxWith(const TResult& rhs) {
    for (const auto& kv : rhs.data_) {
      double& v = data_[kv.first];
      if (v < kv.second) {
        v = kv.second;
      }
    }
    return *this;
  }

  std::unordered_map<int32_t, double> data_;
};

class FlopEstimator : private ExprFunctor<TResult(const PrimExpr& n)>,
                      private StmtFunctor<TResult(const Stmt& n)> {
  arith::Analyzer ana;

 public:
  TResult VisitExpr(const PrimExpr& expr) override { return ExprFunctor::VisitExpr(expr); }
  TResult VisitStmt(const Stmt& stmt) override { return StmtFunctor::VisitStmt(stmt); }

#define TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(Node) \
  TResult VisitExpr_(const Node* op) final {     \
    TResult result = VisitExpr(op->a);           \
    result += VisitExpr(op->b);                  \
    result.Add(op->dtype);                       \
    return result;                               \
  }
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(AddNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(SubNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(MulNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(DivNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(ModNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(FloorDivNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(FloorModNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(MinNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(MaxNode);
#undef TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY
  TResult VisitExpr_(const EQNode* op) override { return TResult(); }
  TResult VisitExpr_(const NENode* op) override { return TResult(); }
  TResult VisitExpr_(const LTNode* op) override { return TResult(); }
  TResult VisitExpr_(const LENode* op) override { return TResult(); }
  TResult VisitExpr_(const GTNode* op) override { return TResult(); }
  TResult VisitExpr_(const GENode* op) override { return TResult(); }

  int64_t GetLoopExtent(const ForNode* node, const arith::Analyzer& ana) {
    int64_t bound = ana.const_int_bound(node->extent)->max_value;
    if (bound == arith::ConstIntBound::kPosInf) {
      return 1;  // Analyzer could not determine a valid bound, use 1 instead.
    } else {
      return bound;
    }
  }

  TResult VisitExpr_(const NotNode* op) override { return VisitExpr(op->a); }
  TResult VisitExpr_(const AndNode* op) final {
    TResult result = VisitExpr(op->a);
    result += VisitExpr(op->b);
    return result;
  }
  TResult VisitExpr_(const OrNode* op) final {
    TResult result = VisitExpr(op->a);
    result += VisitExpr(op->b);
    return result;
  }

  TResult VisitExpr_(const BufferLoadNode* op) override { return TResult(); }
  TResult VisitStmt_(const AttrStmtNode* op) override {
    TResult result = VisitStmt(op->body);
    result += VisitExpr(op->value);
    return result;
  }
  TResult VisitStmt_(const BufferStoreNode* store) override { return VisitExpr(store->value); }
  TResult VisitStmt_(const BlockRealizeNode* block) override {
    return VisitStmt(block->block->body);
  }
  TResult VisitStmt_(const BlockNode* block) override {
    TResult result;
    if (block->init.defined()) {
      result += VisitStmt(block->init.value());
    }
    result += VisitStmt(block->body);
    return result;
  }
  TResult VisitStmt_(const ForNode* loop) override {
    ana.Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    const auto int_imm = GetLoopExtent(loop, ana);
    TResult result = VisitStmt(loop->body);
    result *= int_imm;
    return result;
  }

  TResult VisitStmt_(const IfThenElseNode* branch) override {
    TResult cond = VisitExpr(branch->condition);
    if (branch->else_case) {
      cond += VisitStmt(branch->then_case).MaxWith(VisitStmt(branch->else_case.value()));
    } else {
      cond += VisitStmt(branch->then_case);
    }
    return cond;
  }

  TResult VisitStmt_(const LetStmtNode* let) override {
    TResult value = VisitExpr(let->value);
    value += VisitStmt(let->body);
    return value;
  }

  TResult VisitExpr_(const SelectNode* op) override {
    TResult cond = VisitExpr(op->condition);
    cond += VisitExpr(op->true_value).MaxWith(VisitExpr(op->false_value));
    return cond;
  }

  TResult VisitExpr_(const VarNode* op) override { return TResult(); }
  TResult VisitExpr_(const SizeVarNode* op) override { return TResult(); }
  TResult VisitExpr_(const IntImmNode* op) override { return TResult(); }
  TResult VisitExpr_(const FloatImmNode* op) override { return TResult(); }
  TResult VisitExpr_(const CastNode* op) override { return VisitExpr(op->value); }
  TResult VisitStmt_(const AllocateConstNode* op) override { return VisitStmt(op->body); }
  TResult VisitStmt_(const AllocateNode* op) override { return VisitStmt(op->body); }
  TResult VisitStmt_(const DeclBufferNode* op) override { return VisitStmt(op->body); }

  TResult VisitStmt_(const SeqStmtNode* seq) override {
    TResult result;
    for (const Stmt& stmt : seq->seq) {
      result += VisitStmt(stmt);
    }
    return result;
  }

  TResult VisitExpr_(const CallNode* op) override {
    TResult ret;
    for (const auto& x : op->args) {
      ret += VisitExpr(x);
    }
    return ret;
  }
};

double PostprocessResults(const TResult& result) {
  double cnt = 0.0;
  for (const auto& kv : result.data_) {
    cnt += kv.second;
  }
  return cnt;
}

double EstimateTIRFlops(const Stmt& stmt) {
  FlopEstimator counter;
  return PostprocessResults(counter.VisitStmt(stmt));
}

double EstimateTIRFlops(const IRModule& mod) {
  FlopEstimator counter;
  TResult result;
  double cached_result = 0;
  VisitPrimFuncs(mod, [&result, &counter, &cached_result](const PrimFuncNode* f) {
    if (auto cached = f->attrs.GetAttr<Integer>("estimated_flops")) {
      cached_result += cached.value()->value;
    } else {
      result += counter.VisitStmt(f->body);  //
    }
  });
  return PostprocessResults(result) + cached_result;
}

TVM_REGISTER_GLOBAL("tir.analysis.EstimateTIRFlops").set_body_typed([](ObjectRef obj) -> double {
  if (auto mod = obj.as<IRModule>()) {
    return EstimateTIRFlops(mod.value());
  } else if (auto stmt = obj.as<Stmt>()) {
    return EstimateTIRFlops(stmt.value());
  } else {
    LOG(FATAL) << "TypeError: Expect the input to be either IRModule or Stmt, but gets: "
               << obj->GetTypeKey();
    throw;
  }
});

}  // namespace tir
}  // namespace tvm
