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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/analysis.h>

#include "tvm/sym/analyzer.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

int32_t DataType2Int(DLDataType dtype) {
  static_assert(sizeof(DLDataType) == sizeof(int32_t), "Incorrect size of DLDataType");
  union {
    DLDataType src;
    int32_t dst;
  } converter;
  converter.src = dtype;
  return converter.dst;
}

ffi::String Int2DataTypeStr(int32_t dtype) {
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

  void Add(DLDataType dtype) { data_[DataType2Int(dtype)] += 1; }

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

class FlopEstimator : private tirx::ExprFunctor<TResult(const Expr& n)>,
                      private StmtFunctor<TResult(const Stmt& n)> {
  sym::Analyzer ana;

 public:
  using tirx::ExprFunctor<TResult(const Expr&)>::Dispatch;
  TResult Dispatch(const Stmt& stmt) override { return StmtFunctor::Dispatch(stmt); }

#define TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(Node)       \
  TResult Dispatch_(const Node* op) final {            \
    TResult result = Dispatch(op->a);                  \
    result += Dispatch(op->b);                         \
    result.Add(op->ty.as_or_throw<PrimType>()->dtype); \
    return result;                                     \
  }
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::AddNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::SubNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::MulNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::DivNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::ModNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::FloorDivNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::FloorModNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::MinNode);
  TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY(prim::MaxNode);
#undef TVM_TIR_ESTIMATE_FLOP_VISIT_BINARY
  TResult Dispatch_(const prim::EQNode* op) override { return TResult(); }
  TResult Dispatch_(const prim::NENode* op) override { return TResult(); }
  TResult Dispatch_(const prim::LTNode* op) override { return TResult(); }
  TResult Dispatch_(const prim::LENode* op) override { return TResult(); }
  TResult Dispatch_(const prim::GTNode* op) override { return TResult(); }
  TResult Dispatch_(const prim::GENode* op) override { return TResult(); }

  int64_t GetLoopExtent(const ForNode* node, const sym::Analyzer& ana) {
    int64_t bound = ana->const_int_bound(node->extent)->max_value;
    if (bound == sym::ConstIntBound::kPosInf) {
      return 1;  // Analyzer could not determine a valid bound, use 1 instead.
    } else {
      return bound;
    }
  }

  TResult Dispatch_(const prim::NotNode* op) override { return Dispatch(op->a); }
  TResult Dispatch_(const prim::AndNode* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }
  TResult Dispatch_(const prim::OrNode* op) final {
    TResult result = Dispatch(op->a);
    result += Dispatch(op->b);
    return result;
  }

  TResult Dispatch_(const TensorLoadNode* op) override { return TResult(); }
  TResult Dispatch_(const AttrStmtNode* op) override {
    TResult result = Dispatch(op->body);
    result += Dispatch(op->value);
    return result;
  }
  TResult Dispatch_(const BufferStoreNode* store) override { return Dispatch(store->value); }
  TResult Dispatch_(const SBlockRealizeNode* block) override {
    return Dispatch(block->block->body);
  }
  TResult Dispatch_(const SBlockNode* block) override {
    TResult result;
    if (block->init.has_value()) {
      result += Dispatch(block->init.value());
    }
    result += Dispatch(block->body);
    return result;
  }
  TResult Dispatch_(const ForNode* loop) override {
    ana->Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    const auto int_imm = GetLoopExtent(loop, ana);
    TResult result = Dispatch(loop->body);
    result *= int_imm;
    return result;
  }

  TResult Dispatch_(const IfThenElseNode* branch) override {
    TResult cond = Dispatch(branch->condition);
    if (branch->else_case) {
      cond += Dispatch(branch->then_case).MaxWith(Dispatch(branch->else_case.value()));
    } else {
      cond += Dispatch(branch->then_case);
    }
    return cond;
  }

  TResult Dispatch_(const WhileNode* op) override {
    // TODO(jikechao): Improve while loop FLOP estimation with loop bound analysis
    TResult result = Dispatch(op->condition);
    result += Dispatch(op->body);
    return result;
  }

  TResult Dispatch_(const BindNode* let) override {
    if (auto value = let->value.as<PrimExpr>()) return Dispatch(value.value());
    return TResult();
  }

  TResult Dispatch_(const prim::SelectNode* op) override {
    TResult cond = Dispatch(op->condition);
    cond += Dispatch(op->true_value).MaxWith(Dispatch(op->false_value));
    return cond;
  }

  TResult Dispatch_(const AssertStmtNode* op) override {
    TResult result = Dispatch(op->condition);
    return result;
  }

  TResult Dispatch_(const VarNode* op) override { return TResult(); }
  TResult Dispatch_(const prim::IntImmNode* op) override { return TResult(); }
  TResult Dispatch_(const prim::FloatImmNode* op) override { return TResult(); }
  TResult Dispatch_(const prim::StringImmNode* op) override { return TResult(); }
  TResult Dispatch_(const prim::CastNode* op) override { return Dispatch(op->value); }
  TResult Dispatch_(const AllocBufferNode* op) override { return TResult(); }
  TResult Dispatch_(const DeclBufferNode* op) override { return TResult(); }
  TResult Dispatch_(const EvaluateNode* op) override { return TResult(); }

  TResult Dispatch_(const SeqStmtNode* seq) override {
    TResult result;
    for (const Stmt& stmt : seq->seq) {
      result += Dispatch(stmt);
    }
    return result;
  }

  TResult Dispatch_(const CallNode* op) override {
    TResult ret;
    for (const Expr& arg : op->args) {
      ret += Dispatch(arg);
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
  return PostprocessResults(counter.Dispatch(stmt));
}

double EstimateTIRFlops(const IRModule& mod) {
  FlopEstimator counter;
  TResult result;
  double cached_result = 0;
  VisitPrimFuncs(mod, [&result, &counter, &cached_result](const PrimFuncNode* f) {
    if (auto cached = f->attrs.GetAttr<int64_t>("estimated_flops")) {
      cached_result += cached.value();
    } else {
      result += counter.Dispatch(f->body);  //
    }
  });
  return PostprocessResults(result) + cached_result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.EstimateTIRFlops", [](ffi::ObjectRef obj) -> double {
    if (auto mod = obj.as<IRModule>()) {
      return EstimateTIRFlops(mod.value());
    } else if (auto stmt = obj.as<Stmt>()) {
      return EstimateTIRFlops(stmt.value());
    } else {
      TVM_FFI_THROW(TypeError) << "Expect the input to be either IRModule or Stmt, but gets: "
                               << obj->GetTypeKey();
      throw;
    }
  });
}

}  // namespace s_tir
}  // namespace tvm
