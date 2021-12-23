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
#include <sstream>

#include "../utils.h"

namespace tvm {
namespace tir {

double CountFlop(const IRModule& mod) {
  struct TResult {
    using TTable = std::unordered_map<int32_t, double>;

    TResult() = default;

    explicit TResult(const tvm::DataType& dtype) { Add(dtype); }

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

    struct DType {
      uint8_t code : 8;
      uint8_t bits : 8;
      uint16_t lanes : 16;
    };
    static_assert(sizeof(DType) == 4, "Incorrect size of DType");

    static String Int2Str(int32_t dtype) {
      union {
        DType dst;
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

    static int32_t DataType2Int(const tvm::DataType& dtype) {
      union {
        DType src;
        int32_t dst;
      } converter;
      converter.src.code = dtype.code();
      converter.src.bits = dtype.bits();
      converter.src.lanes = dtype.lanes();
      return converter.dst;
    }

    TTable data_;
  };

  class FlopCounter : public ExprFunctor<TResult(const PrimExpr& n)>,
                      public StmtFunctor<TResult(const Stmt& n)> {
   public:
    ~FlopCounter() {}

    TResult VisitExpr(const PrimExpr& expr) override { return ExprFunctor::VisitExpr(expr); }
    TResult VisitStmt(const Stmt& stmt) override { return StmtFunctor::VisitStmt(stmt); }

    TResult VisitStmt_(const IfThenElseNode* branch) override {
      TResult cond = VisitExpr(branch->condition);
      cond += VisitStmt(branch->then_case).MaxWith(VisitStmt(branch->else_case));
      return cond;
    }

    TResult VisitStmt_(const BufferStoreNode* store) override {
      TResult result = VisitExpr(store->value);
      for (const PrimExpr& e : store->indices) {
        result += VisitExpr(e);
      }
      return result;
    }

    TResult VisitStmt_(const SeqStmtNode* seq) override {
      TResult result;
      for (const Stmt& stmt : seq->seq) {
        result += VisitStmt(stmt);
      }
      return result;
    }

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
      TResult result = VisitStmt(loop->body);
      const auto* int_imm = loop->extent.as<IntImmNode>();
      ICHECK(int_imm) << "TypeError: Expect the extent of a loop to be IntImm, but gets: "
                      << loop->extent->GetTypeKey();
      result *= int_imm->value;
      return result;
    }

#define TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(Node) \
  TResult VisitExpr_(const Node* op) final {        \
    TResult result(op->dtype);                      \
    result += VisitExpr(op->a);                     \
    result += VisitExpr(op->b);                     \
    return result;                                  \
  }
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(AddNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(SubNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(MulNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(DivNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(ModNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(FloorDivNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(FloorModNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(MinNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(MaxNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(EQNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(NENode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(LTNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(LENode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(GTNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(GENode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(AndNode);
    TVM_META_SCHEDULE_FLOP_COUNTER_BINARY(OrNode);
#undef TVM_META_SCHEDULE_FLOP_COUNTER_BINARY
    TResult VisitExpr_(const CastNode* op) override { return VisitExpr(op->value); }
    TResult VisitExpr_(const VarNode* op) override { return TResult(); }
    TResult VisitExpr_(const SizeVarNode* op) override { return TResult(); }
    TResult VisitExpr_(const BufferLoadNode* op) override { return TResult(); }
    TResult VisitExpr_(const IntImmNode* op) override { return TResult(); }
    TResult VisitExpr_(const FloatImmNode* op) override { return TResult(); }
    TResult VisitExpr_(const NotNode* op) override {
      TResult result(op->dtype);
      result += VisitExpr(op->a);
      return result;
    }
    TResult VisitExpr_(const SelectNode* op) override {
      TResult cond = VisitExpr(op->condition);
      cond += VisitExpr(op->true_value).MaxWith(VisitExpr(op->false_value));
      return cond;
    }
    TResult VisitExpr_(const CallNode* op) override {
      TResult ret;
      for (const auto& x : op->args) {
        ret += VisitExpr(x);
      }
      return ret;
    }
  };
  FlopCounter counter;
  TResult result;
  for (const auto& kv : mod->functions) {
    const BaseFunc& base_func = kv.second;
    if (const auto* prim_func = base_func.as<PrimFuncNode>()) {
      result += counter.VisitStmt(prim_func->body);
    }
  }
  double cnt = 0.0;
  int i32 = TResult::DataType2Int(tvm::DataType::Int(32));
  int i64 = TResult::DataType2Int(tvm::DataType::Int(64));
  int u1 = TResult::DataType2Int(tvm::DataType::UInt(1));
  for (const auto& kv : result.data_) {
    if (kv.first != i32 && kv.first != i64 && kv.first != u1) {
      cnt += kv.second;
    }
  }
  return cnt;
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

constexpr const double kMaxTime = 1e10;

std::string GetTaskName(const TuneContext& task, int task_id) {
  std::ostringstream os;
  os << '#' << task_id << ": " << task->task_name;
  return os.str();
}

double GetRunMs(const Array<FloatImm>& run_secs) {
  double total = 0.0;
  for (const FloatImm& i : run_secs) {
    total += i->value;
  }
  return total * 1e3 / run_secs.size();
}

struct TaskInfo {
  std::string name;
  double flop = 0.0;
  int trials = 0;
  int best_round = -1;
  double best_ms = kMaxTime;
  double best_gflops = 0.0;
  int error_count = 0;

  explicit TaskInfo(const String& name) : name(name) {}

  void Update(double run_ms) {
    ++trials;
    if (run_ms < best_ms) {
      best_ms = run_ms;
      best_round = trials;
      best_gflops = flop / run_ms / 1e6;
    }
    LOG(INFO) << "[" << name << "] Trial #" << trials   //
              << std::fixed << std::setprecision(4)     //
              << ": GFLOPs: " << (flop / run_ms / 1e6)  //
              << ". Time: " << run_ms << " ms"          //
              << ". Best GFLOPs: " << best_gflops;
  }

  void UpdateError(std::string err, const MeasureCandidate& candidate) {
    static const auto* f_proc = runtime::Registry::Get("meta_schedule._process_error_message");
    ICHECK(f_proc != nullptr);
    err = (*f_proc)(err).operator std::string();
    ++error_count;
    ++trials;
    LOG(INFO) << "[" << name << "] Trial #" << trials  //
              << std::fixed << std::setprecision(4)    //
              << ": Error in building: " << err << "\n"
              << tir::AsTVMScript(candidate->sch->mod()) << "\n"
              << Concat(candidate->sch->trace().value()->AsPython(false), "\n");
  }
};

class EchoStatisticsNode : public MeasureCallbackNode {
 public:
  void Apply(const TaskScheduler& task_scheduler, int task_id,
             const Array<MeasureCandidate>& measure_candidates,
             const Array<BuilderResult>& builder_results,
             const Array<RunnerResult>& runner_results) final {
    if (this->task_info.empty()) {
      SetupTaskInfo(task_scheduler->tasks);
    }
    ICHECK_EQ(measure_candidates.size(), builder_results.size());
    ICHECK_EQ(measure_candidates.size(), runner_results.size());
    int n = measure_candidates.size();
    TuneContext task = task_scheduler->tasks[task_id];
    TaskInfo& info = this->task_info[task_id];
    std::string task_name = GetTaskName(task, task_id);
    for (int i = 0; i < n; ++i) {
      MeasureCandidate candidate = measure_candidates[i];
      BuilderResult builder_result = builder_results[i];
      RunnerResult runner_result = runner_results[i];
      if (Optional<String> err = builder_result->error_msg) {
        info.UpdateError(err.value(), candidate);
      } else if (Optional<String> err = runner_result->error_msg) {
        info.UpdateError(err.value(), candidate);
      } else {
        ICHECK(runner_result->run_secs.defined());
        info.Update(GetRunMs(runner_result->run_secs.value()));
      }
    }
  }

  void SetupTaskInfo(const Array<TuneContext>& tasks) {
    task_info.reserve(tasks.size());
    int task_id = 0;
    for (const TuneContext& task : tasks) {
      task_info.push_back(TaskInfo(GetTaskName(task, task_id)));
      TaskInfo& info = task_info.back();
      info.flop = tir::CountFlop(task->mod.value());
      ++task_id;
    }
  }

  std::vector<TaskInfo> task_info;

  static constexpr const char* _type_key = "meta_schedule.EchoStatistics";
  TVM_DECLARE_FINAL_OBJECT_INFO(EchoStatisticsNode, MeasureCallbackNode);
};

MeasureCallback MeasureCallback::EchoStatistics() {
  ObjectPtr<EchoStatisticsNode> n = make_object<EchoStatisticsNode>();
  return MeasureCallback(n);
}

TVM_REGISTER_NODE_TYPE(EchoStatisticsNode);
TVM_REGISTER_GLOBAL("meta_schedule.MeasureCallbackEchoStatistics")
    .set_body_typed(MeasureCallback::EchoStatistics);

}  // namespace meta_schedule
}  // namespace tvm
