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
#include "../utils.h"

namespace tvm {
namespace tir {

class WrongBlockIterTypeError : public ScheduleError {
 public:
  explicit WrongBlockIterTypeError(IRModule mod, ForKind for_kind, Var loop_var, Block block)
      : mod_(std::move(mod)), loop_var_(std::move(loop_var)), block_(std::move(block)) {
    op_str_ = for_kind == ForKind::kParallel
                  ? "parallel"
                  : (for_kind == ForKind::kVectorized ? "vectorize" : "bind");
  }
  String FastErrorString() const final {
    std::ostringstream os;
    os << "ScheduleError: The \"" << op_str_
       << "\" cannot be fulfilled with regard to some of its underlying block";
    return os.str();
  }
  String DetailRenderTemplate() const final {
    std::ostringstream os;
    if (op_str_ != "bind") {
      os << "The \"" << op_str_
         << "\" cannot be fulfilled with regard to block {0} because some block iter whose block "
            "binding contains the loop var is not a data parallel block iter";
    } else {
      os << "The \"bind\" cannot be fulfilled with regard to block {0}. This is because some of its"
            " block iter whose block binding contains "
         << loop_var_
         << " does not meet any of the conditions:\n1) the block iter is data parallel;\n2) the "
            "block iter is a reduction block iter, and the thread axis to be bound is "
            "\"threadIdx.x/y/z\"";
    }
    return os.str();
  }
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod_;
  std::string op_str_;
  Var loop_var_;
  Block block_;
};

/*!
 * \brief Check if a loop can be parallelized/vectorized/bound with regard to a specific block
 * \details There are two conditions:
 * 1) The block is required to have affine bindings, and
 * 2) For each block iter whose binding contains the input loop variable, either
 *   - the block iter is data parallel, or
 *   - the block iter is a reduction block iter, and the input `thread_tag` starts with "threadIdx"
 *   in case of cross-thread reduction.
 * \param self The schedule state
 * \param for_kind The desired ForKind (only `kParallel`, `kVectorized` and `kThreadBinding` are
 * allowed)
 * \param loop_var The loop variable of the loop to be checked
 * \param block_realize The block-realize of the block to be checked
 * \param thread_scope The thread scope of the thread axis to be bound, which is an invalid value if
 * the operation is not "bind"
 * \throws ScheduleError If the input loop cannot be parallelized/vectorized/bound with regard to
 * the input block
 */
void CheckLoopParallelizableInBlock(const ScheduleState& self, ForKind for_kind,
                                    const Var& loop_var, const BlockRealize& block_realize,
                                    runtime::ThreadScope thread_scope) {
  const Block& block = block_realize->block;

  // Cond 1. The block is required to have affine bindings.
  // TODO(@automation): fix the check
  // CheckAffineBinding(self, block);

  // Cond 2. For each block iter whose binding contains `loop_var`, only two cases are allowed.
  ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());
  int n_iters = static_cast<int>(block->iter_vars.size());
  for (int i = 0; i < n_iters; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = block_realize->iter_values[i];

    if (!UsesVar(binding, [v = loop_var.get()](const VarNode* var) { return var == v; })) {
      continue;
    }
    // Only two cases are allowed:
    // - The block iter is data parallel, or
    // - The block iter is a reduction block iter, and the `thread_scope` is "threadIdx.x/y/z"
    // in case of cross-thread reduction.
    IterVarType iter_type = iter_var->iter_type;
    if (!(iter_type == kDataPar ||
          (iter_type == kCommReduce && thread_scope.rank == 1 && thread_scope.dim_index != -1))) {
      throw WrongBlockIterTypeError(self->mod, for_kind, loop_var, block);
    }
  }
}

/*!
 * \brief For each block (recursive) under the given loop, check whether the input loop can be
 * parallelized/vectorized/bound with regard to the block
 * \param self The schedule state
 * \param loop The loop to be parallelized/vectorized/bound
 * \param for_kind The desired ForKind (only `kParallel`, `kVectorized` and `kThreadBinding` are
 * allowed)
 * \param thread_scope The thread scope of the thread axis to be bound, which is an invalid value if
 * the operation is not "bind"
 */
void CheckParallelizability(const ScheduleState& self, const For& loop, ForKind for_kind,
                            runtime::ThreadScope thread_scope) {
  PreOrderVisit(loop, [&](const ObjectRef& node) {
    if (const auto* realize = node.as<BlockRealizeNode>()) {
      // If this block doesn't have corresponding StmtSRef in the schedule state, it must be a block
      // inside `tir.init()`. We don't check the condition for such blocks.
      if (!self->stmt2ref.count(realize->block.get())) {
        return false;
      }
      CheckLoopParallelizableInBlock(self, for_kind, loop->loop_var, GetRef<BlockRealize>(realize),
                                     thread_scope);
    }
    return true;
  });
}

/*!
 * \brief The implementation of parallelizing/vectorizing/binding a given loop
 * \param self The schedule state
 * \param loop_sref The sref of the loop to be parallelized/vectorized/bound
 * \param for_kind The type of the operation (only `kParallel`, `kVectorized` and `kThreadBinding`
 * are allowed)
 * \param thread_axis The thread axis that the input loop is bound to, which is defined only when
 * `for_kind` is `kThreadBinding`
 */
void ParallelizeComputation(const ScheduleState& self, const StmtSRef& loop_sref, ForKind for_kind,
                            Optional<IterVar> thread_axis) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);

  /*
   * Check:
   * - 1. the subtree rooted from the input loop in sref tree has compact data flow
   * - 2. all the blocks under the given loop have affine block bindings
   * - 3. the input loop can be only bound to data parallel block iters, or the loop can be bound to
   * reduction block iter if `thread` is `threadIdx.x/y/z` in case of cross-thread reduction
   * When the above conditions are all satisfied, this input loop can be
   * parallelized/vectorized/bound.
   */
  // Step 1. Check whether the subtree rooted from the `loop` in sref tree has compact data flow.
  CheckSubtreeCompactDataflow(self, loop_sref);

  // Step 2. Check whether the loop can be parallelized/vectorized/bound with regard to each
  // underlying block.
  CheckParallelizability(self, GetRef<For>(loop), for_kind,
                         thread_axis.defined()
                             ? runtime::ThreadScope::Create(thread_axis.value()->thread_tag)
                             : runtime::ThreadScope{-1, -1});

  // Step 3. Loop update and IR replacement
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->kind = for_kind;
  new_loop->thread_binding = std::move(thread_axis);
  self->Replace(loop_sref, For(new_loop), {});
}

void Parallel(ScheduleState self, const StmtSRef& loop_sref) {
  ParallelizeComputation(self, loop_sref, ForKind::kParallel, NullOpt);
}

void Vectorize(ScheduleState self, const StmtSRef& loop_sref) {
  ParallelizeComputation(self, loop_sref, ForKind::kVectorized, NullOpt);
}

void Bind(ScheduleState self, const StmtSRef& loop_sref, const IterVar& thread_axis) {
  ParallelizeComputation(self, loop_sref, ForKind::kThreadBinding, thread_axis);
}

void Unroll(ScheduleState self, const StmtSRef& loop_sref) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->kind = ForKind::kUnrolled;
  new_loop->thread_binding = NullOpt;
  self->Replace(loop_sref, For(new_loop), {});
}

/******** InstructionKind Registration ********/

struct ParallelTraits : public UnpackedInstTraits<ParallelTraits> {
  static constexpr const char* kName = "Parallel";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) {
    return sch->Parallel(loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("parallel");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct VectorizeTraits : public UnpackedInstTraits<VectorizeTraits> {
  static constexpr const char* kName = "Vectorize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) {
    return sch->Vectorize(loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("vectorize");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct BindTraits : public UnpackedInstTraits<BindTraits> {
  static constexpr const char* kName = "Bind";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, String thread) {
    return sch->Bind(loop_rv, thread);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, String thread) {
    PythonAPICall py("bind");
    py.Input("loop", loop_rv);
    py.Input("thread_axis", thread);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct UnrollTraits : public UnpackedInstTraits<UnrollTraits> {
  static constexpr const char* kName = "Unroll";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) { return sch->Unroll(loop_rv); }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("unroll");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ParallelTraits);
TVM_REGISTER_INST_KIND_TRAITS(VectorizeTraits);
TVM_REGISTER_INST_KIND_TRAITS(BindTraits);
TVM_REGISTER_INST_KIND_TRAITS(UnrollTraits);

}  // namespace tir
}  // namespace tvm
