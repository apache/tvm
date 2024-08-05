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
#include "../../transforms/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace tir {

class ArgumentInjector : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add stages of writting precomputed index
   * \param scope_sref The parent scope of this mutation
   * \param info The index information
   * \return The new AST rooting at the original parent scope
   */
  ArgumentInjector(int idx, PrimExpr argument) : idx_(idx), argument_(argument) {}

 private:
  PrimExpr VisitExpr_(const CallNode* op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    auto new_args = call->args;
    if (call->op.same_as(builtin::call_extern())) {
      int arg_size = call->args.size();
      ICHECK(arg_size > idx_) << "Index out of bounds";
      if (idx_ > 0){
        new_args.insert(new_args.begin() + idx_, argument_);
      } else {
        ICHECK(-idx_ <= arg_size) << "Index out of bounds";
        new_args.insert(new_args.begin() + arg_size + idx_ + 1, argument_);
      }
      injected_ = true;
    }
    return Call(call->dtype, call->op, new_args, call->span);
  }

  int idx_;
  PrimExpr argument_;
  bool injected_{false};
};

/******** Implementation ********/

void UnsafeInjectCallArgument(ScheduleState self, const StmtSRef& block_sref, int idx,
const PrimExpr& argument) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);

  /* Step 0: Collect new buffer access regions. */
  /* Step 1: Replace old block with the new block */
  auto n = make_object<BlockNode>(*block);
  Block new_block = Block(n);
  ArgumentInjector injector(idx, argument);
  Stmt stmt = injector(new_block);
  new_block = Downcast<Block>(stmt);
  Map<Block, Block> blk_map;
  blk_map.Set(GetRef<Block>(block), new_block);
  self->Replace(block_sref, new_block, blk_map);
}

struct UnsafeInjectCallArgumentTraits
    : public UnpackedInstTraits<UnsafeInjectCallArgumentTraits> {
  static constexpr const char* kName = "UnsafeInjectCallArgument";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 3;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer idx, PrimExpr argument) {
    sch->UnsafeInjectCallArgument(block, idx.IntValue(), argument);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, Integer idx,
                                 PrimExpr argument) {
    PythonAPICall py("unsafe_rewrite_buffer_access");
    py.Input("block", block_rv);
    std::ostringstream os;
    os << "(\"" << idx->value << ")";
    py.Input("buffer", os.str());
    py.Input("argument", argument);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(UnsafeInjectCallArgumentTraits);
}  // namespace tir
}  // namespace tvm
