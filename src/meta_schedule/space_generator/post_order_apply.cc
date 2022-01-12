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
namespace meta_schedule {

/*! \brief Collecting all the non-root blocks */
class BlockCollector : public tir::StmtVisitor {
 public:
  static Array<tir::BlockRV> Collect(const tir::Schedule& sch) {  //
    return BlockCollector(sch).Run();
  }

 private:
  /*! \brief Entry point */
  Array<tir::BlockRV> Run() {
    for (const auto& kv : sch_->mod()->functions) {
      const GlobalVar& gv = kv.first;         // `gv->name_hint` is the name of the function
      const BaseFunc& base_func = kv.second;  // this can be PrimFunc or relay::Function
      if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
        func_name_ = gv->name_hint;
        block_names_.clear();
        blocks_to_collect_.clear();
        VisitStmt(func->body);
        for (const String& block_name : blocks_to_collect_) {
          results_.push_back(sch_->GetBlock(block_name, func_name_));
        }
      }
    }
    return results_;
  }
  /*! \brief Constructor */
  explicit BlockCollector(const tir::Schedule& sch) : sch_(sch) {}
  /*! \brief Override the Stmt visiting behaviour */
  void VisitStmt_(const tir::BlockNode* block) override {
    tir::StmtVisitor::VisitStmt_(block);
    CHECK(block_names_.count(block->name_hint) == 0)
        << "Duplicated block name " << block->name_hint << " in function " << func_name_
        << " not supported!";
    block_names_.insert(block->name_hint);
    blocks_to_collect_.push_back(block->name_hint);
  }

  /*! \brief The schedule to be collected */
  const tir::Schedule& sch_;
  /*! \brief The set of func name and block name pair */
  std::unordered_set<String> block_names_;
  /* \brief The list of blocks to collect in order */
  Array<String> blocks_to_collect_;
  /*! \brief Function name & blocks of collection */
  Array<tir::BlockRV> results_;
  /*! \brief Name of the current PrimFunc */
  String func_name_;
};

/*!
 * \brief Design Space Generator that generates design spaces by applying schedule rules to blocks
 *  in post-DFS order.
 * */
class PostOrderApplyNode : public SpaceGeneratorNode {
 public:
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The schedule rules to be applied in order. */
  Array<ScheduleRule> sch_rules_{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `rand_state_` is not visited
    // `sch_rules_` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    this->rand_state_ = ForkSeed(&context->rand_state);
    CHECK(context->sch_rules.defined())
        << "ValueError: Schedules rules not given in PostOrderApply!";
    this->sch_rules_ = context->sch_rules;
  }

  Array<tir::Schedule> GenerateDesignSpace(const IRModule& mod_) final {
    using ScheduleAndUnvisitedBlocks = std::pair<tir::Schedule, Array<tir::BlockRV>>;
    tir::Schedule sch = tir::Schedule::Traced(                          //
        /*mod=*/mod_,                                                   //
        /*rand_state=*/ForkSeed(&this->rand_state_),                    //
        /*debug_mode=*/tir::kVerifySRefTree | tir::kVerifyCachedFlags,  //
        /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);

    std::vector<ScheduleAndUnvisitedBlocks> stack;
    Array<tir::Schedule> result{sch};
    // Enumerate the schedule rules first because you can
    // always concat multiple schedule rules as one
    Array<tir::BlockRV> all_blocks = BlockCollector::Collect(sch);
    for (ScheduleRule sch_rule : sch_rules_) {
      for (const tir::Schedule& sch : result) {
        stack.emplace_back(sch, all_blocks);
      }
      result.clear();

      while (!stack.empty()) {
        // get the stack.top()
        tir::Schedule sch;
        Array<tir::BlockRV> blocks;
        std::tie(sch, blocks) = stack.back();
        stack.pop_back();
        // if all blocks are visited
        if (blocks.empty()) {
          result.push_back(sch);
          continue;
        }
        // otherwise, get the last block that is not visited
        tir::BlockRV block_rv = blocks.back();
        blocks.pop_back();
        if (sch->HasBlock(block_rv)) {
          Array<tir::Schedule> applied = sch_rule->Apply(sch, /*block=*/block_rv);
          for (const tir::Schedule& sch : applied) {
            stack.emplace_back(sch, blocks);
          }
        } else {
          stack.emplace_back(sch, blocks);
        }
      }
    }
    return result;
  }
  static constexpr const char* _type_key = "meta_schedule.PostOrderApply";
  TVM_DECLARE_FINAL_OBJECT_INFO(PostOrderApplyNode, SpaceGeneratorNode);
};

SpaceGenerator SpaceGenerator::PostOrderApply() {
  ObjectPtr<PostOrderApplyNode> n = make_object<PostOrderApplyNode>();
  return SpaceGenerator(n);
}

TVM_REGISTER_NODE_TYPE(PostOrderApplyNode);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorPostOrderApply")
    .set_body_typed(SpaceGenerator::PostOrderApply);

}  // namespace meta_schedule
}  // namespace tvm
