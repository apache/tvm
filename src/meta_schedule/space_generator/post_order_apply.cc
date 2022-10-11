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

/*! \brief Collecting all the blocks */
class BlockCollector : public tir::StmtVisitor {
 public:
  static Array<tir::BlockRV> Collect(const tir::Schedule& sch,
                                     const runtime::PackedFunc f_block_filter = nullptr) {  //
    return BlockCollector(sch, f_block_filter).Run();
  }

 private:
  /*! \brief Entry point */
  Array<tir::BlockRV> Run() {
    std::vector<tir::BlockRV> results;
    for (const auto& kv : sch_->mod()->functions) {
      const GlobalVar& gv = kv.first;         // `gv->name_hint` is the name of the function
      const BaseFunc& base_func = kv.second;  // this can be PrimFunc or relay::Function
      if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
        func_name_ = gv->name_hint;
        block_names_.clear();
        blocks_to_collect_.clear();
        VisitStmt(func->body);
        for (const String& name : blocks_to_collect_) {
          results.push_back(sch_->GetBlock(name, func_name_));
        }
      }
    }
    return results;
  }
  /*! \brief Constructor */
  explicit BlockCollector(const tir::Schedule& sch,
                          const runtime::PackedFunc f_block_filter = nullptr)
      : sch_(sch), f_block_filter_(f_block_filter) {}
  /*! \brief Override the Stmt visiting behaviour */
  void VisitStmt_(const tir::BlockNode* block) override {
    tir::StmtVisitor::VisitStmt_(block);
    CHECK(block_names_.count(block->name_hint) == 0)
        << "Duplicated block name " << block->name_hint << " in function " << func_name_
        << " not supported!";
    block_names_.insert(block->name_hint);

    // If filter function is provided, use it to selectively collect blocks.
    // Otherwise collect all blocks.
    Bool collect_block = Bool(true);
    if (f_block_filter_ != nullptr) {
      collect_block = f_block_filter_(GetRef<tir::Block>(block));
    }
    if (collect_block) {
      blocks_to_collect_.push_back(block->name_hint);
    }
  }

  /*! \brief The schedule to be collected */
  const tir::Schedule& sch_;
  /*! \brief An optional packed func that allows only certain blocks to be collected. */
  const runtime::PackedFunc f_block_filter_;
  /*! \brief The set of func name and block name pair */
  std::unordered_set<String> block_names_;
  /* \brief The list of blocks to collect in order */
  Array<String> blocks_to_collect_;
  /*! \brief Name of the current PrimFunc */
  String func_name_;
};

/*!
 * \brief Design Space Generator that generates design spaces by applying schedule rules to blocks
 *  in post-DFS order.
 * */
class PostOrderApplyNode : public SpaceGeneratorNode {
 public:
  /*!
   * \brief Optional block names to target. If not specified all blocks will have spaces generated.
   */
  runtime::PackedFunc f_block_filter_ = nullptr;
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;

  void VisitAttrs(tvm::AttrVisitor* v) {
    SpaceGeneratorNode::VisitAttrs(v);
    // `rand_state_` is not visited
    // `sch_rules_` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    SpaceGeneratorNode::InitializeWithTuneContext(context);
    this->rand_state_ = ForkSeed(&context->rand_state);
  }

  Array<tir::Schedule> GenerateDesignSpace(const IRModule& mod) final {
    using ScheduleAndUnvisitedBlocks = std::pair<tir::Schedule, Array<tir::BlockRV>>;
    CHECK(sch_rules.defined()) << "ValueError: `sch_rules` is not set in PostOrderApply";
    tir::Schedule sch = tir::Schedule::Traced(
        /*mod=*/mod,
        /*rand_state=*/ForkSeed(&this->rand_state_),
        /*debug_mode=*/0,
        /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);

    std::vector<ScheduleAndUnvisitedBlocks> stack;
    Array<tir::Schedule> result{sch};
    // Enumerate the schedule rules first because you can
    // always concat multiple schedule rules as one
    Array<tir::BlockRV> all_blocks = BlockCollector::Collect(sch, f_block_filter_);
    Array<Optional<ScheduleRule>> rules{NullOpt};
    rules.insert(rules.end(), sch_rules.value().begin(), sch_rules.value().end());
    for (Optional<ScheduleRule> sch_rule : rules) {
      if (sch_rule.defined()) {
        for (const tir::Schedule& sch : result) {
          stack.emplace_back(sch, all_blocks);
        }
      } else {
        for (const tir::Schedule& sch : result) {
          stack.emplace_back(sch, Array<tir::BlockRV>{all_blocks.rbegin(), all_blocks.rend()});
        }
      }
      result.clear();
      while (!stack.empty()) {
        // get the stack.top()
        auto [sch, blocks] = stack.back();
        stack.pop_back();
        // if all blocks are visited
        if (blocks.empty()) {
          result.push_back(sch);
          continue;
        }
        // otherwise, get the last block that is not visited
        tir::BlockRV block_rv = blocks.back();
        blocks.pop_back();
        if (!sch->HasBlock(block_rv)) {
          stack.emplace_back(sch, blocks);
          continue;
        }

        Optional<String> ann = tir::GetAnn<String>(sch->GetSRef(block_rv), "schedule_rule");
        const runtime::PackedFunc* custom_schedule_fn =
            ann.defined() ? runtime::Registry::Get(ann.value()) : nullptr;
        const bool has_schedule_rule = custom_schedule_fn != nullptr;

        if (ann.defined() && ann.value() != "None" && !has_schedule_rule) {
          LOG(WARNING) << "Custom schedule rule not found, ignoring schedule_rule annotation: "
                       << ann.value();
        }

        if ((has_schedule_rule && sch_rule.defined()) ||
            (!has_schedule_rule && !sch_rule.defined()) ||
            (ann.defined() && ann.value() == "None")) {
          stack.emplace_back(sch, blocks);
          continue;
        }

        Array<tir::Schedule> applied{nullptr};
        if (sch_rule.defined()) {
          applied = sch_rule.value()->Apply(sch, /*block=*/block_rv);
        } else {
          ICHECK(custom_schedule_fn)
              << "ValueError: Custom schedule rule not found: " << ann.value();
          applied = (*custom_schedule_fn)(sch, block_rv);
        }

        for (const tir::Schedule& sch : applied) {
          stack.emplace_back(sch, blocks);
        }
      }
    }
    return result;
  }

  SpaceGenerator Clone() const final {
    ObjectPtr<PostOrderApplyNode> n = make_object<PostOrderApplyNode>(*this);
    CloneRules(this, n.get());
    return SpaceGenerator(n);
  }
  static constexpr const char* _type_key = "meta_schedule.PostOrderApply";
  TVM_DECLARE_FINAL_OBJECT_INFO(PostOrderApplyNode, SpaceGeneratorNode);
};

SpaceGenerator SpaceGenerator::PostOrderApply(runtime::PackedFunc f_block_filter,
                                              Optional<Array<ScheduleRule>> sch_rules,
                                              Optional<Array<Postproc>> postprocs,
                                              Optional<Map<Mutator, FloatImm>> mutator_probs) {
  ObjectPtr<PostOrderApplyNode> n = make_object<PostOrderApplyNode>();
  n->sch_rules = std::move(sch_rules);
  n->postprocs = std::move(postprocs);
  n->mutator_probs = std::move(mutator_probs);
  n->f_block_filter_ = std::move(f_block_filter);
  return SpaceGenerator(n);
}

TVM_REGISTER_NODE_TYPE(PostOrderApplyNode);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorPostOrderApply")
    .set_body_typed(SpaceGenerator::PostOrderApply);

}  // namespace meta_schedule
}  // namespace tvm
