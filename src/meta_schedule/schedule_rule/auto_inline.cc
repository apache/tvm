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

/*! \brief The type of inline to be performed on a specific block */
enum class InlineType : int32_t {
  /*! \brief No inline opportunity */
  kNoInline = 0,
  /*! \brief Inline the block into its consumer */
  kInlineIntoConsumer = 1,
  /*! \brief Inline the block into its producer */
  kInlineIntoProducer = 2,
};

bool IsInSpatialPrimFunc(const tir::Schedule& sch, const tir::StmtSRef& block_sref) {
  using namespace tvm::tir;
  const StmtSRefNode* sref = block_sref.get();
  for (; sref->parent != nullptr; sref = sref->parent) {
  }
  ICHECK(sref->stmt != nullptr && sref->stmt->IsInstance<BlockNode>());
  return IsSpatialPrimFunc(GetRef<PrimFunc>(GetRootPrimFunc(sch->mod(), sref->stmt, nullptr)));
}

/*! \brief The rule that inlines spatial blocks if it satisfies some conditions. */
class AutoInlineNode : public ScheduleRuleNode {
 public:
  /*! \brief Checks if the specific block should be inlined */
  inline InlineType CheckInline(const tir::Schedule& sch, const tir::BlockRV& block_rv);

  // Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    InlineType inline_type = CheckInline(sch, block_rv);
    if (inline_type == InlineType::kInlineIntoConsumer) {
      sch->ComputeInline(block_rv);
    } else if (inline_type == InlineType::kInlineIntoProducer) {
      sch->ReverseComputeInline(block_rv);
    }
    return {sch};
  }

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<AutoInlineNode> n = make_object<AutoInlineNode>(*this);
    return ScheduleRule(n);
  }

 public:
  /*! \brief If allows to inline a block into its producer */
  bool into_producer;
  /*! \brief If allows to inline a block into its consumer */
  bool into_consumer;
  /*! \brief Always inline constant tensors */
  bool inline_const_tensor;
  /*! \brief Always disallow if-then-else-like constructs */
  bool disallow_if_then_else;
  /*! \brief Always require the read-to-write mapping to be injective to do auto inline */
  bool require_injective;
  /*! \brief Always require the read-to-write mapping to be ordered to do auto inline */
  bool require_ordered;
  /*! \brief The operators that are disallowed in auto inline */
  Array<Op> disallow_op;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("into_producer", &into_producer);
    v->Visit("into_consumer", &into_consumer);
    v->Visit("inline_const_tensor", &inline_const_tensor);
    v->Visit("disallow_if_then_else", &disallow_if_then_else);
    v->Visit("require_injective", &require_injective);
    v->Visit("require_ordered", &require_ordered);
    v->Visit("disallow_op", &disallow_op);
  }

  static constexpr const char* _type_key = "meta_schedule.AutoInline";
  TVM_DECLARE_FINAL_OBJECT_INFO(AutoInlineNode, ScheduleRuleNode);
};

inline InlineType AutoInlineNode::CheckInline(const tir::Schedule& sch,
                                              const tir::BlockRV& block_rv) {
  using namespace tvm::tir;
  StmtSRef block_sref = sch->GetSRef(block_rv);
  bool is_pure_sptial = IsInSpatialPrimFunc(sch, block_sref);
  ScheduleState state = sch->state();
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  BlockRealize realize = GetBlockRealize(state, block_sref);
  // Cond 1. The block has only one write buffer
  if (block->writes.size() != 1) {
    return InlineType::kNoInline;
  }
  // Cond 2. For a block that generates a constant tensor, ignore all other conditions
  if (inline_const_tensor && block->reads.empty()) {
    Array<tir::StmtSRef> consumer_srefs = GetConsumers(state, block_sref);
    if (!consumer_srefs.empty() && CanComputeInline(state, block_sref)) {
      return InlineType::kInlineIntoConsumer;
    }
  }
  // Cond 3. The block doesn't contain any disallowed operators
  if (!is_pure_sptial && !disallow_op.empty() && HasOp(realize, disallow_op)) {
    return InlineType::kNoInline;
  }
  // Cond 4. The block doesn't have any if-then-else-like constructs
  if (!is_pure_sptial && disallow_if_then_else && HasIfThenElse(realize)) {
    return InlineType::kNoInline;
  }
  // Cond 5. The mapping from read indices to write indices are injective and ordered
  if (!is_pure_sptial && (require_injective || require_ordered)) {
    const BufferRegion& write_region = block->writes[0];
    for (const BufferRegion& read_region : block->reads) {
      bool injective, ordered;
      auto _ = std::ignore;
      std::tie(/*exists=*/_, /*surjective=*/_, injective, ordered, /*no_const_read=*/_,
               /*no_shift_read=*/_) = AnalyzeReadWritePattern(read_region, write_region);
      if (require_injective && injective == false) {
        return InlineType::kNoInline;
      }
      if (require_ordered && ordered == false) {
        return InlineType::kNoInline;
      }
    }
  }
  // Last cond: Check inline into the consumers or the spatial producer
  tir::StmtSRef scope_block = tir::GetScopeRoot(sch->state(), block_sref,
                                                /*require_stage_pipeline=*/false);
  if (into_consumer) {
    Array<tir::StmtSRef> consumer_srefs = GetConsumers(state, block_sref);
    if (!consumer_srefs.empty() && CanComputeInline(state, block_sref)) {
      return InlineType::kInlineIntoConsumer;
    }
  }
  if (into_producer) {
    Array<tir::StmtSRef> producer_srefs = GetProducers(state, block_sref);
    if (producer_srefs.size() == 1 &&
        tir::IsCompleteBlock(sch->state(), producer_srefs[0], scope_block) &&
        CanReverseComputeInline(state, block_sref) &&
        !GetAnn<String>(producer_srefs[0], tir::attr::meta_schedule_auto_tensorize).defined()) {
      return InlineType::kInlineIntoProducer;
    }
  }
  return InlineType::kNoInline;
}

ScheduleRule ScheduleRule::AutoInline(bool into_producer,          //
                                      bool into_consumer,          //
                                      bool inline_const_tensor,    //
                                      bool disallow_if_then_else,  //
                                      bool require_injective,      //
                                      bool require_ordered,        //
                                      Optional<Array<String>> disallow_op) {
  ObjectPtr<AutoInlineNode> n = make_object<AutoInlineNode>();
  n->into_producer = into_producer;
  n->into_consumer = into_consumer;
  n->inline_const_tensor = inline_const_tensor;
  n->disallow_if_then_else = disallow_if_then_else;
  n->require_injective = require_injective;
  n->require_ordered = require_ordered;
  n->disallow_op.clear();
  if (disallow_op.defined()) {
    Array<String> op_names = disallow_op.value();
    n->disallow_op.reserve(op_names.size());
    for (const String& op_name : op_names) {
      n->disallow_op.push_back(Op::Get(op_name));
    }
  }
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(AutoInlineNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleAutoInline")
    .set_body_typed(ScheduleRule::AutoInline);

/*! \brief Inline blocks that produce a constant scalar. */
class InlineConstantScalarsNode : public ScheduleRuleNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    // Look for a block of the form
    // block compile_engine_const(iter_var(vi, range(min=0, ext=1))) {
    //   reads([])
    //   writes([compile_engine_const[]])
    //   compile_engine_const[] = 59
    // }
    auto block = sch->Get(block_rv);
    if (block->reads.size() == 0 && block->writes.size() == 1 &&
        block->writes[0]->buffer->shape.size() == 0) {
      sch->ComputeInline(block_rv);
    }
    return {sch};
  }

  ScheduleRule Clone() const final {
    ObjectPtr<InlineConstantScalarsNode> n = make_object<InlineConstantScalarsNode>(*this);
    return ScheduleRule(n);
  }

  static constexpr const char* _type_key = "meta_schedule.InlineConstantScalars";
  TVM_DECLARE_FINAL_OBJECT_INFO(InlineConstantScalarsNode, ScheduleRuleNode);
};

ScheduleRule ScheduleRule::InlineConstantScalars() {
  ObjectPtr<InlineConstantScalarsNode> n = make_object<InlineConstantScalarsNode>();
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(InlineConstantScalarsNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleInlineConstantScalars")
    .set_body_typed(ScheduleRule::InlineConstantScalars);
}  // namespace meta_schedule
}  // namespace tvm
