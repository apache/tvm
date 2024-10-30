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
 * \file schedule_dataflow_rewrite.cc
 */
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

#include "../../tir/transforms/ir_utils.h"
#include "message_passing.h"
#include "operation_inline.h"

namespace tvm {
namespace te {
// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Object* n = v.get();
  for (size_t i = 0; i < array_node->size(); ++i) {
    if (array_node->at(i).get() == n) return i;
  }
  return array_node->size();
}

// The replacer of cache.
class VarReplacer : public tir::StmtExprMutator {
 public:
  explicit VarReplacer(const std::unordered_map<const VarNode*, PrimExpr>& vsub) : vsub_(vsub) {}
  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = vsub_.find(op);
    if (it != vsub_.end()) return it->second;
    return GetRef<PrimExpr>(op);
  }

  tir::CommReducer MutateCommReducer(tir::CommReducer combiner) {
    // Replace free variables in combiner
    auto new_identity = tir::UpdateArray(combiner->identity_element,
                                         [this](const PrimExpr& e) { return this->VisitExpr(e); });
    auto new_result = tir::UpdateArray(combiner->result,
                                       [this](const PrimExpr& e) { return this->VisitExpr(e); });

    if (combiner->identity_element.same_as(new_identity) &&
        combiner->identity_element.same_as(new_result)) {
      return combiner;
    } else {
      return tir::CommReducer(combiner->lhs, combiner->rhs, new_result, new_identity);
    }
  }

  PrimExpr VisitExpr_(const tir::ReduceNode* op) final {
    PrimExpr new_e = StmtExprMutator::VisitExpr_(op);
    const tir::ReduceNode* new_reduce = new_e.as<tir::ReduceNode>();
    tir::CommReducer new_combiner = MutateCommReducer(op->combiner);
    if (op->combiner.same_as(new_combiner)) {
      return new_e;
    } else {
      return tir::Reduce(new_combiner, new_reduce->source, new_reduce->axis, new_reduce->condition,
                         new_reduce->value_index, new_reduce->init);
    }
  }

 private:
  const std::unordered_map<const VarNode*, PrimExpr>& vsub_;
};

PrimExpr InjectPredicate(const Array<PrimExpr>& predicates, PrimExpr body) {
  using tir::ReduceNode;
  using tir::SelectNode;
  if (predicates.size() == 0) return body;
  const ReduceNode* reduce = body.as<ReduceNode>();

  if (reduce) {
    auto n = make_object<ReduceNode>(*reduce);
    n->condition = foldl([](PrimExpr a, PrimExpr b, Span span) { return logical_and(a, b, span); },
                         n->condition, predicates);
    return PrimExpr(n);
  }
  return Select(foldl([](PrimExpr a, PrimExpr b, Span span) { return logical_and(a, b, span); },
                      const_true(1), predicates),
                body, make_zero(body.dtype()));
}

// Replace data flow appears in all stages given the tensor change.
// Also update vmap if subsequent dataflow need to be replaced.
// Need to keep an update to the date transitive closure property on the vmap by a reverse map.
void ReplaceDataFlow(const Array<Stage>& stages, std::unordered_map<Tensor, Tensor>* vmap,
                     std::unordered_map<Tensor, Tensor>* rvmap) {
  for (Stage s : stages) {
    Operation op = s->op->ReplaceInputs(s->op, *vmap);
    if (!op.same_as(s->op)) {
      for (int i = 0; i < op->num_outputs(); ++i) {
        auto it = rvmap->find(s->op.output(i));
        if (it != rvmap->end()) {
          (*vmap)[it->second] = op.output(i);
        } else {
          (*vmap)[s->op.output(i)] = op.output(i);
          (*rvmap)[op.output(i)] = s->op.output(i);
        }
      }
      s->op = op;
    }
  }
}

inline bool ReduceEqual(const tir::ReduceNode* a, const tir::ReduceNode* b) {
  StructuralEqual struct_equal;
  return struct_equal(a->combiner, b->combiner) && struct_equal(a->source, b->source) &&
         struct_equal(a->axis, b->axis) && struct_equal(a->condition, b->condition) &&
         struct_equal(a->init, b->init);
}

Tensor Schedule::cache_read(const Tensor& tensor, const std::string& scope,
                            const Array<Operation>& readers) {
  (*this)->InvalidateCache();
  // create identity mapping.
  std::ostringstream os;
  os << tensor->op->name;
  if (tensor->op->num_outputs() != 1) {
    os << ".v" << tensor->value_index;
  }

  // when a schedule has multiple cache_read on the same tensor,
  // we make sure their op names are unique. e.g., w.shared, w_d.shared, w_d_d.shared
  for (auto pair : (*this)->stage_map) {
    auto stage = pair.second;
    if (stage->op->name == os.str() + "." + scope) {
      os << ".d";
    }
  }
  os << "." << scope;

  std::unordered_map<Tensor, Tensor> vsub;
  Stage s = operator[](tensor->op);
  Tensor sugar_tensor = s->op.output(tensor->value_index);
  Tensor cache = compute(
      sugar_tensor->shape,
      [&sugar_tensor](const Array<Var>& i) {
        return sugar_tensor(Array<PrimExpr>(i.begin(), i.end()));
      },
      os.str());
  vsub[sugar_tensor] = cache;

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  for (Operation op : readers) {
    Stage s = operator[](op);
    Operation repl_op = s->op->ReplaceInputs(s->op, vsub);
    ICHECK(!repl_op.same_as(s->op)) << "Cannot find " << tensor << " in the inputs of " << s->op;
    vmap[s->op.output(0)] = repl_op.output(0);
    rvmap[repl_op.output(0)] = s->op.output(0);
    s->op = repl_op;
  }
  ReplaceDataFlow((*this)->stages, &vmap, &rvmap);
  Array<Stage>& stages = (*this)->stages;
  Stage op_stage = operator[](tensor->op);
  size_t pos = FindNodeRef(stages.GetArrayNode(), op_stage);
  Stage cache_stage = Stage(cache->op, this->operator->());
  ICHECK_LT(pos, stages.size());
  stages.insert(stages.begin() + pos + 1, cache_stage);
  // in order to obtain correct copy on schedule_record,
  // make sure "set_scope" primitive is applied after stage being added
  cache_stage.set_scope(scope);
  (*this)->stage_map.Set(cache->op, cache_stage);
  // Update group
  cache_stage->group = op_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache;
}

template <typename OpType>
void PrepareAxisMapping(Stage orig_stage, OpType* op, std::unordered_set<IterVar>* p_red_axis,
                        Array<IterVar>* p_new_axis, std::unordered_map<IterVar, Range>* p_dom_map,
                        std::unordered_map<const VarNode*, PrimExpr>* p_vsub,
                        std::unordered_map<const VarNode*, PrimExpr>* p_vsub2newvar,
                        std::vector<PrimExpr>* p_predicates) {
  auto& red_axis = *p_red_axis;
  auto& new_axis = *p_new_axis;
  auto& dom_map = *p_dom_map;
  auto& vsub = *p_vsub;
  auto& vsub2newvar = *p_vsub2newvar;
  auto& predicates = *p_predicates;
  arith::Analyzer analyzer;

  for (IterVar iv : op->reduce_axis) {
    red_axis.insert(iv);
  }
  for (IterVar iv : op->axis) {
    dom_map[iv] = iv->dom;
    analyzer.Bind(iv->var, iv->dom);
  }
  te::PassDownDomain(orig_stage, &dom_map, &analyzer, true);
  {
    // The source->cache
    std::unordered_map<IterVar, PrimExpr> value_map;
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      ICHECK_EQ(iv->iter_type, kDataPar) << "Can only relayout with in data parallel dimensions";
      Range dom = dom_map.at(iv);
      IterVar new_iv = IterVar(dom, iv->var.copy_with_suffix(".c"), iv->iter_type);
      new_axis.push_back(new_iv);
      if (is_one(dom->min)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
        vsub2newvar[iv->var.get()] = new_iv->var;
      }
    }
    // skip reduction iteration.
    std::unordered_set<IterVar> skip_bound_check;
    for (IterVar iv : op->reduce_axis) {
      skip_bound_check.insert(iv);
    }
    PassUpIndex(orig_stage, dom_map, &value_map, true);
    predicates = MakeBoundCheck(orig_stage, dom_map, value_map, true, skip_bound_check);
    // The root axis
    for (IterVar iv : op->axis) {
      if (value_map.count(iv)) {
        vsub[iv->var.get()] = value_map.at(iv);
      }  // to handle tensor axis
    }
  }
}

Array<Tensor> ReplaceOriginalOp(Schedule sch, Stage orig_stage, const std::string& scope,
                                Operation cache_op, Operation orig_new_op, size_t tensor_size) {
  Array<Tensor> cache_tensor_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_tensor_list.push_back(cache_tensor);
  }
  // The replace of the dataflow
  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  vmap[orig_stage->op.output(0)] = orig_new_op.output(0);
  rvmap[orig_new_op.output(0)] = orig_stage->op.output(0);
  for (size_t i = 0; i < tensor_size; i++) {
    vmap[orig_stage->op.output(0)] = orig_new_op.output(0);
    rvmap[orig_new_op.output(0)] = orig_stage->op.output(0);
  }
  ReplaceDataFlow(sch->stages, &vmap, &rvmap);
  // mutate orig stage
  orig_stage->op = orig_new_op;
  orig_stage->all_iter_vars = orig_stage->op->root_iter_vars();
  orig_stage->leaf_iter_vars = orig_stage->all_iter_vars;
  orig_stage->relations = Array<IterVarRelation>();
  // create schedule for new cached stage.
  Array<Stage>& stages = sch->stages;
  size_t pos = FindNodeRef(stages.GetArrayNode(), orig_stage);
  Stage cache_stage = Stage(cache_op, sch.operator->());
  ICHECK_LT(pos, stages.size());
  stages.insert(stages.begin() + pos, cache_stage);
  // in order to obtain correct copy on schedule_record,
  // make sure "set_scope" primitive is applied after stage being added
  cache_stage.set_scope(scope);
  sch->stage_map.Set(cache_op, cache_stage);
  // Update group
  cache_stage->group = orig_stage->group;
  if (cache_stage->group.defined()) {
    ++cache_stage->group->num_child_stages;
  }
  return cache_tensor_list;
}

// Cache write and relayout the data according to loop pattern
Array<Tensor> CacheWriteWithReLayout(Schedule sch, const Array<Tensor>& tensor_array,
                                     const std::string& scope) {
  size_t tensor_size = tensor_array.size();
  sch->InvalidateCache();
  Tensor tensor = tensor_array[0];
  Stage orig_stage = sch[tensor->op];
  const ComputeOpNode* compute = orig_stage->op.as<ComputeOpNode>();

  std::unordered_set<IterVar> red_axis;
  Array<IterVar> new_axis;
  std::unordered_map<IterVar, Range> dom_map;

  std::unordered_map<const VarNode*, PrimExpr> vsub;
  std::unordered_map<const VarNode*, PrimExpr> vsub2newvar;
  std::vector<PrimExpr> predicates;

  PrepareAxisMapping(orig_stage, compute, &red_axis, &new_axis, &dom_map, &vsub, &vsub2newvar,
                     &predicates);

  PrimExpr body;
  Array<PrimExpr> body_list;
  const tir::ReduceNode* first_reduce = nullptr;
  for (auto cbody : compute->body) {
    body = VarReplacer(vsub)(cbody);
    body = InjectPredicate(predicates, body);
    body = VarReplacer(vsub2newvar)(body);
    // Reduce nodes in ONE computeOp must be the same except value_index
    // This is right only if the original body ensures Reduce nodes are the same
    if (body->IsInstance<tir::ReduceNode>()) {
      const tir::ReduceNode* reduce_body = body.as<tir::ReduceNode>();
      if (first_reduce != nullptr) {
        ICHECK(ReduceEqual(reduce_body, first_reduce));
        body = tir::Reduce(first_reduce->combiner, first_reduce->source, first_reduce->axis,
                           first_reduce->condition, reduce_body->value_index, reduce_body->init);
      } else {
        first_reduce = reduce_body;
      }
    } else {
      ICHECK(first_reduce == nullptr) << "cannot mix reduce and other node in ONE compute bodys";
    }
    body_list.push_back(body);
  }
  // The reader args
  Array<PrimExpr> args;
  {
    // cache->compute
    std::unordered_map<IterVar, PrimExpr> value_map;
    for (IterVar iv : compute->axis) {
      value_map[iv] = iv->var;
    }
    te::PassDownIndex(orig_stage, dom_map, &value_map, true);
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      args.push_back(value_map.at(iv));
    }
  }
  Operation cache_op =
      ComputeOp(compute->name + "." + scope, compute->tag, compute->attrs, new_axis, body_list);

  Array<PrimExpr> cache_expr_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_expr_list.push_back(cache_tensor(args));
  }
  Operation orig_new_op =
      ComputeOp(compute->name, compute->tag, compute->attrs, compute->axis, cache_expr_list);
  return ReplaceOriginalOp(sch, orig_stage, scope, cache_op, orig_new_op, tensor_size);
}

// for tensor compute op
Array<Tensor> CacheWriteWithReLayoutTensor(Schedule sch, const Array<Tensor>& tensor_array,
                                           const std::string& scope) {
  size_t tensor_size = tensor_array.size();
  sch->InvalidateCache();
  Tensor tensor = tensor_array[0];
  Stage orig_stage = sch[tensor->op];
  const TensorComputeOpNode* tensor_op = orig_stage->op.as<TensorComputeOpNode>();
  ICHECK_EQ(tensor_op->num_outputs(), 1)
      << "cache write only support single output tensor_compute_op";

  std::unordered_set<IterVar> red_axis;
  Array<IterVar> new_axis;
  std::unordered_map<IterVar, Range> dom_map;

  std::unordered_map<const VarNode*, PrimExpr> vsub;
  std::unordered_map<const VarNode*, PrimExpr> vsub2newvar;
  std::vector<PrimExpr> predicates;

  PrepareAxisMapping(orig_stage, tensor_op, &red_axis, &new_axis, &dom_map, &vsub, &vsub2newvar,
                     &predicates);

  for (int i = tensor_op->schedulable_ndim; i < static_cast<int>(tensor_op->axis.size()); ++i) {
    IterVar iv = tensor_op->axis[i];
    IterVar new_iv = IterVar(iv->dom, iv->var.copy_with_suffix(".c"), iv->iter_type);
    new_axis.push_back(new_iv);
  }
  Array<Region> new_regions;
  for (Region old_region : tensor_op->input_regions) {
    Region region;
    for (Range r : old_region) {
      PrimExpr min = VarReplacer(vsub2newvar)(r->min);
      PrimExpr extent = VarReplacer(vsub2newvar)(r->extent);
      region.push_back(Range::FromMinExtent(min, extent));
    }
    new_regions.push_back(region);
  }

  Array<PrimExpr> new_scalar_inputs;
  for (PrimExpr old_input : tensor_op->scalar_inputs) {
    new_scalar_inputs.push_back(VarReplacer(vsub2newvar)(old_input));
  }

  Operation cache_op =
      TensorComputeOp(tensor_op->name + "." + scope, tensor_op->tag, new_axis,
                      tensor_op->reduce_axis, tensor_op->schedulable_ndim, tensor_op->intrin,
                      tensor_op->inputs, new_regions, new_scalar_inputs);

  // axis will be used in generating compute op
  Array<IterVar> compute_axis = tensor_op->axis;
  for (size_t i = tensor_op->schedulable_ndim; i < tensor_op->axis.size(); ++i) {
    IterVar iv = tensor_op->axis[i];
    IterVar aiv = IterVar(iv->dom, iv->var, kDataPar);
    compute_axis.Set(i, aiv);
  }

  // The reader args
  Array<PrimExpr> args;
  {
    // cache->compute
    std::unordered_map<IterVar, PrimExpr> value_map;
    for (IterVar iv : compute_axis) {
      value_map[iv] = iv->var;
    }
    PassDownIndex(orig_stage, dom_map, &value_map, true);
    for (IterVar iv : orig_stage->leaf_iter_vars) {
      if (red_axis.count(iv)) continue;
      args.push_back(value_map.at(iv));
    }
    // tensorized region axis
    for (size_t i = tensor_op->schedulable_ndim; i < tensor_op->axis.size(); ++i) {
      IterVar iv = compute_axis[i];
      args.push_back(value_map.at(iv));
    }
  }

  Array<PrimExpr> cache_expr_list;
  for (size_t i = 0; i < tensor_size; i++) {
    Tensor cache_tensor = cache_op.output(i);
    cache_expr_list.push_back(cache_tensor(args));
  }
  Operation orig_new_op =
      ComputeOp(tensor_op->name, tensor_op->tag, {}, compute_axis, cache_expr_list);
  return ReplaceOriginalOp(sch, orig_stage, scope, cache_op, orig_new_op, tensor_size);
}

Array<Tensor> Schedule::cache_write(const Array<Tensor>& tensor_array, const std::string& scope) {
  (*this)->InvalidateCache();
  ICHECK(tensor_array.size() > 0) << "size of tensor_array must be greater than 0";
  Tensor tensor = tensor_array[0];
  Stage orig_stage = operator[](tensor->op);
  const ComputeOpNode* compute = tensor->op.as<ComputeOpNode>();
  ICHECK(static_cast<size_t>(compute->num_outputs()) == tensor_array.size())
      << "size of input tensor list must be same as number of stage outputs";
  for (size_t i = 1; i < tensor_array.size(); i++) {
    Stage tmp_stage = operator[](tensor_array[i]->op);
    ICHECK(orig_stage.same_as(tmp_stage)) << "Input tensor list must be generated by ONE computeOp";
  }
  return CacheWriteWithReLayout(*this, tensor_array, scope);
}

Tensor Schedule::cache_write(const Tensor& tensor, const std::string& scope) {
  // support original compute and tensor compute both
  (*this)->InvalidateCache();
  if (tensor->op.as<ComputeOpNode>()) {
    return (CacheWriteWithReLayout(*this, {tensor}, scope))[0];
  } else if (tensor->op.as<TensorComputeOpNode>()) {
    return (CacheWriteWithReLayoutTensor(*this, {tensor}, scope))[0];
  } else {
    LOG(FATAL) << "cache write only take ComputeOp or TensorComputeOp as writers";
  }
}

void RebaseNonZeroMinLoop(ScheduleNode* sch) {
  std::unordered_map<IterVar, IterVar> rebase_map;
  for (Stage s : sch->stages) {
    if (s->attach_type == kInlinedAlready) continue;

    auto root_iter_vars = s->op->root_iter_vars();
    ArrayNode* leaf_vars = s->leaf_iter_vars.CopyOnWrite();
    for (IterVar iv : root_iter_vars) {
      size_t idx = FindNodeRef(leaf_vars, iv);
      auto it = s->iter_var_attrs.find(iv);
      // don;t need to rebase path that are binded.
      if (it != s->iter_var_attrs.end() && (*it).second->bind_thread.defined()) {
        continue;
      }
      if (idx < leaf_vars->size()) {
        // insert rebase
        IterVar rebased = IterVar(Range(), iv->var.copy_with_suffix(""), iv->iter_type);
        s->relations.push_back(te::Rebase(iv, rebased));
        if (s->iter_var_attrs.count(iv)) {
          s->iter_var_attrs.Set(rebased, s->iter_var_attrs.at(iv));
        }
        leaf_vars->SetItem(idx, rebased);
        rebase_map[iv] = rebased;
      }
    }
  }
  // remap the parent relation
  for (Stage s : sch->stages) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
  for (Stage s : sch->groups) {
    if (s->attach_type != kScope) continue;
    if (rebase_map.count(s->attach_ivar)) {
      s->attach_ivar = rebase_map.at(s->attach_ivar);
    }
  }
}

void InjectInline(ScheduleNode* sch, bool feature_extraction_mode) {
  sch->InvalidateCache();

  std::vector<Array<PrimExpr>> new_body(sch->stages.size());
  std::vector<bool> changed(sch->stages.size(), false);
  std::vector<Stmt> new_hybrid_body(sch->stages.size());
  std::vector<bool> hybrid_changed(sch->stages.size(), false);
  // (sshtin): this workaround allows to inline extern ops into their consumer.
  // All inputs for extern op should not be inlined because inlining may happen
  // before TE generation for particular extern op. That may lead to
  // crash during lowering or building stages.
  // The problem description:
  // In case of operations fusing, arguments inlining
  // prevents creation of ProducerNode for extern operation.
  // Instead of the creation it is supposed to use operation argument as inlined buffer
  // but extern_op TIR generation can be peformed after inlining procedure so
  // newly generated TIR does not have reference to input data at all.
  std::unordered_map<Operation, Operation> ext_ops;
  for (size_t i = 0; i < sch->stages.size(); i++) {
    Stage stage = sch->stages[i];
    auto ext_op = stage->op.as<ExternOpNode>();
    if (ext_op) {
      auto inps = ext_op->InputTensors();
      for (size_t ii = 0; ii < inps.size(); ++ii) {
        if (ext_ops.find(inps[ii]->op) == ext_ops.end()) {
          ext_ops[inps[ii]->op] = stage->op;
        }
      }
    }
  }
  // inline all the ops
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage stage = sch->stages[i - 1];
    if (stage->attach_type == kInline) {
      stage->attach_type = kInlinedAlready;
      Array<Var> args;
      PrimExpr body;
      {
        // setup args
        const ComputeOpNode* compute = stage->op.as<ComputeOpNode>();
        ICHECK(compute) << "can only inline compute op";
        for (auto iv : compute->axis) {
          args.push_back(iv->var);
        }
        if (ext_ops.find(stage->op) != ext_ops.end()) {
          // sshtin: The extern op can try to get access to the input tensors as a raw data,
          // that can lead to error in IR builder.
          stage->attach_type = kGroupRoot;
          continue;
        }
        ICHECK_EQ(compute->body.size(), 1U) << "can only inline compute op with 1 output";
        if (feature_extraction_mode && compute->attrs.count("const_matrix")) {
          // Use constant value to replace access of const matrices.
          // This produces wrong IR but is good enough for feature extraction purposes.
          // This simplification can accelerate the feature extration and evolutionary search.
          body = make_const(compute->output_dtype(0), 1.0f);
        } else {
          body = compute->body[0];
        }
      }
      for (size_t j = i; j < sch->stages.size(); ++j) {
        Stage s = sch->stages[j];
        const ComputeOpNode* compute = s->op.as<ComputeOpNode>();
        const HybridOpNode* hybrid = s->op.as<HybridOpNode>();
        if (compute) {
          if (!new_body[j].size()) {
            new_body[j] = compute->body;
          }
          if (new_body[j][0]->IsInstance<tir::ReduceNode>()) {
            // specially handle reduction inline for multiplre reductions.
            const tir::ReduceNode* reduce = new_body[j][0].as<tir::ReduceNode>();
            for (size_t k = 1; k < new_body[j].size(); ++k) {
              const tir::ReduceNode* reduce_ = new_body[j][k].as<tir::ReduceNode>();
              ICHECK(reduce_);
              ICHECK(ReduceEqual(reduce_, reduce)) << "The Reduce inputs of ComputeOp should "
                                                   << "have the same attribute except value_index";
            }
            PrimExpr new_value = Inline(tir::Evaluate(new_body[j][0]), stage->op, args, body)
                                     .as<tir::EvaluateNode>()
                                     ->value;
            if (!new_value.same_as(new_body[j][0])) {
              changed[j] = true;
              const tir::ReduceNode* r = new_value.as<tir::ReduceNode>();
              ICHECK(r != nullptr);
              ICHECK_EQ(new_body[j].size(), r->source.size());
              for (size_t k = 0; k < new_body[j].size(); ++k) {
                auto n = make_object<tir::ReduceNode>(*r);
                n->value_index = static_cast<int>(k);
                n->dtype = r->source[k].dtype();
                new_body[j].Set(k, PrimExpr(n));
              }
            }
          } else {
            for (size_t k = 0; k < new_body[j].size(); ++k) {
              PrimExpr new_value = Inline(tir::Evaluate(new_body[j][k]), stage->op, args, body)
                                       .as<tir::EvaluateNode>()
                                       ->value;
              if (!new_value.same_as(new_body[j][k])) {
                new_body[j].Set(k, new_value);
                changed[j] = true;
              }
            }
          }
        } else if (hybrid) {
          if (!new_hybrid_body[j].defined()) {
            new_hybrid_body[j] = hybrid->body;
          }
          Stmt new_stmt = Inline(new_hybrid_body[j], stage->op, args, body);
          if (!new_stmt.same_as(new_hybrid_body[j])) {
            new_hybrid_body[j] = new_stmt;
            hybrid_changed[j] = true;
          }
        }
      }
    }
  }
  std::unordered_map<Tensor, Tensor> repl;
  // rewrite dataflow
  for (size_t i = 0; i < sch->stages.size(); ++i) {
    Stage s = sch->stages[i];
    if (s->attach_type == kInlinedAlready) continue;
    if (new_body[i].size()) {
      // Logics from ReplaceDataFlow
      const ComputeOpNode* compute = sch->stages[i]->op.as<ComputeOpNode>();
      ICHECK(compute);
      Operation op = s->op;
      if (changed[i]) {
        op = ComputeOp(compute->name, compute->tag, compute->attrs, compute->axis, new_body[i]);
      }
      op = op->ReplaceInputs(op, repl);
      if (!op.same_as(s->op)) {
        for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
          repl[s->op.output(idx)] = op.output(idx);
        }
        s->op = op;
      }
    } else if (hybrid_changed[i]) {
      const HybridOpNode* hybrid = sch->stages[i]->op.as<HybridOpNode>();
      ICHECK(hybrid);
      Operation op = HybridOp(hybrid->name, hybrid->tag, hybrid->attrs, hybrid->inputs,
                              hybrid->outputs, new_hybrid_body[i]);
      op = op->ReplaceInputs(op, repl);
      for (int idx = 0; idx < s->op->num_outputs(); ++idx) {
        repl[s->op.output(idx)] = op.output(idx);
      }
      s->op = op;
    } else {
      Operation op = s->op->ReplaceInputs(s->op, repl);
      if (!op.same_as(s->op)) {
        for (int j = 0; j < op->num_outputs(); ++j) {
          repl[s->op.output(j)] = op.output(j);
        }
        s->op = op;
      }
    }
  }
}

void LegalizeInvalidAttach(ScheduleNode* sch) {
  // Legalize the compute_at location if the target iterator of compute_at is split or fused.
  // Case 1: If the target of compute_at is split,
  //         we will move the compute_at location to the inner iterator.
  // Case 2: If the target of compute_at is fused,
  //         we will move the compute_at location to the newly fused iterator.
  // Note that case 2 can only happen if the target of compute_at
  // is the innermost operand of fuse operation.

  // Map an old invalid attach point to its new valid attach point
  std::unordered_map<IterVar, IterVar> replace_map;

  for (Stage stage : sch->stages) {
    std::unordered_set<const Object*> visited;
    for (Stage s = stage; s.defined();) {
      // The following logic is simiar to the `CreateAttachPath` in `src/te/schedule/graph.h`,
      // because we follow the validation check in that function to legalize the attach.
      ICHECK(!visited.count(s.get())) << "Find loop in compute_at attach group";
      visited.insert(s.get());
      Stage spec = s.GetAttachSpec();
      if (spec->attach_type != kScope) {
        break;
      }
      bool start_attach = false;
      IterVar attach_ivar = spec->attach_ivar;
      s = spec->attach_stage;
      ICHECK(attach_ivar.defined());
      ICHECK(s.defined());

      for (size_t i = s->leaf_iter_vars.size(); i != 0; --i) {
        IterVar iv = s->leaf_iter_vars[i - 1];
        if (!start_attach && iv.same_as(attach_ivar)) {
          start_attach = true;
          break;
        }
      }

      if (!start_attach) {
        IterVar new_attach_ivar = attach_ivar;
        bool updated = true;
        // recursively update the relations
        while (updated) {
          updated = false;
          for (const auto& rel : s->relations) {
            if (const FuseNode* r = rel.as<FuseNode>()) {
              if (new_attach_ivar.same_as(r->inner)) {
                new_attach_ivar = r->fused;
                updated = true;
              }
            } else if (const SplitNode* r = rel.as<SplitNode>()) {
              if (new_attach_ivar.same_as(r->parent)) {
                new_attach_ivar = r->inner;
                updated = true;
              }
            }
          }
          replace_map[attach_ivar] = new_attach_ivar;
        }
      }
    }
  }

  // remap the parent relation
  for (Stage s : sch->stages) {
    if (s->attach_type != kScope) continue;
    if (replace_map.count(s->attach_ivar)) {
      s->attach_ivar = replace_map.at(s->attach_ivar);
    }
  }
  for (Stage s : sch->groups) {
    if (s->attach_type != kScope) continue;
    if (replace_map.count(s->attach_ivar)) {
      s->attach_ivar = replace_map.at(s->attach_ivar);
    }
  }
}

Schedule Schedule::normalize() {
  Schedule sn = copy();
  InjectInline(sn.operator->(), false);
  RebaseNonZeroMinLoop(sn.operator->());
  LegalizeInvalidAttach(sn.operator->());
  return sn;
}

Schedule Schedule::normalize_for_feature_extraction() {
  Schedule sn = copy();
  InjectInline(sn.operator->(), true);
  RebaseNonZeroMinLoop(sn.operator->());
  LegalizeInvalidAttach(sn.operator->());
  return sn;
}

// Handle reduction factor.
Array<Tensor> Schedule::rfactor(const Tensor& tensor, const IterVar& axis, int factor_axis) {
  (*this)->InvalidateCache();
  using tir::ReduceNode;
  ICHECK_EQ(axis->iter_type, kCommReduce) << "Can only factor reduction axis";
  Stage reduce_stage = operator[](tensor->op);
  const ComputeOpNode* compute_op = reduce_stage->op.as<ComputeOpNode>();
  ICHECK(compute_op) << "Can only factor ComputeOp";
  ArrayNode* leaf_vars = reduce_stage->leaf_iter_vars.CopyOnWrite();
  {
    size_t axis_pos = FindNodeRef(leaf_vars, axis);
    ICHECK_NE(axis_pos, leaf_vars->size())
        << "Cannot find IterVar " << axis << " in leaf iter vars";
  }
  // Find touched reduction axis.
  std::unordered_map<IterVar, int> touch_map;
  touch_map[axis] = 1;
  te::PassUpBitMaskOr(reduce_stage, &touch_map, true);
  te::PassDownBitMaskOr(reduce_stage, &touch_map, true);
  // skip reduction iteration.
  std::unordered_set<IterVar> skip_bound_check;
  // Verify normal axis are not touched.
  for (IterVar iv : compute_op->axis) {
    ICHECK(!touch_map.count(iv)) << "Factor axis touches normal axis.";
    skip_bound_check.insert(iv);
  }
  // get analyzer.
  arith::Analyzer analyzer;
  // Get the replace index
  std::unordered_map<IterVar, Range> dom_map;
  std::unordered_map<IterVar, PrimExpr> value_map;
  for (IterVar iv : compute_op->reduce_axis) {
    if (touch_map.count(iv)) {
      dom_map[iv] = iv->dom;
    } else {
      skip_bound_check.insert(iv);
    }
    analyzer.Bind(iv->var, iv->dom);
  }
  te::PassDownDomain(reduce_stage, &dom_map, &analyzer, true);
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv)) {
      Range dom = dom_map.at(iv);
      if (is_one(dom->extent)) {
        value_map[iv] = dom->min;
      } else {
        value_map[iv] = iv->var;
      }
    }
  }
  te::PassUpIndex(reduce_stage, dom_map, &value_map, true);
  std::vector<PrimExpr> predicates =
      MakeBoundCheck(reduce_stage, dom_map, value_map, true, skip_bound_check);

  // Get the factored op node.
  const int factor_axis_pos =
      factor_axis >= 0 ? factor_axis : static_cast<int>(compute_op->axis.size() + 1) + factor_axis;
  ICHECK_LE(factor_axis_pos, compute_op->axis.size());
  auto n = make_object<ComputeOpNode>();
  n->name = compute_op->name + ".rf";
  {
    // axis relacement.
    IterVar iv(dom_map.at(axis), axis->var, kDataPar);
    ICHECK(is_zero(iv->dom->min)) << "Can only factor reduction domain starting from 0";

    const int size = compute_op->axis.size();
    for (int idx = 0; idx < size; ++idx) {
      if (factor_axis_pos == idx) {
        n->axis.push_back(iv);
      }
      n->axis.push_back(compute_op->axis[idx]);
    }
    if (factor_axis_pos == size) {
      n->axis.push_back(iv);
    }
  }
  // predicate generation, copy not touched axis.
  int idx = tensor->value_index;
  const ReduceNode* reduce = compute_op->body[idx].as<ReduceNode>();
  ICHECK(reduce) << "Can only rfactor non-inline reductions";
  predicates.push_back(reduce->condition);

  PrimExpr predicate =
      likely(foldl([](PrimExpr a, PrimExpr b, Span span) { return logical_and(a, b, span); },
                   const_true(1), predicates));

  std::unordered_map<const VarNode*, PrimExpr> vsub;

  for (IterVar iv : compute_op->reduce_axis) {
    if (!touch_map.count(iv)) {
      n->reduce_axis.push_back(iv);
    } else {
      ICHECK(value_map.count(iv));
      PrimExpr index = value_map.at(iv);
      vsub[iv->var.get()] = index;
    }
  }

  // Copy touched axis.
  for (IterVar iv : reduce_stage->leaf_iter_vars) {
    if (touch_map.count(iv) && !iv.same_as(axis)) {
      ICHECK_EQ(iv->iter_type, kCommReduce);
      IterVar ncpy(dom_map.at(iv), iv->var, iv->iter_type, iv->thread_tag, iv->span);
      n->reduce_axis.push_back(ncpy);
    }
  }
  VarReplacer replacer(vsub);
  Array<PrimExpr> new_source =
      tir::UpdateArray(reduce->source, [&replacer](const PrimExpr& e) { return replacer(e); });

  PrimExpr new_pred = replacer(predicate);

  std::vector<PrimExpr> body;
  for (size_t idx = 0; idx < reduce->source.size(); ++idx) {
    body.emplace_back(Reduce(reduce->combiner, new_source, n->reduce_axis, new_pred, idx, {}));
  }
  n->body = Array<PrimExpr>(body);
  // refresh relations, keep the un-touched relations.
  Array<IterVarRelation> rels;
  for (IterVarRelation rel : reduce_stage->relations) {
    bool touched = false;
    if (const SplitNode* r = rel.as<SplitNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else if (const FuseNode* r = rel.as<FuseNode>()) {
      if (touch_map.count(r->fused)) touched = true;
    } else if (const RebaseNode* r = rel.as<RebaseNode>()) {
      if (touch_map.count(r->parent)) touched = true;
    } else {
      LOG(FATAL) << "unknown relation type";
    }
    if (!touched) {
      rels.push_back(rel);
    }
  }
  // initialize the factored stage.
  Operation factor_op(n);
  Array<Stage>& stages = (*this)->stages;
  size_t stage_pos = FindNodeRef(stages.GetArrayNode(), reduce_stage);
  Stage factor_stage = Stage(factor_op, this->operator->());
  factor_stage->relations = rels;
  ICHECK_LT(stage_pos, stages.size());
  stages.insert(stages.begin() + stage_pos, factor_stage);
  (*this)->stage_map.Set(factor_op, factor_stage);
  factor_stage->group = reduce_stage->group;
  if (factor_stage->group.defined()) {
    ++factor_stage->group->num_child_stages;
  }
  // Replace the old reduction.
  IterVar repl_red_axis = reduce_axis(dom_map.at(axis), axis->var->name_hint + ".v");
  Array<Tensor> factor_tensors;
  Array<Tensor> old_tensors;
  int size = factor_op->num_outputs();
  for (int idx = 0; idx < size; ++idx) {
    factor_tensors.push_back(factor_op.output(idx));
    old_tensors.push_back(reduce_stage->op.output(idx));
  }
  Array<Tensor> repl_tensors = compute(
      old_tensors[0]->shape,
      [&](const Array<Var>& i) {
        Array<PrimExpr> indices;
        const int idx_size = static_cast<int>(i.size());
        for (int idx = 0; idx < idx_size; ++idx) {
          if (factor_axis_pos == idx) {
            indices.push_back(repl_red_axis->var);
          }
          indices.push_back(i[idx]);
        }
        Array<PrimExpr> new_init = reduce->init;
        if (!reduce->init.empty()) {
          std::unordered_map<const VarNode*, PrimExpr> init_vsub;
          for (const auto& init : reduce->init) {
            if (init->IsInstance<ProducerLoadNode>()) {
              ICHECK_EQ(compute_op->axis.size(), idx_size)
                  << "'init' should have the number of dimensions as output when using with "
                     "rfactor";
              for (int idx = 0; idx < idx_size; idx++) {
                init_vsub[compute_op->axis[idx]->var.get()] = i[idx];
              }
            }
          }
          VarReplacer init_replacer(init_vsub);
          new_init = tir::UpdateArray(
              reduce->init, [&init_replacer](const PrimExpr& e) { return init_replacer(e); });
        }
        if (factor_axis_pos == idx_size) {
          indices.push_back(repl_red_axis->var);
        }
        Array<PrimExpr> factor_exprs;
        for (int idx = 0; idx < size; ++idx) {
          factor_exprs.push_back(factor_tensors[idx](indices));
        }
        Array<PrimExpr> reductions;
        Array<IterVar> axis = {repl_red_axis};
        PrimExpr cond = const_true();
        for (int idx = 0; idx < size; ++idx) {
          reductions.push_back(Reduce(reduce->combiner, factor_exprs, axis, cond, idx, new_init));
        }
        return reductions;
      },
      reduce_stage->op->name + ".repl");

  std::unordered_map<Tensor, Tensor> vmap;
  std::unordered_map<Tensor, Tensor> rvmap;
  for (int idx = 0; idx < size; ++idx) {
    vmap[old_tensors[idx]] = repl_tensors[idx];
    rvmap[repl_tensors[idx]] = old_tensors[idx];
  }
  ReplaceDataFlow((*this)->stages, &vmap, &rvmap);
  // revamp the reduction stage.
  reduce_stage->op = repl_tensors[0]->op;
  reduce_stage->all_iter_vars = repl_tensors[0]->op->root_iter_vars();
  reduce_stage->leaf_iter_vars = reduce_stage->all_iter_vars;
  reduce_stage->relations = Array<IterVarRelation>();
  return factor_tensors;
}
}  // namespace te
}  // namespace tvm
