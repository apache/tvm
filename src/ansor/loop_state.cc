/*!
 *  Copyright (c) 2020 by Contributors
 */
#include "loop_state.h"
#include <tvm/te/operation.h>
#include "utils.h"

namespace tvm {
namespace ansor {

TVM_REGISTER_OBJECT_TYPE(StepNode);
TVM_REGISTER_NODE_TYPE(StateNode);

inline std::string CleanName(const std::string& str) {
  // to make the name valid in python code
  std::string ret = str;
  StrReplace(&ret, ".", "_");
  StrReplace(&ret, "@", "_");
  StrReplace(&ret, "outer", "o");
  StrReplace(&ret, "inner", "i");
  return ret;
}

/********** Reorder **********/
ReorderStep ReorderStepNode::make(int stage_id, const std::vector<int>& after_ids) {
  auto node = make_object<ReorderStepNode>();
  node->stage_id = stage_id;
  node->after_ids = after_ids;
  return ReorderStep(node);
}

void ReorderStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                      StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];
  CHECK_EQ(after_ids.size(), axes.size());

  std::vector<IterVar> new_axes;
  new_axes.reserve(axes.size());
  for (auto i : after_ids) {
    new_axes.push_back(axes[i]);
  }
  stage.reorder(new_axes);
  (*stage_to_axes)[stage] = std::move(new_axes);
}

std::string ReorderStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                              StageToAxesMap *stage_to_axes,
                                              te::Schedule *schedule,
                                              const std::vector<Step>& transform_steps) const {
  const te::Stage& stage = (*stages)[stage_id];
  std::stringstream ss;

  ss << "s[" << CleanName(stage->op->func_name()) << "].reorder(";
  for (size_t i = 0; i < after_ids.size(); ++i) {
    ss << CleanName((*stage_to_axes)[stage][after_ids[i]]->var->name_hint);
    if (i != after_ids.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Split **********/
std::vector<IterVar> ApplySplitToSchedule(std::vector<te::Stage> *stages,
                                          StageToAxesMap *stage_to_axes,
                                          int stage_id,
                                          int iter_id,
                                          const std::vector<PrimExpr>& lengths,
                                          bool inner_to_outer) {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];

  std::vector<IterVar> outs;
  if (inner_to_outer) {
    IterVar outer = axes[iter_id], inner;
    for (int i = static_cast<int>(lengths.size()) - 1; i >= 0; i--) {
      IterVar to_split = outer;
      stage.split(to_split, lengths[i], &outer, &inner);
      outs.push_back(inner);
    }
    outs.push_back(outer);
  } else {
    IterVar outer, inner = axes[iter_id];
    for (size_t i = 0; i < lengths.size(); i++) {
      IterVar to_split = inner;
      stage.split_by_nparts(to_split, lengths[i], &outer, &inner);
      outs.push_back(outer);
    }
    outs.push_back(inner);
  }

  std::vector<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + iter_id);
  if (inner_to_outer) {
    new_axes.insert(new_axes.end(), outs.rbegin(), outs.rend());
  } else {
    new_axes.insert(new_axes.end(), outs.begin(), outs.end());
  }
  new_axes.insert(new_axes.end(), axes.begin() + iter_id + 1, axes.end());
  (*stage_to_axes)[stage] = std::move(new_axes);

  return outs;
}

std::string PrintSplitAsPythonAPI(std::vector<te::Stage> *stages,
                                  StageToAxesMap *stage_to_axes,
                                  int stage_id,
                                  int iter_id,
                                  const std::vector<PrimExpr>& lengths,
                                  bool inner_to_outer) {
  te::Stage& stage = (*stages)[stage_id];
  auto to_split = (*stage_to_axes)[stage][iter_id];
  const auto& func_name = CleanName(stage->op->func_name());
  const auto& outs = ApplySplitToSchedule(stages, stage_to_axes, stage_id,
                                          iter_id, lengths, inner_to_outer);

  std::stringstream ss;
  int size = static_cast<int>(lengths.size());
  if (inner_to_outer) {
    for (int i = size - 1; i >= 0; i--) {
      ss << CleanName(outs[size - i]->var->name_hint) << ", "
        << CleanName(outs[size - i - 1]->var->name_hint)
        << " = s[" << func_name << "].split("
        << CleanName(to_split->var->name_hint)
        << ", factor=" << lengths[i] << ")\n";
      to_split = outs[size - i];
    }
  } else {
    for (int i = 0; i < size; i++) {
      ss << CleanName(outs[i]->var->name_hint) << ", "
        << CleanName(outs[i + 1]->var->name_hint)
        << " = s[" << func_name << "].split("
        << CleanName(to_split->var->name_hint)
        << ", nparts=" << lengths[i] << ")\n";
      to_split = outs[i + 1];
    }
  }

  return ss.str();
}

SplitStep SplitStepNode::make(int stage_id, int iter_id,
                              PrimExpr extent, const std::vector<PrimExpr>& lengths,
                              bool inner_to_outer) {
  auto node = make_object<SplitStepNode>();
  node->stage_id = stage_id;
  // Extent can be a unreducible expression in some special cases
  if (extent->IsInstance<IntImmNode>()) {
    node->extent = std::move(extent);
  }
  node->iter_id = iter_id;
  node->lengths = lengths;
  node->inner_to_outer = inner_to_outer;
  return SplitStep(node);
}

std::vector<IterVar> SplitStepNode::ApplyToSchedule(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes) const {
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id,
                              lengths, inner_to_outer);
}

std::string SplitStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id,
                               lengths, inner_to_outer);
}

/********** Follow Split **********/
FollowSplitStep FollowSplitStepNode::make(int stage_id, int iter_id,
                                          int src_step_id, int n_split) {
  auto node = make_object<FollowSplitStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->src_step_id = src_step_id;
  node->n_split = n_split;
  return FollowSplitStep(node);
}

void FollowSplitStepNode::ExtractSplitLengths(const std::vector<Step>& transform_steps,
                                              std::vector<PrimExpr>* lengths) const {
  CHECK_LT(src_step_id, transform_steps.size());
  auto ps = transform_steps[src_step_id].as<SplitStepNode>();
  CHECK(ps != nullptr);

  // get lengths from src step
  lengths->reserve(n_split);
  int j = 0;
  for (; j < n_split - 1; ++j) {
    lengths->push_back(ps->lengths[j]);
  }
  PrimExpr last_factor = 1;
  for (; j < static_cast<int>(ps->lengths.size()); ++j) {
    if (ps->lengths[j].defined()) {
      last_factor *= ps->lengths[j];
    } else {
      last_factor = PrimExpr();
      break;
    }
  }
  lengths->push_back(std::move(last_factor));
}

std::vector<IterVar> FollowSplitStepNode::ApplyToSchedule(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    const std::vector<Step>& transform_steps) const {
  std::vector<PrimExpr> lengths;
  ExtractSplitLengths(transform_steps, &lengths);
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id,
                              lengths, true);
}

std::string FollowSplitStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  std::vector<PrimExpr> lengths;
  ExtractSplitLengths(transform_steps, &lengths);
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id,
                               lengths, true);
}

/********** Follow Fused Split **********/
FollowFusedSplitStep FollowFusedSplitStepNode::make(int stage_id, int iter_id,
          const std::vector<int>& src_step_ids, int level, bool factor_or_nparts) {
  auto node = make_object<FollowFusedSplitStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->src_step_ids = src_step_ids;;
  node->level = level;
  node->factor_or_nparts = factor_or_nparts;
  return FollowFusedSplitStep(node);
}

PrimExpr FollowFusedSplitStepNode::ExtractSplitLength(const std::vector<Step>& transform_steps) const {
  PrimExpr ret(1);

  for (int src_step_id : src_step_ids) {
    CHECK_LT(src_step_id, transform_steps.size());
    auto ps = transform_steps[src_step_id].as<SplitStepNode>();
    CHECK(ps != nullptr);
    if (ps->lengths[level].defined() && ret.defined()) {
      ret *= ps->lengths[level];
    } else {
      return PrimExpr();
    }
  }

  return ret;
}

std::vector<IterVar> FollowFusedSplitStepNode::ApplyToSchedule(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    const std::vector<Step>& transform_steps) const {
  const PrimExpr& length = ExtractSplitLength(transform_steps);
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id,
                              {length}, factor_or_nparts);
}

std::string FollowFusedSplitStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  const PrimExpr& length = ExtractSplitLength(transform_steps);
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id,
                              {length}, factor_or_nparts);
}


/********** Fuse **********/
FuseStep FuseStepNode::make(int stage_id, const std::vector<int>& fused_ids) {
  auto node = make_object<FuseStepNode>();
  node->stage_id = stage_id;
  node->fused_ids = fused_ids;
  return FuseStep(node);
}

IterVar FuseStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                      StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];

  Array<IterVar> to_fuse;
  for (auto i : fused_ids) {
    to_fuse.push_back(axes[i]);
  }
  IterVar fused_axis;
  stage.fuse(to_fuse, &fused_axis);
  std::vector<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + fused_ids[0]);
  new_axes.push_back(fused_axis);
  new_axes.insert(new_axes.end(), axes.begin() + fused_ids.back() + 1,
      axes.end());
  (*stage_to_axes)[stage] = std::move(new_axes);

  return fused_axis;
}

std::string FuseStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                           StageToAxesMap *stage_to_axes,
                                           te::Schedule *schedule,
                                           const std::vector<Step>& transform_steps) const {
  const auto& stage = (*stages)[stage_id];
  std::stringstream to_fuse;

  for (size_t i = 0; i < fused_ids.size(); ++i) {
    to_fuse << CleanName((*stage_to_axes)[stage][fused_ids[i]]->var->name_hint);
    if (i != fused_ids.size() - 1) {
      to_fuse << ", ";
    }
  }

  std::stringstream ss;
  const auto& fused = ApplyToSchedule(stages, stage_to_axes);

  ss << CleanName(fused->var->name_hint) << " = s["
     << CleanName(stage->op->func_name()) << "].fuse("
     << to_fuse.str() << ")\n";

  return ss.str();
}

/********** Annotation **********/
AnnotationStep AnnotationStepNode::make(int stage_id, int iter_id, IteratorAnnotation ann) {
  auto node = make_object<AnnotationStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->annotation = ann;
  return AnnotationStep(node);
}

void AnnotationStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                         StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];

  switch (annotation) {
    case kUnroll:    stage.unroll(axes[iter_id]); break;
    case kVectorize: stage.vectorize(axes[iter_id]); break;
    case kParallel:  stage.parallel(axes[iter_id]); break;
    case kVThread:   stage.bind(axes[iter_id], te::thread_axis(Range(), "vthread")); break;
    case kBlockX:    stage.bind(axes[iter_id], te::thread_axis(Range(), "blockIdx.x")); break;
    case kBlockY:    stage.bind(axes[iter_id], te::thread_axis(Range(), "blockIdx.y")); break;
    case kThreadX:
      if (axes[iter_id]->iter_type == kCommReduce) {
        const auto &thread_x = te::thread_axis(Range(), "threadIdx.x");
        stage.bind(axes[iter_id], thread_x);
        stage.set_store_predicate(thread_x->var == 0);
      } else {
        stage.bind(axes[iter_id], te::thread_axis(Range(), "threadIdx.x"));
      }
      break;
    case kThreadY:   stage.bind(axes[iter_id], te::thread_axis(Range(), "threadIdx.y")); break;
    case kNone: break;
    default: LOG(FATAL) << "Invalid Annotation " << annotation; break;
  }
}

std::string AnnotationStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                 StageToAxesMap *stage_to_axes,
                                                 te::Schedule *schedule,
                                                 const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& iter = (*stage_to_axes)[stage][iter_id];

  bool bind_reduce_iter = iter->iter_type == kCommReduce && annotation == kThreadX;
  if (bind_reduce_iter) {
    ss << "thread_x = tvm.thread_axis(\"threadIdx.x\")\n";
  }

  ss << "s[" << CleanName(stage->op->func_name()) << "].";
  switch (annotation) {
    case kUnroll:    ss << "unroll("; break;
    case kVectorize: ss << "vectorize("; break;
    case kParallel:  ss << "parallel("; break;
    case kVThread:
    case kBlockX:
    case kBlockY:
    case kThreadX:
    case kThreadY:   ss << "bind("; break;
    case kNone:      break;
    default:
      LOG(FATAL) << "Invalid annotation " << annotation; break;
  }
  ss << CleanName(iter->var->name_hint);
  switch (annotation) {
    case kVThread:   ss << ", tvm.thread_axis(\"vthread\")"; break;
    case kBlockX:    ss << ", tvm.thread_axis(\"blockIdx.x\")"; break;
    case kBlockY:    ss << ", tvm.thread_axis(\"blockIdy.y\")"; break;
    case kThreadX:
      if (bind_reduce_iter) {
        ss << ", thread_x";
      } else {
        ss << ", tvm.thread_axis(\"threadIdx.x\")";
      }
      break;
    case kThreadY:   ss << ", tvm.thread_axis(\"threadIdx.y\")"; break;
    default:         break;
  }
  ss << ")\n";

  if (bind_reduce_iter) {
    ss << "s[" << CleanName(stage->op->func_name()) << "]"
       << ".set_store_predicate(thread_x.var.equal(0))\n";
  }

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Compute at **********/
ComputeAtStep ComputeAtStepNode::make(int stage_id, int target_stage_id, int target_iter_id) {
  auto node = make_object<ComputeAtStepNode>();
  node->stage_id = stage_id;
  node->target_stage_id = target_stage_id;
  node->target_iter_id = target_iter_id;
  return ComputeAtStep(node);
}

void ComputeAtStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                        StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const IterVar& target_axis =
      (*stage_to_axes)[(*stages)[target_stage_id]][target_iter_id];
  stage.compute_at((*stages)[target_stage_id], target_axis);
}

std::string ComputeAtStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                StageToAxesMap *stage_to_axes,
                                                te::Schedule *schedule,
                                                const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& target_stage = (*stages)[target_stage_id];

  ss << "s[" << CleanName(stage->op->func_name()) << "].compute_at(s["
      << CleanName(target_stage->op->func_name()) << "], "
      << CleanName((*stage_to_axes)[target_stage][target_iter_id]->var->name_hint);

  ss << ")\n";
  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Compute Root **********/
ComputeRootStep ComputeRootStepNode::make(int stage_id) {
  auto node = make_object<ComputeRootStepNode>();
  node->stage_id = stage_id;
  return ComputeRootStep(node);
}

void ComputeRootStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                          StageToAxesMap *stage_to_axes) const {
  (*stages)[stage_id].compute_root();
}

std::string ComputeRootStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                  StageToAxesMap *stage_to_axes,
                                                  te::Schedule *schedule,
                                                  const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];

  ss << "s[" << CleanName(stage->op->func_name()) << "].compute_root()\n";
  ApplyToSchedule(stages, stage_to_axes);

  return ss.str();
}

/********** Compute Inline **********/
ComputeInlineStep ComputeInlineStepNode::make(int stage_id) {
  auto node = make_object<ComputeInlineStepNode>();
  node->stage_id = stage_id;
  return ComputeInlineStep(node);
}

void ComputeInlineStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                            StageToAxesMap *stage_to_axes) const {
  (*stages)[stage_id].compute_inline();
}

std::string ComputeInlineStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes,
    te::Schedule *schedule,
    const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];

  ss << "s[" << CleanName(stage->op->func_name()) << "].compute_inline()\n";
  ApplyToSchedule(stages, stage_to_axes);

  return ss.str();
}

/********** Pack for vec **********/
PackForVecStep PackForVecStepNode::make(int stage_id, int iter_id, int vec_size) {
  auto node = make_object<PackForVecStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->vec_size = vec_size;
  return PackForVecStep(node);
}

void PackForVecStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes, te::Schedule *schedule) const {
  LOG(FATAL) << "Not implemented";
}

std::string PackForVecStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                 StageToAxesMap *stage_to_axes,
                                                 te::Schedule *schedule,
                                                 const std::vector<Step>& transform_steps) const {
  LOG(FATAL) << "Not implemented";
  return "";
}

/********** Cache read **********/
CacheReadStep CacheReadStepNode::make(int stage_id, std::string scope_name,
                                      const std::vector<int>& reader_stage_ids) {
  auto node = make_object<CacheReadStepNode>();
  node->stage_id = stage_id;
  node->scope_name = std::move(scope_name);
  node->reader_stage_ids = reader_stage_ids;
  return CacheReadStep(node);
}

te::Tensor CacheReadStepNode::ApplyToSchedule(std::vector<te::Stage>* stages,
    StageToAxesMap *stage_to_axes, te::Schedule *schedule) const {
  te::Stage& stage = (*stages)[stage_id];

  Array<te::Operation> readers;
  for (const auto& i : reader_stage_ids) {
    readers.push_back((*stages)[i]->origin_op);
  }
  auto out = schedule->cache_read(stage->origin_op.output(0), scope_name, readers);

  const auto& new_stage = (*schedule)[out->op];
  UpdateStageAxis(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id + 1, new_stage);

  return out;
}

std::string CacheReadStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                StageToAxesMap *stage_to_axes,
                                                te::Schedule *schedule,
                                                const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  // copy stage here, for the original stage will change after apply
  auto stage = (*stages)[stage_id];
  std::vector<te::Stage> reader_stages;
  for (size_t i = 0; i < reader_stage_ids.size(); ++i) {
    reader_stages.push_back((*stages)[reader_stage_ids[i]]);
  }

  auto out = ApplyToSchedule(stages, stage_to_axes, schedule);

  ss << CleanName(out->op->func_name()) << " = "
      << "s.cache_read(" << CleanName(stage->op->func_name()) << ", \""
      << scope_name << "\", ["
      << CleanName(reader_stages[0]->op->func_name());
  for (size_t i = 1; i < reader_stage_ids.size(); ++i) {
    ss << ", " << CleanName(reader_stages[i]->op->func_name());
  }
  ss << "])\n";

  const auto& iters = out->op->root_iter_vars();
  for (size_t i = 0; i < iters.size(); ++i) {
    ss << CleanName(iters[i]->var->name_hint);
    if (i != iters.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = " << "tuple(" << CleanName(out->op->func_name())
      << ".op.axis)\n";

  return ss.str();
}

/********** Cache write **********/
CacheWriteStep CacheWriteStepNode::make(int stage_id, std::string scope_name) {
  auto node = make_object<CacheWriteStepNode>();
  node->stage_id = stage_id;
  node->scope_name = std::move(scope_name);
  return CacheWriteStep(node);
}

Array<te::Tensor> CacheWriteStepNode::ApplyToSchedule(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule) const {
  te::Stage& stage = (*stages)[stage_id];

  Array<te::Tensor> tensor_array;
  // If the target stage has multi outputs, TVM requires to cache_write
  // all of them or schedule.cache_write will raise an error
  for (auto i = 0; i < stage->op->num_outputs(); ++i) {
    tensor_array.push_back(stage->origin_op.output(i));
  }
  auto outs = schedule->cache_write(tensor_array, scope_name);

  UpdateStageAxis(stage, stage_to_axes);
  // Even if there is multi outputs, TVM schedule only generate one
  // new stage
  const auto& new_stage = (*schedule)[outs[0]->op];
  UpdateStageAxis(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id, new_stage);

  return outs;
}

std::string CacheWriteStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                 StageToAxesMap *stage_to_axes,
                                                 te::Schedule *schedule,
                                                 const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  // copy stage here, for the original stage will change after apply
  te::Stage stage = (*stages)[stage_id];

  auto outs = ApplyToSchedule(stages, stage_to_axes, schedule);

  for (size_t i = 0; i < outs.size(); ++i) {
    ss << CleanName(outs[i]->op->func_name()) << ", ";
  }
  ss << "= " << "s.cache_write(["
     << CleanName(stage->op.output(0)->op->name);
  for (auto i = 1; i < stage->op->num_outputs(); ++i) {
    ss << ", " << CleanName(stage->op.output(i)->op->name);
  }
  ss << "], \"" << scope_name << "\")\n";

  for (const auto& out : outs) {
    const auto& iters = out->op->root_iter_vars();
    for (size_t i = 0; i < iters.size(); ++i) {
      ss << CleanName(iters[i]->var->name_hint);
      if (i != iters.size() - 1) {
        ss << ", ";
      }
    }
    ss << " = " << "tuple(" << CleanName(out->op->func_name())
      << ".op.axis)"
      << " + " << "tuple(" << CleanName(out->op->func_name())
      << ".op.reduce_axis)\n";
  }

  return ss.str();
}

/********** Pragma **********/
PragmaStep PragmaStepNode::make(int stage_id, int iter_id,
                                std::string pragma_type) {
  auto node = make_object<PragmaStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->pragma_type = std::move(pragma_type);
  return PragmaStep(node);
}

void PragmaStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                     StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];
  if (StrStartsWith(pragma_type, "auto_unroll_max_step")) {
    size_t pos = pragma_type.find('$');
    int value = atoi(pragma_type.c_str() + pos + 1);
    stage.pragma(axes[iter_id], "auto_unroll_max_step", value);
    stage.pragma(axes[iter_id], "unroll_explicit", true);
  } else {
    stage.pragma(axes[iter_id], pragma_type);
  }
}

std::string PragmaStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                             StageToAxesMap *stage_to_axes,
                                             te::Schedule *schedule,
                                             const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];

  if (StrStartsWith(pragma_type, "auto_unroll_max_step")) {
    size_t pos = pragma_type.find('$');
    int value = atoi(pragma_type.c_str() + pos + 1);
    ss << "s[" << CleanName(stage->op->func_name()) << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint)
       << ", \"auto_unroll_max_step\", " << value << ")\n";
    ss << "s[" << CleanName(stage->op->func_name()) << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint)
       << ", \"unroll_explicit\", True)\n";
  } else {
    ss << "s[" << CleanName(stage->op->func_name()) << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint) << ", \""
       << pragma_type << "\")\n";
  }

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Rfactor **********/
RfactorStep RfactorStepNode::make(int stage_id, int iter_id, int factor_iter_id) {
  auto node = make_object<RfactorStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->factor_iter_id = factor_iter_id;
  return RfactorStep(node);
}

Array<te::Tensor> RfactorStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes, te::Schedule *schedule) const {
  const auto& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];

  const te::Tensor& tensor = stage->origin_op.output(0);
  const IterVar& axis = axes[iter_id];
  auto outs = schedule->rfactor(tensor, axis, factor_iter_id);

  UpdateStageAxis(stage, stage_to_axes);

  const auto& new_stage = (*schedule)[outs[0]->op];
  UpdateStageAxis(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id, new_stage);

  return outs;
}

std::string RfactorStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                               StageToAxesMap *stage_to_axes,
                               te::Schedule *schedule,
                               const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];

  const auto& tensor_name = CleanName(stage->origin_op.output(0)->op->name);
  const auto& axis_name = CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint);

  const auto& outs = ApplyToSchedule(stages, stage_to_axes, schedule);

  for (size_t i = 0; i < outs.size(); ++i) {
    ss << CleanName(outs[i]->op->func_name());
    if (i != outs.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = " << "s.rfactor("
     << tensor_name << ", "
     << axis_name << ", "
     << factor_iter_id << ")\n";

  for (const auto& out : outs) {
    const auto& iters = out->op->root_iter_vars();
    for (size_t i = 0; i < iters.size(); ++i) {
      ss << CleanName(iters[i]->var->name_hint);
      if (i != iters.size() - 1) {
        ss << ", ";
      }
    }
    ss << " = " << "tuple(" << CleanName(out->op->func_name())
      << ".op.axis)"
      << " + " << "tuple(" << CleanName(out->op->func_name())
      << ".op.reduce_axis)\n";
  }

  const auto& output = (*stages)[stage_id + 1]->op.output(0);
  const auto& iters = output->op->root_iter_vars();
  for (size_t i = 0; i < iters.size(); ++i) {
    ss << CleanName(iters[i]->var->name_hint);
    if (i != iters.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = " << "tuple(s[" << CleanName(output->op->func_name())
    << "].op.axis)"
    << " + " << "tuple(s[" << CleanName(output->op->func_name())
    << "].op.reduce_axis)\n";

  return ss.str();
}

/********** StorageAlign **********/

StorageAlignStep StorageAlignStepNode::make(int stage_id, int iter_id,
                                            int factor, int offset) {
  auto node = make_object<StorageAlignStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->factor = factor;
  node->offset = offset;
  return StorageAlignStep(node);
}

void StorageAlignStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];
  stage.storage_align(axes[iter_id], factor, offset);
}

std::string StorageAlignStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  ss << "s[" << CleanName(stage->op->func_name()) << "].storage_align("
     << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint) << ", "
     << factor << ", " << offset << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

// Maker for other classes
Iterator IteratorNode::make(std::string name, Range range,
                            IteratorType iter_type, IteratorAnnotation annotation,
                            const std::vector<Iterator>* ori_iters) {
  auto node = make_object<IteratorNode>();
  node->name = std::move(name);
  node->range = std::move(range);
  node->iter_type = iter_type;
  node->annotation = annotation;
  if (ori_iters != nullptr) {
    node->ori_iters = *ori_iters;
  }
  return Iterator(node);
}

Stage StageNode::make(te::Operation op) {
  auto node = make_object<StageNode>();
  if (op->IsInstance<te::ComputeOpNode>()) {
    node->op_type = kCompute;
    auto *pop = op.as<te::ComputeOpNode>();

    for (const auto& axis : pop->axis) {
      node->iters.push_back(IteratorNode::make(CleanName(axis->var->name_hint),
          axis->dom, kSpace, kNone));
    }
    for (const auto& axis : pop->reduce_axis) {
      node->iters.push_back(IteratorNode::make(CleanName(axis->var->name_hint),
          axis->dom, kReduce, kNone));
    }
  } else if (op->IsInstance<te::PlaceholderOpNode>()) {
    node->op_type = kPlaceholder;
  } else {
    LOG(FATAL) << "Unsupported operator type" << op->_type_key;
  }

  node->compute_at = kRoot;
  node->op = std::move(op);
  node->auto_unroll_max_step = 0;
  node->storage_offset = 0;
  return Stage(node);
}

Stage StageNode::make(te::Operation op, StageType op_type, const std::vector<Iterator>& iters,
                      ComputeAtType compute_at, int16_t auto_unroll_max_step, int storage_offset) {
  auto node = make_object<StageNode>();
  node->op = std::move(op);
  node->op_type = op_type;
  node->iters = iters;
  node->compute_at = compute_at;
  node->auto_unroll_max_step = auto_unroll_max_step;
  node->storage_offset = storage_offset;
  return Stage(node);
}

Stage StageNode::make(te::Operation op, StageType op_type, std::vector<Iterator>&& iters,
                      ComputeAtType compute_at, int16_t auto_unroll_max_step, int storage_offset) {
  auto node = make_object<StageNode>();
  node->op = std::move(op);
  node->op_type = op_type;
  node->iters = std::move(iters);
  node->compute_at = compute_at;
  node->auto_unroll_max_step = auto_unroll_max_step;
  node->storage_offset = storage_offset;
  return Stage(node);
}

State StateNode::make_empty_state() {
  auto node = make_object<StateNode>();
  node->attach_map = AttachMapNode::make();
  node->complete = false;
  node->aux_info = ObjectRef();
  return State(node);
}

State StateNode::make(const Array<te::Operation>& ops) {
  auto node = make_object<StateNode>();
  for (const auto& op : ops) {
    node->stages.push_back(StageNode::make(op));
  }
  node->attach_map = AttachMapNode::make();
  node->complete = true;
  node->aux_info = ObjectRef();
  return State(node);
}

State StateNode::make(const std::vector<Stage>& stages,
                      const std::vector<Step>& transform_steps,
                      bool complete, ObjectRef aux_info) {
  auto node = make_object<StateNode>();
  node->stages = stages;
  node->transform_steps = transform_steps;
  node->attach_map = AttachMapNode::make();
  node->complete = complete;
  node->aux_info = std::move(aux_info);
  return State(node);
}

AttachMap AttachMapNode::make() {
  auto node = make_object<AttachMapNode>();
  return AttachMap(node);
}

// Schedule primitives api
void State::reorder(int stage_id, const std::vector<Iterator>& order) {
  const Stage& stage = operator->()->stages[stage_id];

  CHECK_EQ(order.size(), stage->iters.size()) << "The order of all iterators "
                                                 "should be specified";
  std::vector<int> after_ids;
  GetIndices(stage->iters, order, &after_ids);
  ReorderStep step = ReorderStepNode::make(stage_id, after_ids);
  CopyOnWrite()->transform_steps.push_back(step);
  DoReorderStep(step);
}

std::vector<Iterator> State::split(int stage_id,
    const Iterator& it, const std::vector<PrimExpr>& lengths, bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];

  SplitStep step = SplitStepNode::make(stage_id, GetIndex(stage->iters, it),
      it->range.defined() ? it->range->extent : PrimExpr(), lengths,
      inner_to_outer);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoSplitStep(step);
}

std::vector<Iterator> State::follow_split(int stage_id,
    const Iterator& it, int src_step_id, int n_split) {
  const Stage& stage = operator->()->stages[stage_id];

  FollowSplitStep step = FollowSplitStepNode::make(stage_id,
      GetIndex(stage->iters, it), src_step_id, n_split);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoFollowSplitStep(step);
}


std::vector<Iterator> State::follow_fused_split(int stage_id, const Iterator& it,
        const std::vector<int>& src_step_ids, int level, bool factor_or_nparts) {
  const Stage& stage = operator->()->stages[stage_id];

  FollowFusedSplitStep step = FollowFusedSplitStepNode::make(stage_id,
      GetIndex(stage->iters, it), src_step_ids, level, factor_or_nparts);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoFollowFusedSplitStep(step);
}

Iterator State::fuse(int stage_id, const std::vector<Iterator>& iters) {
  const Stage& stage = operator->()->stages[stage_id];
  std::vector<int> indices;
  GetIndices(stage->iters, iters, &indices);
  FuseStep step = FuseStepNode::make(stage_id, indices);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoFuseStep(step);
}

Iterator State::vectorize(int stage_id, const Iterator& it) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step = AnnotationStepNode::make(stage_id, GetIndex(stage->iters, it),
      kVectorize);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

Iterator State::parallel(int stage_id, const Iterator& it) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step = AnnotationStepNode::make(stage_id, GetIndex(stage->iters, it),
                                                 kParallel);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

Iterator State::unroll(int stage_id, const Iterator& it, int max_unroll) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step = AnnotationStepNode::make(stage_id, GetIndex(stage->iters, it),
                                                 kUnroll);

  // don't unroll if the extent is larger than max_unroll
  if (max_unroll != -1 && it->range.defined()) {
    if (auto imm = it->range->extent.as<IntImmNode>()) {
      if (imm->value > max_unroll) {
        return it;
      }
    }
  }

  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

void State::compute_at(int stage_id, int target_stage_id, const Iterator& target_iter) {
  const Stage& target_stage = operator->()->stages[target_stage_id];
  ComputeAtStep step = ComputeAtStepNode::make(stage_id, target_stage_id,
      GetIndex(target_stage->iters, target_iter));
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeAtStep(step);
}

void State::compute_root(int stage_id) {
  ComputeRootStep step = ComputeRootStepNode::make(stage_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeRootStep(step);
}

void State::compute_inline(int stage_id) {
  ComputeInlineStep step = ComputeInlineStepNode::make(stage_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeInlineStep(step);
}

void State::pack_for_vec(int stage_id, const Iterator& target_iter, int vec_size) {
  const Stage& stage = operator->()->stages[stage_id];
  PackForVecStep step = PackForVecStepNode::make(stage_id,
      GetIndex(stage->iters, target_iter), vec_size);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoPackForVecStep(step);
}

Iterator State::bind_thread(int stage_id, const Iterator& it,
                            IteratorAnnotation thread_type) {
  const Stage& stage = operator->()->stages[stage_id];
  if (thread_type < kVThread || thread_type > kThreadY) {
    LOG(FATAL) << "thread_type error, valide: kVThread, kBlockX, kThreadX, "
               << "kThreadY";
  }
  AnnotationStep step = AnnotationStepNode::make(stage_id,
      GetIndex(stage->iters, it), thread_type);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

int State::cache_read(int stage_id, const std::string& scope_name,
                      const std::vector<int>& reader_stage_ids, const ComputeDAG& task_dag) {
  CacheReadStep step = CacheReadStepNode::make(stage_id, scope_name, reader_stage_ids);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoCacheReadStep(step, task_dag);
}

int State::cache_write(int stage_id, const std::string& scope_name,
                        const ComputeDAG& task_dag) {
  CacheWriteStep step = CacheWriteStepNode::make(stage_id, scope_name);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoCacheWriteStep(step, task_dag);
}

void State::pragma(int stage_id, const Iterator& it, const std::string& pragma_type) {
  const Stage& stage = operator->()->stages[stage_id];
  PragmaStep step = PragmaStepNode::make(stage_id, GetIndex(stage->iters, it),
                                         pragma_type);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoPragmaStep(step);
}

int State::rfactor(int stage_id, const Iterator& it, int factor_iter_id,
    const ComputeDAG& task_dag) {
  const Stage& stage = operator->()->stages[stage_id];
  RfactorStep step = RfactorStepNode::make(stage_id, GetIndex(stage->iters, it), factor_iter_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoRfactorStep(step, task_dag);
}

void State::storage_align(int stage_id, const Iterator& it, int factor,
                          int offset) {
  const Stage& stage = operator->()->stages[stage_id];
  StorageAlignStep step = StorageAlignStepNode::make(stage_id,
      GetIndex(stage->iters, it), factor, offset);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoStorageAlignStep(step);
}

// Steps' implementations
void State::DoReorderStep(const ReorderStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  std::vector<Iterator> iters;
  for (auto x : step->after_ids) {
    iters.push_back(stage->iters[x]);
  }

  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] = StageNode::make(stage->op, stage->op_type,
                                                   std::move(iters), stage->compute_at,
                                                   stage->auto_unroll_max_step,
                                                   stage->storage_offset);
}

// common part for DoSplitStep, DoFollowSplitStep, and DoFollowFusedSplitStep
std::vector<Iterator> State::DoSplitStepCommon(int stage_id, int iter_id,
                                               const std::vector<PrimExpr>& lengths,
                                               bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];
  const Iterator& it = stage->iters[iter_id];
  size_t old_iter_size = stage->iters.size();

  PrimExpr tosplit_min, tosplit_extent;
  if (it->range.defined()) {
    tosplit_min = it->range->min;
    tosplit_extent = it->range->extent;
  } else {
    tosplit_min = tosplit_extent = PrimExpr();
  }

  std::vector<Iterator> outs;
  for (size_t i = 0; i < lengths.size(); ++i) {
    PrimExpr l; std::string name;
    if (inner_to_outer) {
      l = lengths[lengths.size() - i - 1];
      name = it->name + "." + std::to_string(lengths.size() - i);
    } else {
      l = lengths[i];
      name = it->name + "." + std::to_string(i);
    }
    Iterator res;
    if (l.defined() && tosplit_min.defined() && tosplit_extent.defined()) {
      res = IteratorNode::make(name, Range::make_by_min_extent(tosplit_min, l),
                               it->iter_type, kNone);
      tosplit_min = 0;
      tosplit_extent = indexdiv(tosplit_extent + l - 1, l);
    } else {
      res = IteratorNode::make(name, Range(), it->iter_type, kNone);
      tosplit_min = tosplit_extent = PrimExpr();
    }
    outs.push_back(std::move(res));
  }

  Range range;
  if (tosplit_min.defined() && tosplit_extent.defined()) {
    range = Range::make_by_min_extent(tosplit_min, tosplit_extent);
  }
  if (inner_to_outer) {
    outs.push_back(IteratorNode::make(it->name + ".0", range, it->iter_type, kNone));
    std::reverse(outs.begin(), outs.end());
  } else {
    outs.push_back(IteratorNode::make(it->name + "." + std::to_string(lengths.size()),
                                      range, it->iter_type, kNone));
  }

  std::vector<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(), stage->iters.begin() + iter_id);
  new_iters.insert(new_iters.end(), outs.begin(), outs.end());
  new_iters.insert(new_iters.end(), stage->iters.begin() + iter_id+1, stage->iters.end());

  StateNode* pstate = CopyOnWrite();
  pstate->stages[stage_id] = StageNode::make(stage->op, stage->op_type,
          std::move(new_iters), stage->compute_at, stage->auto_unroll_max_step,
          stage->storage_offset);

  // we have to replace the iterators in attach map, these two vectors keep the replacement mapping
  std::vector<AttachMap::IterKey> from_iters;
  std::vector<AttachMap::IterKey> to_iters;
  for (size_t i = iter_id; i < old_iter_size; ++i) {
    from_iters.emplace_back(stage_id, i);
    to_iters.emplace_back(stage_id, i + lengths.size());
  }
  pstate->attach_map.ReplaceIters(from_iters, to_iters);
  return outs;
}

std::vector<Iterator> State::DoSplitStep(const SplitStep& step) {
  return DoSplitStepCommon(step->stage_id, step->iter_id, step->lengths,
                           step->inner_to_outer);
}

std::vector<Iterator> State::DoFollowSplitStep(const FollowSplitStep& step) {
  std::vector<PrimExpr> lengths;
  step->ExtractSplitLengths(operator->()->transform_steps, &lengths);
  return DoSplitStepCommon(step->stage_id, step->iter_id, lengths, true);
}

std::vector<Iterator> State::DoFollowFusedSplitStep(const FollowFusedSplitStep& step) {
  const PrimExpr& length = step->ExtractSplitLength(operator->()->transform_steps);
  return DoSplitStepCommon(step->stage_id, step->iter_id, {length}, step->factor_or_nparts);
}

Iterator State::DoFuseStep(const FuseStep& step) {
  int stage_id = step->stage_id;
  const Stage& stage = operator->()->stages[stage_id];
  int old_iter_size = static_cast<int>(stage->iters.size());

  std::string new_name;
  PrimExpr new_extent = 1;
  IteratorType new_iter_type = kSpecial;

  std::vector<Iterator> ori_iters;
  for (size_t i = 0; i < step->fused_ids.size(); ++i) {
    if (i > 0) {
      CHECK_EQ(step->fused_ids[i], step->fused_ids[i-1] + 1);
    }

    if (i != step->fused_ids.size() - 1) {
      const auto& iter_to_attached_stage = operator->()->attach_map->iter_to_attached_stages;
      if (iter_to_attached_stage.find(std::make_pair(stage_id, step->fused_ids[i]))
         != iter_to_attached_stage.end()) {
        LOG(FATAL) << "Invalid Fuse. Because you want to fuse iterators "
                      "that have been attached by some stages";
      }
    }

    const Iterator& it = stage->iters[step->fused_ids[i]];
    ori_iters.push_back(it);
    new_name += it->name + "@";

    if (it->range.defined() && new_extent.defined()) {
      new_extent = new_extent * it->range->extent;
    } else {
      new_extent = PrimExpr();
    }

    if (i == 0) {
      new_iter_type = it->iter_type;
    } else {
      if (new_iter_type != it->iter_type) {
        new_iter_type = kMixed;
      }
    }
  }

  Range range;
  if (new_extent.defined()) {
    range = Range::make_by_min_extent(0, new_extent);
  }
  Iterator new_it = IteratorNode::make(new_name, range, new_iter_type, kNone, &ori_iters);
  std::vector<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(),
                                    stage->iters.begin() + step->fused_ids.front());
  new_iters.push_back(new_it);
  new_iters.insert(new_iters.end(), stage->iters.begin() + step->fused_ids.back() + 1,
                                    stage->iters.end());

  StateNode* pstate = CopyOnWrite();
  pstate->stages[stage_id] = StageNode::make(stage->op, stage->op_type,
          std::move(new_iters), stage->compute_at, stage->auto_unroll_max_step,
          stage->storage_offset);

  // we have to replace the iterators in attach map, these two vectors keep the replacement mapping
  std::vector<AttachMap::IterKey> from_iters;
  std::vector<AttachMap::IterKey> to_iters;
  const int begin_id = step->fused_ids.front(), end_id = step->fused_ids.back();
  for (int i = 0; i < old_iter_size; ++i) {
    if (i <= begin_id) {
      continue;
    } else if (i > end_id) {  // move forward
      from_iters.emplace_back(stage_id, i);
      to_iters.emplace_back(stage_id, i - end_id + begin_id);
    } else {   // move to the fused id
      from_iters.emplace_back(stage_id, i);
      to_iters.emplace_back(stage_id, begin_id);
    }
  }
  pstate->attach_map.ReplaceIters(from_iters, to_iters);
  return new_it;
}

Iterator State::DoAnnotationStep(const AnnotationStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];
  Iterator it = stage->iters[step->iter_id];

  Iterator new_it = IteratorNode::make(it->name, it->range, it->iter_type,
      step->annotation, &it->ori_iters);
  Stage new_stage = stage;
  new_stage.CopyOnWrite()->iters[step->iter_id] = new_it;
  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] = std::move(new_stage);
  return new_it;
}

void State::DoComputeAtStep(const ComputeAtStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  // after compute_at, we don't know the accurate length information any more
  // If we do want to know the accurate lengths, we can call ComputeDAG::ReplayAndInferBound
  std::vector<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    size_t s = it->name.size();
    if (s >= 2 && it->name[s-2] == '.' && it->name[s-1] >= '1' && it->name[s-1] <= '4') {
      // We use a dangerous heuristic rule here : For multi level splitted iterators, we assume
      // their length does not change after compute_at.
      // Reason: These iterators are generated in MultiStagePolicy by multi level tiling, they will
      // be carefully compute_at their consumers. In this case, their lengths do not change.
      // We do this to keep the AnnotateCPU pass to annotate more efficiently.
      new_iters.push_back(it);
    } else {
      new_iters.push_back(IteratorNode::make(it->name, Range(), it->iter_type,
          it->annotation, &it->ori_iters));
    }
  }

  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] = StageNode::make(stage->op, stage->op_type,
          std::move(new_iters), kIter, stage->auto_unroll_max_step,
          stage->storage_offset);
  pstate->attach_map.SetComputeAtIter(step->stage_id, step->target_stage_id, step->target_iter_id);
}

void State::DoComputeRootStep(const ComputeRootStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  // after compute_root, we don't know the accurate length information any more
  // If we do want to know the accurate lengths, we can call ComputeDAG::ReplayAndInferBound
  std::vector<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    new_iters.push_back(IteratorNode::make(it->name, Range(), it->iter_type,
        it->annotation, &it->ori_iters));
  }

  // update attach map
  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] = StageNode::make(stage->op, stage->op_type,
          std::move(new_iters), kRoot, stage->auto_unroll_max_step,
          stage->storage_offset);
  pstate->attach_map.DeleteStage(step->stage_id);
}

void State::DoComputeInlineStep(const ComputeInlineStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  StateNode* pstate = CopyOnWrite();

  // CHECK the validity of compute_inline
  const auto& iter_to_attached_stages = pstate->attach_map->iter_to_attached_stages;
  for (size_t i = 0; i < stage->iters.size(); ++i) {
    CHECK_EQ(iter_to_attached_stages.count(std::make_pair(step->stage_id, i)), 0)
      << "Invalid compute_inline: Because there are some other stages "
         "that are attached to the target stage";
  }

  pstate->stages[step->stage_id].CopyOnWrite()->compute_at = kInlined;
  pstate->attach_map.DeleteStage(step->stage_id);
}

void State::DoPackForVecStep(const PackForVecStep& step) {
  LOG(FATAL) << "Not implemented";
}

// Common part for steps that add new stages (e.g. CacheReadStep, CacheWriteStep, RfactorStep)
void AddStageModificationSteps(size_t step_id, const std::vector<Step>& transform_steps,
    std::vector<Step>* replay_steps) {
  const Step& step = transform_steps[step_id];
  if (step->IsInstance<CacheWriteStepNode>() || step->IsInstance<CacheReadStepNode>()) {
    replay_steps->push_back(step);
  } else if (step->IsInstance<RfactorStepNode>()) {
    // add FuseStepNode required by rfactor
    if (step_id >= 2 && transform_steps[step_id - 2]->IsInstance<FuseStepNode>()) {
      const Step& fuse_step = transform_steps[step_id - 2];
      if (fuse_step->stage_id == step->stage_id) {
        replay_steps->push_back(fuse_step);
      }
    }
    // add SplitStepNode required by rfactor
    CHECK_GE(step_id, 1);
    CHECK(transform_steps[step_id - 1]->IsInstance<SplitStepNode>());
    const Step& split_step = transform_steps[step_id - 1];
    CHECK_EQ(split_step->stage_id, step->stage_id);
    replay_steps->push_back(split_step);
    // add RfactorStepNode
    replay_steps->push_back(step);
  }
}

int State::DoCacheReadStep(const CacheReadStep& step, const ComputeDAG& dag) {
  StateNode* pstate = CopyOnWrite();
  std::vector<Step> replay_steps;
  for (size_t i = 0; i < pstate->transform_steps.size(); ++i) {
    AddStageModificationSteps(i, pstate->transform_steps, &replay_steps);
    if (pstate->transform_steps[i].same_as(step)) {
      break;
    }
  }
  dag.ReplayAndGetDAG(replay_steps, &(pstate->task_dag));

  // target -> target + target_store
  // Should update target's op, insert new stage, update the later stage's op
  pstate->stages[step->stage_id].CopyOnWrite()->op =
      operator->()->task_dag->ops[step->stage_id];
  pstate->stages.insert(pstate->stages.begin() + step->stage_id + 1,
      StageNode::make(operator->()->task_dag->ops[step->stage_id + 1]));
  for (size_t i = step->stage_id + 2; i < operator->()->stages.size(); ++i) {
    pstate->stages[i].CopyOnWrite()->op = operator->()->task_dag->ops[i];
  }
  pstate->attach_map =
      operator->()->attach_map.ApplyStageIdOfffset(step->stage_id + 1, 1);

  return step->stage_id + 1;
}

int State::DoCacheWriteStep(const CacheWriteStep& step, const ComputeDAG& dag) {
  StateNode* pstate = CopyOnWrite();
  std::vector<Step> replay_steps;
  for (size_t i = 0; i < pstate->transform_steps.size(); ++i) {
    AddStageModificationSteps(i, pstate->transform_steps, &replay_steps);
    if (pstate->transform_steps[i].same_as(step)) {
      break;
    }
  }
  dag.ReplayAndGetDAG(replay_steps, &(pstate->task_dag));

  // target -> target_compute + target
  // Assume target stage has never been applied any steps before cache_write
  // Should insert new stage, update target stage, update the later stage's op
  pstate->stages.insert(pstate->stages.begin() + step->stage_id,
      StageNode::make(operator->()->task_dag->ops[step->stage_id]));
  pstate->stages[step->stage_id + 1] =
      StageNode::make(operator->()->task_dag->ops[step->stage_id + 1]);
  for (size_t i = step->stage_id + 2; i < operator->()->stages.size(); ++i) {
    pstate->stages[i].CopyOnWrite()->op = operator->()->task_dag->ops[i];
  }
  pstate->attach_map =
      operator->()->attach_map.ApplyStageIdOfffset(step->stage_id, 1);

  return step->stage_id;
}

void State::DoPragmaStep(const PragmaStep& step) {
  if (step->pragma_type == "debug_skip_region") {
    StateNode* pstate = CopyOnWrite();
    pstate->attach_map.DeleteStage(step->stage_id);
  } else if (StrStartsWith(step->pragma_type, "auto_unroll_max_step")) {
    StateNode* pstate = CopyOnWrite();
    StageNode* stage = pstate->stages[step->stage_id].CopyOnWrite();
    size_t pos = step->pragma_type.find('$');
    stage->auto_unroll_max_step = atoi(step->pragma_type.c_str() + pos + 1);
  } else if (step->pragma_type == "tensor_core") {
    // Nothing needs to be done here
  } else {
    LOG(FATAL) << "Invalid pragma: " << step->pragma_type;
  }
}

int State::DoRfactorStep(const RfactorStep& step, const ComputeDAG& dag) {
  StateNode* pstate = CopyOnWrite();
  const auto compute_at_type = pstate->stages[step->stage_id]->compute_at;
  std::vector<Step> replay_steps;
  for (size_t i = 0; i < pstate->transform_steps.size(); ++i) {
    AddStageModificationSteps(i, pstate->transform_steps, &replay_steps);
    if (pstate->transform_steps[i].same_as(step)) {
      break;
    }
  }
  dag.ReplayAndGetDAG(replay_steps, &(pstate->task_dag));

  // target -> target_compute + target
  // Should insert new stage, update target stage, update the later stage's op
  pstate->stages.insert(pstate->stages.begin() + step->stage_id,
      StageNode::make(operator->()->task_dag->ops[step->stage_id]));
  // maintain the compute_at type of target stage
  Stage target_stage = StageNode::make(operator->()->task_dag->ops[step->stage_id + 1]);
  target_stage.CopyOnWrite()->compute_at = compute_at_type;
  pstate->stages[step->stage_id + 1] = target_stage;

  for (size_t i = step->stage_id + 2; i < operator->()->stages.size(); ++i) {
    pstate->stages[i].CopyOnWrite()->op = operator->()->task_dag->ops[i];
  }
  pstate->attach_map =
      operator->()->attach_map.ApplyStageIdOfffset(step->stage_id, 1);

  return step->stage_id;
}

void State::DoStorageAlignStep(const StorageAlignStep& step) {
  StateNode* pstate = CopyOnWrite();
  StageNode* stage = pstate->stages[step->stage_id].CopyOnWrite();
  stage->storage_offset = step->offset;
}

void State::DoStep(const Step& step, const ComputeDAG& dag) {
  if (auto ps = step.as<ReorderStepNode>()) {
    DoReorderStep(GetRef<ReorderStep>(ps));
  } else if (auto ps = step.as<SplitStepNode>()) {
    DoSplitStep(GetRef<SplitStep>(ps));
  } else if (auto ps = step.as<FollowSplitStepNode>()) {
    DoFollowSplitStep(GetRef<FollowSplitStep>(ps));
  } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
    DoFollowFusedSplitStep(GetRef<FollowFusedSplitStep>(ps));
  } else if (auto ps = step.as<FuseStepNode>()) {
    DoFuseStep(GetRef<FuseStep>(ps));
  } else if (auto ps = step.as<AnnotationStepNode>()) {
    DoAnnotationStep(GetRef<AnnotationStep>(ps));
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    DoComputeAtStep(GetRef<ComputeAtStep>(ps));
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    DoComputeRootStep(GetRef<ComputeRootStep>(ps));
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    DoComputeInlineStep(GetRef<ComputeInlineStep>(ps));
  } else if (auto ps = step.as<PackForVecStepNode>()) {
    DoPackForVecStep(GetRef<PackForVecStep>(ps));
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    DoCacheReadStep(GetRef<CacheReadStep>(ps), dag);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    DoCacheWriteStep(GetRef<CacheWriteStep>(ps), dag);
  } else if (auto ps = step.as<PragmaStepNode>()) {
    DoPragmaStep(GetRef<PragmaStep>(ps));
  } else if (auto ps = step.as<RfactorStepNode>()) {
    DoRfactorStep(GetRef<RfactorStep>(ps), dag);
  } else if (auto ps = step.as<StorageAlignStepNode>()) {
    DoStorageAlignStep(GetRef<StorageAlignStep>(ps));
  } else {
    LOG(FATAL) << "Invalid step: " << step;
  }
}

void State::DoSteps(const std::vector<Step>& steps, const ComputeDAG& dag) {
  // Use complete rate for the study in the paper
  const char* complete_rate_str = getenv("ANSOR_PROGRAM_COMPLETE_RATE");
  double complete_rate = -1.0;
  if (complete_rate_str) {
    complete_rate = std::stod(complete_rate_str);
  }
  size_t ct = 0;

  for (const auto& step : steps) {
    if (complete_rate >= 0 && ct++ > steps.size() * complete_rate) {
      break;
    }
    DoStep(step, dag);
  }
}


void PrintStage(std::ostream* os, int stage_id, const StateNode* state, size_t base_indent,
                bool delete_trivial_loop) {
  const Stage& stage = state->stages[stage_id];

  if (stage->auto_unroll_max_step != 0) {
    for (size_t j = 0; j < base_indent; ++j) {
      *os << " ";
    }
    *os << stage->op->func_name() << " auto_unroll: "
        << stage->auto_unroll_max_step << "\n";
  }
  if (stage->storage_offset != 0) {
    for (size_t j = 0; j < base_indent; ++j) {
      *os << " ";
    }
    *os << stage->op->func_name() << " storage_offset: "
        << stage->storage_offset << "\n";
  }

  size_t indent = 0;
  for (size_t i = 0; i < stage->iters.size(); ++i) {
    const Iterator& iter = stage->iters[i];

    if (!(delete_trivial_loop && iter->range.defined() && is_one(iter->range->extent))) {
      for (size_t j = 0; j < base_indent + indent; ++j) {
        *os << " ";
      }
      switch (iter->annotation) {
        case kNone:      *os << "for "; break;
        case kUnroll:    *os << "unroll "; break;
        case kParallel:  *os << "parallel "; break;
        case kVectorize: *os << "vectorize "; break;
        case kVThread:   *os << "vthread "; break;
        case kBlockX:    *os << "gpu.blockIdx.x "; break;
        case kBlockY:    *os << "gpu.blockIdx.y "; break;
        case kThreadX:   *os << "gpu.threadIdx.x "; break;
        case kThreadY:   *os << "gpu.threadIdx.y "; break;
      }
      if (iter->range.defined()) {
        *os << iter->name << " (" << iter->range->min << "," << iter->range->extent << ")" << "\n";
      } else {
        *os << iter->name << " (None)" << "\n";
      }

      indent += 2;
    }

    if (state != nullptr) {
      AttachMap::IterKey iter_key(stage_id, i);
      auto pair = state->attach_map->iter_to_attached_stages.find(iter_key);
      if (pair != state->attach_map->iter_to_attached_stages.end()) {
        for (const auto& attach_stage_id : pair->second) {
          PrintStage(os, attach_stage_id, state, base_indent + indent, delete_trivial_loop);
        }
      }
    }
  }

  for (size_t j = 0; j < base_indent + indent; ++j) {
    *os << " ";
  }
  *os << stage->op->func_name() << " = ...\n";
}

void PrintState(std::ostream* os, const StateNode* node, bool delete_trivial_loop) {
  // Gather placeholders
  std::vector<std::string> placeholders;
  for (const auto& stage : node->stages) {
    if (stage->op_type == kPlaceholder) {
      placeholders.push_back(stage->op->name);
    }
  }

  *os << "Placeholder: ";
  for (size_t i = 0; i < placeholders.size(); ++i) {
    *os << placeholders[i];
    if (i != placeholders.size() - 1) {
      *os << ", ";
    }
  }
  *os << "\n";

  // Print all stages
  for (size_t i = 0; i < node->stages.size(); ++i) {
    const Stage& stage = node->stages[i];
    if (stage->op_type == kPlaceholder) {
      continue;
    } else if (stage->op_type == kCompute) {
      if (stage->compute_at == kRoot) {
        PrintStage(os, i, node, 0, delete_trivial_loop);
      }
    } else {
      LOG(FATAL) << "Invalid op type";
    }
  }
}

std::string State::ToStr(bool delete_trivial_loop) const {
  std::ostringstream os;
  PrintState(&os, operator->(), delete_trivial_loop);
  return os.str();
}

void AttachMap::SetComputeAtIter(int stage_id, int target_stage_id, int target_iter_id) {
  AttachMapNode* pnode = CopyOnWrite();

  // delete the current entry of stage
  DeleteStageEntry(pnode, stage_id);

  // store the new relation
  IterKey iter_key(target_stage_id, target_iter_id);
  pnode->stage_to_attach_iter[stage_id] = std::make_pair(target_stage_id, target_iter_id);
  pnode->iter_to_attached_stages[iter_key].push_back(stage_id);
}

void AttachMap::DeleteStage(int stage_id) {
  AttachMapNode* pnode = CopyOnWrite();

  // delete the entry of old stage
  DeleteStageEntry(pnode, stage_id);
}

void AttachMap::ReplaceIters(const std::vector<IterKey>& old_iters,
                             const std::vector<IterKey>& new_iters) {
  AttachMapNode* pnode = CopyOnWrite();

  CHECK_EQ(old_iters.size(), new_iters.size());
  for (size_t i = 0; i < old_iters.size(); ++i) {
    auto entry = pnode->iter_to_attached_stages.find(old_iters[i]);
    if (entry == pnode->iter_to_attached_stages.end()) {
      continue;
    }

    // replace iter in the value of `stage_to_attach_iter`
    for (const auto& s : entry->second) {
      pnode->stage_to_attach_iter[s] = new_iters[i];
    }

    // replace iter in the key of `iter_to_attached_stages`
    std::vector<int> attached_stages = std::move(entry->second);
    pnode->iter_to_attached_stages.erase(entry);
    pnode->iter_to_attached_stages[new_iters[i]] = std::move(attached_stages);
  }
}

void AttachMap::DeleteStageEntry(AttachMapNode *pnode, int stage_id) {
  auto old_entry = pnode->stage_to_attach_iter.find(stage_id);
  if (old_entry != pnode->stage_to_attach_iter.end()) {
    // delete value in `iter_to_attached_stages`
    auto entry2 = pnode->iter_to_attached_stages.find(old_entry->second);
    DeleteItem(&entry2->second, stage_id);
    if (entry2->second.size() == 0) {
      pnode->iter_to_attached_stages.erase(entry2);
    }
    // delete key in `stage_to_attach_iter`
    pnode->stage_to_attach_iter.erase(old_entry);
  }
}

AttachMap AttachMap::ApplyStageIdOfffset(int start_id, int offset) const {
  AttachMap map = AttachMapNode::make();
  auto pmap = map.CopyOnWrite();
  for (const auto& x : operator->()->stage_to_attach_iter) {
    auto key = x.first;
    if (key >= start_id) {
      key += offset;
    }
    auto value = x.second;
    if (value.first >= start_id) {
      value.first += offset;
    }
    pmap->stage_to_attach_iter.insert(std::make_pair(key, value));
  }
  for (const auto& x : operator->()->iter_to_attached_stages) {
    auto key = x.first;
    if (key.first >= start_id) {
      key.first += offset;
    }
    auto value = x.second;
    for (auto& i : value) {
      if (i >= start_id) {
        i += offset;
      }
    }
    pmap->iter_to_attached_stages.insert(std::make_pair(key, value));
  }
  return map;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<StateNode>([](const ObjectRef& ref, ReprPrinter *p) {
  auto* node = static_cast<const StateNode*>(ref.get());
  PrintState(&p->stream, node, true);
});

}  // namespace ansor
}  // namespace tvm
