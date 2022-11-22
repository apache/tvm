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

#include "./te_compiler_cache.h"

#include <tvm/driver/driver_api.h>
#include <tvm/ir/name_supply.h>
#include <tvm/ir/type_functor.h>
#include <tvm/meta_schedule/database.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/runtime/builtin_fp16.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/function.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/topi/tags.h>

#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../printer/text_printer.h"
#include "../../te/operation/create_primfunc.h"
#include "../op/memory/memory.h"
#include "../src/meta_schedule/module_equality.h"
#include "../src/meta_schedule/trace_apply.h"
#include "../transforms/meta_schedule_layout_rewrite.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace tec {

TVM_REGISTER_NODE_TYPE(LoweredOutputNode);
TVM_REGISTER_NODE_TYPE(CachedFuncNode);
TVM_REGISTER_NODE_TYPE(CCacheKeyNode);
TVM_REGISTER_NODE_TYPE(CCacheValueNode);

LoweredOutput::LoweredOutput(tvm::Array<te::Tensor> outputs, OpImplementation impl) {
  auto n = make_object<LoweredOutputNode>();
  n->outputs = std::move(outputs);
  n->implementation = std::move(impl);
  data_ = std::move(n);
}

CCacheKey::CCacheKey(Function source_func, Target target, VirtualDevice vd) {
  auto n = make_object<CCacheKeyNode>();
  n->source_func = std::move(source_func);
  n->target = std::move(target);
  n->virtual_device = std::move(vd);
  data_ = std::move(n);
}

CachedFunc::CachedFunc(tvm::Target target, GlobalVar prim_fn_var, tvm::Array<te::Tensor> inputs,
                       tvm::Array<te::Tensor> outputs, te::Schedule schedule,
                       tir::PrimFunc prim_func, tvm::Array<Integer> shape_func_param_states,
                       IRModule funcs,
                       std::unordered_map<const ConstantNode*, te::Tensor> constant_tensors) {
  auto n = make_object<CachedFuncNode>();
  n->target = target;
  n->prim_fn_var = prim_fn_var;
  n->inputs = inputs;
  n->outputs = outputs;
  n->schedule = schedule;
  n->prim_func = prim_func;
  n->shape_func_param_states = shape_func_param_states;
  n->funcs = funcs;
  n->constant_tensors = constant_tensors;
  data_ = std::move(n);
}

Array<IndexExpr> GetShape(const Array<IndexExpr>& shape) {
  // for now, we always use int32 shape when possible
  // even if the result of shape inference becomes int64.
  Array<IndexExpr> res;
  for (IndexExpr val : shape) {
    const int64_t* pval = tir::as_const_int(val);
    if (pval != nullptr) {
#ifndef TVM_INDEX_DEFAULT_I64
      ICHECK_LE(pval[0], std::numeric_limits<int32_t>::max())
          << "dimension must be less then int32_t's max value";
      ICHECK_GE(pval[0], std::numeric_limits<int32_t>::min())
          << "dimension must be less then int32_t's max value";
      res.push_back(IntImm(DataType::Int(32), *pval));
#else
      res.push_back(val);
#endif  // TVM_INDEX_DEFAULT_I64
    } else if (val->IsInstance<tir::AnyNode>()) {
      // currently all 'any' we meet in shape function are non-negative.
      res.push_back(val.as<tir::AnyNode>()->ToSizeVar());
    } else {
      res.push_back(val);
    }
  }
  return res;
}

// Helper class that is used during lowering to TE.
// It matches sequence of Ops and lower them into single TOPI operation. All supported patterns are
// enumerated in "supported_patterns_".
class QnnPatternMatcher {
 public:
  QnnPatternMatcher()
      : qnn_conv2d_op_(Op::Get("qnn.conv2d")),
        qnn_dense_op_(Op::Get("qnn.dense")),
        qnn_requantize_op_(Op::Get("qnn.requantize")),
        bias_add_op_(Op::Get("add")) {}

  // Memoize visited operations
  void Register(const CallNode* call_node) {
    ICHECK(call_node->op.as<OpNode>());
    Op op = Downcast<Op>(call_node->op);
    if (op == qnn_conv2d_op_) {
      registered_ops_.push_front(P_QConv2d);
      ICHECK(anchor_op_ == nullptr);
      anchor_op_ = call_node;
    } else if (op == qnn_requantize_op_) {
      registered_ops_.push_front(P_QRequantize);
    } else if (op == bias_add_op_) {
      registered_ops_.push_front(P_BiasAdd);
    } else if (op == qnn_dense_op_) {
      registered_ops_.push_front(P_QDense);
      ICHECK(anchor_op_ == nullptr);
      anchor_op_ = call_node;
    } else {
      registered_ops_.push_front(P_Opaque);
    }
  }

  // Check whether given Op is a part of matched pattern.
  bool find(const Op& op) {
    if (registered_ops_.empty()) return false;

    if (op == qnn_conv2d_op_ || op == qnn_requantize_op_ || op == bias_add_op_ ||
        op == qnn_dense_op_) {
      for (const auto& pat : supported_patterns_) {
        auto it =
            std::search(registered_ops_.begin(), registered_ops_.end(), pat.begin(), pat.end());
        if (it != registered_ops_.end()) return true;
      }
    }
    return false;
  }

  // returns whether given Op is last in the pattern sequence.
  bool IsLeafOp(const Op& op) { return op == qnn_requantize_op_; }

  const CallNode* GetAnchorOp() { return anchor_op_; }

  void Clear() { registered_ops_.clear(); }

 private:
  const Op& qnn_conv2d_op_;
  const Op& qnn_dense_op_;
  const Op& qnn_requantize_op_;
  const Op& bias_add_op_;

  // Main (complicated) operation in the primitive (for example qnn.conv2d, qnn.dense etc.).
  const CallNode* anchor_op_ = nullptr;

  enum POper { P_QConv2d, P_QDense, P_BiasAdd, P_QRequantize, P_Opaque };

  std::deque<POper> registered_ops_;

  const std::vector<std::deque<POper>> supported_patterns_ = {
      {P_QDense, P_BiasAdd, P_QRequantize},   // Pattern qnn.dense -> bias_add -> qnn.requantize
      {P_QDense, P_QRequantize},              // Patter qnn.dense -> qnn.requantize
      {P_QConv2d, P_BiasAdd, P_QRequantize},  // Pattern qnn.conv2d -> bias_add -> qnn.requantize
      {P_QConv2d, P_QRequantize}              // Patter qnn.conv2d -> qnn.requantize
  };
};

// Lowers Relay primitive Function to TE Compute
class LowerToTECompute : public backend::MemoizedExprTranslator<Array<te::Tensor>> {
 public:
  LowerToTECompute(Target target, NameSupply constants_name_supply)
      : target_(target),
        device_copy_op_(Op::Get("device_copy")),
        constants_name_supply_(constants_name_supply) {}

  Array<te::Tensor> Lower(const Function& relay_func) {
    for (Var param : relay_func->params) {
      Array<tvm::te::Tensor> inputs;
      for (const auto& ttype : FlattenTupleType(param->checked_type())) {
        auto name_hint = param->vid->name_hint;
        tvm::te::Tensor tensor = tvm::te::placeholder(
            GetShape(ttype->shape), ttype->dtype, (name_hint == "") ? "placeholder" : name_hint);
        inputs.push_back(tensor);
        fn_inputs_.push_back(tensor);
      }
      memo_[param] = inputs;
    }
    readable_name_stream_ << "fused";

    Array<te::Tensor> outputs = this->VisitExpr(relay_func->body);

    candidate_name_ = readable_name_stream_.str();

    constexpr static size_t kMaxFuncNameLength = 80;
    // WARNING: Please make sure to also update TVM_CRT_MAX_STRLEN_FUNCTION_NAME
    //          whenever the value of kMaxFuncNameLength changes
    if (candidate_name_.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name << candidate_name_.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hex << std::hash<std::string>{}(candidate_name_) << "_";
      candidate_name_ = truncated_name.str();
    }

    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const VarNode* op) final {
    LOG(FATAL) << "Unexpected free variable " << PrettyPrint(GetRef<Var>(op));
    return {};
  }

  Array<te::Tensor> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    void* data = op->data->data;
    DataType dtype = DataType(op->data->dtype);
    if (op->is_scalar()) {
      auto value = te::compute(
          {},
          [&](const Array<tvm::tir::Var>&) {
            if (dtype == DataType::Int(16)) {
              return make_const(dtype, static_cast<const int16_t*>(data)[0]);
            } else if (dtype == DataType::Int(8)) {
              return make_const(dtype, static_cast<const int8_t*>(data)[0]);
            } else if (dtype == DataType::UInt(8) || dtype == DataType::Bool()) {
              return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
            } else if (dtype == DataType::Int(32)) {
              return make_const(dtype, static_cast<const int32_t*>(data)[0]);
            } else if (dtype == DataType::Int(64)) {
              return make_const(dtype, static_cast<const int64_t*>(data)[0]);
            } else if (dtype == DataType::Float(16)) {
              return make_const(dtype, __gnu_h2f_ieee(static_cast<const uint16_t*>(data)[0]));
            } else if (dtype == DataType::Float(32)) {
              return make_const(dtype, static_cast<const float*>(data)[0]);
            } else if (dtype == DataType::Float(64)) {
              return make_const(dtype, static_cast<const double*>(data)[0]);
            } else {
              LOG(FATAL) << dtype << " not handled";
              return tvm::PrimExpr();
            }
          },
          "compile_engine_const", topi::kBroadcast);
      scalars_.push_back(value->op);
      return {value};
    } else {
      const auto* ttype = op->checked_type().as<TensorTypeNode>();
      std::stringstream ss;
      std::string s = readable_name_stream_.str();
      std::replace(s.begin(), s.end(), '.', '_');
      ss << constants_name_supply_->FreshName(s + "_constant");
      tvm::te::Tensor tensor = tvm::te::placeholder(GetShape(ttype->shape), ttype->dtype, ss.str());
      constant_tensors_[op] = tensor;
      return {tensor};
    }
  }

  Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
    static auto flower_call = tvm::runtime::Registry::Get("relay.backend.lower_call");
    ICHECK(flower_call) << "relay.backend.lower_call is not registered.";

    pattern_matcher_.Register(call_node);

    Array<te::Tensor> inputs;
    // int count_tuple = 0;
    for (Expr arg : call_node->args) {
      if (arg->checked_type().as<TupleTypeNode>()) {
        // ++count_tuple;
      }
      for (te::Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
    }

    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);

    // TODO(mbs): device_copy cleanup
    ICHECK_NE(op, device_copy_op_) << "device_copy cannot be lowered";

    Array<te::Tensor> outputs;

    if (pattern_matcher_.find(op)) {
      if (pattern_matcher_.IsLeafOp(op)) {
        // Lower anchor op when pattern leaf op was reached
        auto anchor_op = pattern_matcher_.GetAnchorOp();
        LoweredOutput lowered_out =
            (*flower_call)(GetRef<Call>(anchor_op), inputs, target_, call_node->checked_type());
        outputs = lowered_out->outputs;
        Op a_op = Downcast<Op>(anchor_op->op);
        op_implementations_[a_op.operator->()] = lowered_out->implementation;

        pattern_matcher_.Clear();
      } else {
        // Forward inputs as "outputs" for successor.
        readable_name_stream_ << '_' << op->name;
        return inputs;
      }
    } else {
      LoweredOutput lowered_out = (*flower_call)(GetRef<Call>(call_node), inputs, target_);
      outputs = lowered_out->outputs;
      op_implementations_[op.operator->()] = lowered_out->implementation;
    }

    if (outputs.size() != 1) {
      const auto* tuple_type = call_node->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type) << "Expected output to be a tuple type "
                         << PrettyPrint(call_node->checked_type());

      ICHECK_EQ(tuple_type->fields.size(), outputs.size());
    }

    readable_name_stream_ << '_' << op->name;
    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Primitive Functions can not contain nested functions.";
    return Array<te::Tensor>();
  }

  Array<te::Tensor> VisitExpr_(const LetNode* op) final {
    Array<te::Tensor> val = VisitExpr(op->value);
    ICHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<te::Tensor> VisitExpr_(const TupleNode* op) final {
    Array<te::Tensor> fields;
    for (Expr field : op->fields) {
      // TODO(mbs): Generalize to be equivalent to FlattenTupleType.
      ICHECK(field->checked_type().as<TensorTypeNode>()) << "Only allow Tuple of Tensor";
      Array<te::Tensor> res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<te::Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    const auto* tuple_type = op->tuple->type_as<TupleTypeNode>();
    Array<te::Tensor> tuple = VisitExpr(op->tuple);
    ICHECK_EQ(tuple_type->fields.size(), tuple.size());
    ICHECK_GE(op->index, 0);
    ICHECK_LT(static_cast<size_t>(op->index), tuple.size());
    return {tuple[op->index]};
  }

 public:
  // Additional outputs
  Array<tvm::te::Tensor> fn_inputs_;
  Array<te::Operation> scalars_;
  std::unordered_map<const ConstantNode*, te::Tensor> constant_tensors_;
  std::unordered_map<const OpNode*, OpImplementation> op_implementations_;
  std::string candidate_name_;

 private:
  QnnPatternMatcher pattern_matcher_;

  tvm::Target target_;
  std::ostringstream readable_name_stream_;
  // Cache device copy op for equivalence checking to reduce registry lookup
  // overhead for each invocation of call node when retrieving schedules.
  const Op& device_copy_op_;
  // A NameSupply object passed from a caller, used to assign unique names to constants
  // across different invocations of LowerToTECompute.
  NameSupply constants_name_supply_;
};

using namespace tvm::tir;

class LayoutFreeConstantCollector : public StmtVisitor {
 public:
  Array<runtime::NDArray> constants;

 private:
  void VisitStmt_(const BlockNode* op) final {
    StmtVisitor::VisitStmt_(op);
    if (Optional<ObjectRef> ann = op->annotations.Get("layout_free_placeholders")) {
      for (Buffer buffer : Downcast<Array<Buffer>>(ann)) {
        layout_free_buffer_vars_.insert(buffer->data.get());
      }
    }
  }

  void VisitStmt_(const AllocateConstNode* op) final {
    StmtVisitor::VisitStmt_(op);
    if (auto it = layout_free_buffer_vars_.find(op->buffer_var.get());
        it != layout_free_buffer_vars_.end()) {
      constants.push_back(op->data.value());
    }
  }

  std::unordered_set<const tir::VarNode*> layout_free_buffer_vars_;
};

using NDArrayMap =
    std::unordered_map<runtime::NDArray, runtime::NDArray, ObjectPtrHash, ObjectPtrEqual>;

// Replace constants in AllocateConst nodes according to the given mapping
class AllocateConstReplaceConstant : public StmtExprMutator {
 public:
  explicit AllocateConstReplaceConstant(const NDArrayMap& constant_map)
      : constant_map_(constant_map) {}

  static PrimFunc Rewrite(PrimFunc f, const NDArrayMap& constant_map) {
    AllocateConstReplaceConstant rewriter(constant_map);
    PrimFuncNode* n = f.CopyOnWrite();
    n->body = rewriter(std::move(n->body));
    return f;
  }

 private:
  Stmt VisitStmt_(const AllocateConstNode* op) final {
    if (auto it = constant_map_.find(op->data.value()); it != constant_map_.end()) {
      auto rewriten_constant = it->second;
      Array<PrimExpr> rewritten_extents;
      for (auto s : rewriten_constant.Shape()) {
        rewritten_extents.push_back(PrimExpr(static_cast<int>(s)));
      }
      return AllocateConst(op->buffer_var, op->dtype, rewritten_extents, rewriten_constant,
                           op->body, op->annotations, op->span);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  NDArrayMap constant_map_;
};

// Construct a schedule for a given Relay primitive function and target.
class ScheduleBuilder : public ExprVisitor {
 public:
  explicit ScheduleBuilder(Target target)
      : target_(target),
        mod_eq_structural_(meta_schedule::ModuleEquality::Create("ignore-ndarray")) {
    // Whether to use auto_scheduler schedule.
    use_auto_scheduler_ = backend::IsAutoSchedulerEnabled();
    if (backend::IsMetaScheduleEnabled()) {
      database_ = meta_schedule::Database::Current();
      CHECK(database_.defined()) << "ValueError: `use_meta_schedule` is enabled in Relay "
                                    "build, but no `meta_schedule.Database` context is provided. ";
    } else {
      database_ = NullOpt;
    }
  }

  CachedFunc Create(const Function& relay_func, GlobalVarSupply global_var_supply,
                    NameSupply constant_name_supply) {
    LowerToTECompute lower_te_compute(target_, constant_name_supply);
    Array<te::Tensor> tensor_outs = lower_te_compute.Lower(relay_func);
    Array<te::Tensor> fn_inputs = lower_te_compute.fn_inputs_;
    VisitExpr(relay_func->body);

    // TODO(mbs): This should be the definitive global by which the PrimFunc is known and
    // no other GlobalVar ctors should appear inside the lowering machinery.
    auto prim_fn_var = global_var_supply->FreshGlobal(lower_te_compute.candidate_name_);
    prim_fn_var->checked_type_ = relay_func->checked_type();

    // Fusion over tupled results may leave identity relationships
    // between inputs and outputs, copy identity output tensors,
    // since tir lowering do not support aliasing output to input buffer.
    for (size_t i = 0; i < tensor_outs.size(); ++i) {
      if (tensor_outs[i]->op.as<te::PlaceholderOpNode>()) {
        tensor_outs.Set(i, topi::identity(tensor_outs[i]));
      }
    }

    te::Schedule schedule{nullptr};
    tir::PrimFunc prim_func{nullptr};
    // No need to register schedule for device copy op.
    if (anchor_attrs_.as<DeviceCopyAttrs>() == nullptr) {
      if (use_auto_scheduler_) {
        const auto* fauto_schedule =
            runtime::Registry::Get("auto_scheduler.relay_integration.auto_schedule_topi_compute");
        ICHECK(fauto_schedule != nullptr)
            << "auto_scheduler.relay_integration.auto_schedule_topi_compute is not registered";
        ObjectRef obj = (*fauto_schedule)(prim_fn_var->name_hint, tensor_outs);
        if (obj.defined()) {
          schedule = Downcast<te::Schedule>(obj);
        }
      }
      if (database_) {
        using tvm::meta_schedule::TuningRecord;
        using tvm::tir::IndexMap;
        using tvm::tir::Instruction;
        using tvm::tir::InstructionKind;
        using tvm::tir::PrimFunc;
        using tvm::tir::Schedule;
        backend::FTECompilerTIRConverter tir_converter = backend::GetTIRConverter();
        Array<te::Tensor> te_args = Concat(fn_inputs, tensor_outs);
        Array<runtime::NDArray> constants;
        for (auto [const_node, te_tensor] : lower_te_compute.constant_tensors_) {
          te_args.push_back(te_tensor);
          constants.push_back(const_node->data);
        }
        if (Optional<PrimFunc> f = tir_converter(te_args, constants)) {
          IRModule query_mod = backend::PrimFuncToIRModule(f.value());
          if (Optional<TuningRecord> opt_record = database_.value()->QueryTuningRecord(
                  /*mod=*/query_mod,
                  /*target=*/target_,
                  /*workload_name=*/prim_fn_var->name_hint)) {
            LayoutFreeConstantCollector const_collector;
            const_collector(f.value()->body);

            static InstructionKind kind_transform_layout = InstructionKind::Get("TransformLayout");
            TuningRecord record = opt_record.value();
            for (const Instruction& inst : record->trace->insts) {
              if (inst->kind.same_as(kind_transform_layout)) {
                ICHECK_EQ(inst->attrs.size(), 4);
                auto index_map = Downcast<IndexMap>(inst->attrs[2]);

                if (!const_collector.constants.empty()) {
                  // In this case, RewriteLayout is acting on an AllocateConst node.
                  // After tuning, we reach this code path twice: First by
                  // the Relay MetaScheduleLayoutRewrite pass, and next by the final
                  // compilation (Relay to TE schedule lowering).
                  //
                  // Due to Relay MetaScheduleLayoutRewrite and FoldConstant passes,
                  // the Relay subgraph for which we query the database during the
                  // final compilation has its weight tensor transformed according to
                  // the index map, determined during tuning. For example,
                  //
                  // fn (%p0: Tensor[(1, 56, 56, 64), float32]) {
                  //   %0 = nn.conv2d(%p0, meta[relay.Constant][0],
                  //                       /*ty=Tensor[(4, 2, 2, 3, 3, 32, 8), float32]*/, ...);
                  //   add(%0, meta[relay.Constant][1])
                  // }
                  //
                  // Note that the database does not have an entry corresponding to such subgraphs,
                  // since an input subgraph to the tuning system always has its weight tensor in
                  // the original layout, e.g.
                  //
                  // fn (%p0: Tensor[(1, 56, 56, 64), float32]) {
                  //   %0 = nn.conv2d(%p0, meta[relay.Constant][0],
                  //                       /*ty=Tensor[(3, 3, 64, 64), float32]*/, ...);
                  //   add(%0, meta[relay.Constant][1])
                  // }
                  //
                  // Thus, in both of the two cases where we reach this code path, we need careful
                  // logic to make sure that (1) the database lookup during the final compilation
                  // succeeds and (2) the application of a schedule trace is well defined.

                  ICHECK(const_collector.constants.size() == 1)
                      << "Only one layout-free constant is supported by RewriteLayout for now";
                  auto constant = const_collector.constants[0];

                  if (constant.Shape().size() == index_map->initial_indices.size()) {
                    // This is the first case, reached during the MetaScheduleLayoutRewrite pass.
                    //
                    // A layout-free constant having the same rank as an input to the index map
                    // is assumed to be transformed by this index map.
                    // TODO(masahi): If there are multiple layout-free constants in one
                    // TIR mod (e.g. conv2d -> conv2d fusion), this assumption does not hold.
                    // We need to determine which constant the given index map acts on.
                    //
                    // We know that, during the final compilation, we will query the database
                    // for a subgraph that the tuner has never seen. We workaround this problem
                    // by adding a dummy entry to the database. The dummy entry is carefully
                    // constructed so that the lookup during the final compilation would succeed.
                    runtime::NDArray rewritten_constant = index_map->MapNDArray(constant);
                    auto f_dummy = AllocateConstReplaceConstant::Rewrite(
                        f.value(), {{constant, rewritten_constant}});
                    auto workload_dummy =
                        database_.value()->CommitWorkload(backend::PrimFuncToIRModule(f_dummy));
                    TuningRecord rec_dummy(record->trace, workload_dummy, record->run_secs,
                                           record->target, record->args_info);
                    database_.value()->CommitTuningRecord(rec_dummy);
                  } else {
                    // The constant is already transformed, so this is the second case, reached
                    // during the final compilation.
                    //
                    // The schedule trace is supposed to be applied to the weight in its original
                    // layout. But as explained above, the Relay subgraph we get in this case
                    // has its weight tensor transformed according to the corresponding index map.
                    // So effectively, we undo the layout transformation on the weight to restore
                    // the original PrimFunc that the schedule trace is supposed to act on.
                    ICHECK(index_map->inverse_index_map);
                    auto inverse_map = Downcast<IndexMap>(index_map->inverse_index_map.value());
                    ICHECK(constant.Shape().size() == inverse_map->initial_indices.size());
                    runtime::NDArray orig_constant = inverse_map->MapNDArray(constant);
                    auto f_ = AllocateConstReplaceConstant::Rewrite(f.value(),
                                                                    {{constant, orig_constant}});
                    query_mod = backend::PrimFuncToIRModule(f_);
                  }
                }
                MetaScheduleLayoutRewriter::LayoutQueuePush(index_map);
              }
            }

            Schedule sch = Schedule::Traced(query_mod, /*seed=*/-1, /*debug_mask=*/0,
                                            tir::ScheduleErrorRenderLevel::kDetail);

            if (!mod_eq_structural_->Equal(query_mod, opt_record.value()->workload->mod)) {
              // When the database lookup succeeds while structural equality check fails,
              // it implies that the anchor block based equality has been used during tuning.
              // The trace in the record cannot directly be applied to this query module.
              meta_schedule::ScheduleUsingAnchorTrace(sch, record->trace, target_);
            } else {
              record->trace->ApplyToSchedule(sch, /*remove_postproc=*/false);
            }

            IRModule mod = sch->mod();
            ICHECK_EQ(mod->functions.size(), 1);
            mod = tir::transform::RemoveWeightLayoutRewriteBlock(/*skip_ndarray_rewrite*/ false)(
                std::move(mod));
            prim_func = Downcast<PrimFunc>(mod->Lookup("main"));
            // Need to copy attrs from relay function over to prim func. Most notably the structural
            // hash.
            prim_func = WithAttrs(prim_func, relay_func->attrs->dict);
          } else {
            int dispatch = backend::UseMetaScheduleDispatch();
            // (dispatch & 2): controls whether to print TVMScript for missing TIR
            // (dispatch & 4): controls whether to raise fatal errors for missing TIR
            if (dispatch & 2) {
              LOG(WARNING) << "Cannot find workload: " << prim_fn_var->name_hint << "\n"
                           << tir::AsTVMScript(f.value());
            } else {
              LOG(WARNING) << "Cannot find workload: " << prim_fn_var->name_hint;
            }
            if (dispatch & 4) {
              LOG(FATAL);
            }
          }
        }
      }
      // Use TOPI schedule if user specified, or the function has no auto_scheduler schedule.
      if (!schedule.defined() && !prim_func.defined()) {
        if (anchor_op_.defined()) {
          auto anchor_impl = lower_te_compute.op_implementations_.find(anchor_op_.operator->());
          ICHECK(anchor_impl != lower_te_compute.op_implementations_.end());
          schedule = anchor_impl->second.Schedule(anchor_attrs_, tensor_outs, target_);
        } else {
          auto default_sched = GenericFunc::Get("schedule_injective");
          ICHECK(default_sched.defined()) << "schedule_injective not registered for " << target_;
          With<Target> tctx(target_);
          schedule = default_sched(tensor_outs);
        }
      }
      if (schedule.defined()) {
        for (const auto& scalar : lower_te_compute.scalars_) {
          if (schedule->Contain(scalar)) {
            schedule[scalar].compute_inline();
          }
        }
      }
    }

    IRModule funcs = IRModule(Map<GlobalVar, BaseFunc>({}));
    return CachedFunc(target_, prim_fn_var, fn_inputs, tensor_outs, schedule, prim_func, {}, funcs,
                      lower_te_compute.constant_tensors_);
  }

  void VisitExpr_(const CallNode* call_node) final {
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");

    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);

    for (Expr arg : call_node->args) {
      VisitExpr(arg);
    }

    int op_pattern = fpattern[op];
    if (!use_auto_scheduler_ && !database_.defined() && op_pattern >= kCommReduce) {
      ICHECK(!anchor_op_.defined() || anchor_op_pattern_ < kCommReduce)
          << "Cannot apply TOPI schedule to a primitive function with two complicated ops"
          << " anchor=" << anchor_op_ << " current=" << op;
    }
    if (op_pattern >= anchor_op_pattern_) {
      anchor_op_ = op;
      anchor_attrs_ = call_node->attrs;
      anchor_op_pattern_ = op_pattern;
    }
  }

 private:
  tvm::Target target_;
  Op anchor_op_;
  Attrs anchor_attrs_;
  int anchor_op_pattern_{0};
  bool use_auto_scheduler_;
  Optional<meta_schedule::Database> database_;
  std::unique_ptr<meta_schedule::ModuleEquality> mod_eq_structural_;
};

/*!
 * \brief Create schedule for target.
 * \param source_func The primitive function to be lowered.
 * \param target The target we want to create schedule for.
 * \return Pair of schedule and cache.
 *  The funcs field in cache is not yet populated.
 */
CachedFunc PrimFuncFor(const Function& source_func, const Target& target,
                       GlobalVarSupply global_var_supply, NameSupply constant_name_supply) {
  return ScheduleBuilder(target).Create(source_func, global_var_supply, constant_name_supply);
}

// Creates shape function from functor.
class MakeShapeFunc : public backend::MemoizedExprTranslator<Array<te::Tensor>> {
 public:
  MakeShapeFunc() {}

  CachedFunc Create(const Function& prim_func, const Target& target,
                    GlobalVarSupply global_var_supply) {
    VLOG_CONTEXT << "MakeShapeFunc";
    TShapeDataDependent shape_func_param_states;

    for (auto param : prim_func->params) {
      param_states_[param] = kNoNeed;
      Array<tvm::te::Tensor> data_inputs;
      Array<tvm::te::Tensor> shape_inputs;

      for (const auto& ttype : FlattenTupleType(param->checked_type())) {
        // Add data placeholder (in case we discover we need it below)
        Shape shape = GetShape(ttype->shape);
        tvm::te::Tensor data_tensor =
            tvm::te::placeholder(shape, ttype->dtype, "data_" + param->vid->name_hint);
        data_inputs.push_back(data_tensor);
        // Add shape placeholder (in case we discover we need it below)
        int64_t ndim = shape.size();
        Shape sshape;
        if (ndim > 0) {
          sshape.push_back(tvm::Integer(ndim));
        }
        tvm::te::Tensor shape_tensor =
            tvm::te::placeholder(sshape, DataType::Int(64), "shape_" + param->vid->name_hint);
        shape_inputs.push_back(shape_tensor);
      }
      param_data_[param] = data_inputs;
      param_shapes_[param] = shape_inputs;
    }

    // Setup the name;
    readable_name_stream_ << "shape_func";

    // Create the tensor expressions representing the output shapes.
    Array<te::Tensor> outputs = VisitExpr(prim_func->body);

    // Generate a name.
    auto candidate_name = readable_name_stream_.str();

    constexpr static size_t kMaxFuncNameLength = 80;
    // WARNING: Please make sure to also update TVM_CRT_MAX_STRLEN_FUNCTION_NAME
    //          whenever the value of kMaxFuncNameLength changes
    if (candidate_name.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name << candidate_name.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hex << std::hash<std::string>{}(candidate_name) << "_";
      candidate_name = truncated_name.str();
    }

    // Set all the inputs correctly, and accumulate their types from the p.o.v. of the
    // shape function rather than the primitive it is derived for.
    Array<te::Tensor> inputs;
    Array<Type> shape_function_arg_types;
    for (auto param : prim_func->params) {
      int state = param_states_[param];
      shape_func_param_states.push_back(IntImm(DataType::Int(32), state));
      if (state & kNeedInputData) {
        // Pass the primitive arguments directly (though in flattened form and on the host)
        for (auto t : param_data_[param]) {
          inputs.push_back(t);
          shape_function_arg_types.push_back(TensorType(t->GetShape(), t->GetDataType()));
        }
      }
      if (state & kNeedInputShape) {
        // Pass the shapes of the primitive arguments (also on the host)
        for (auto t : param_shapes_[param]) {
          inputs.push_back(t);
          shape_function_arg_types.push_back(TensorType(t->GetShape(), t->GetDataType()));
        }
      }
    }

    // TODO(mbs): This should be the definitive global by which the PrimFunc is known and
    // no  other GlobalVar ctors should appear inside the lowering machinery.
    auto prim_fn_gvar = global_var_supply->FreshGlobal(candidate_name);

    // Gather the result types, again from the p.o.v. of the shape function rather than
    // the primitive it is derived for.
    Array<Type> shape_function_res_types;
    for (const auto& t : outputs) {
      shape_function_res_types.push_back(TensorType(t->GetShape(), t->GetDataType()));
    }

    // Assign the shape function its true type.
    FuncType type(shape_function_arg_types, TupleType(shape_function_res_types),
                  /*type_params=*/{}, /*type_constraints=*/{});
    VLOG(1) << "shape function '" << prim_fn_gvar->name_hint << "' has type:" << std::endl
            << PrettyPrint(type) << std::endl
            << "corresponding to primitive of type:" << std::endl
            << PrettyPrint(prim_func->checked_type());
    prim_fn_gvar->checked_type_ = std::move(type);

    // generate schedule for shape func
    Array<te::Operation> out_ops;
    for (auto t : outputs) {
      out_ops.push_back(t->op);
    }
    te::Schedule schedule = te::create_schedule(out_ops);
    tvm::te::AutoInlineInjective(schedule);
    for (const auto& scalar : scalars_) {
      auto scalar_op = scalar->op;
      if (schedule->Contain(scalar_op)) {
        schedule[scalar_op].compute_inline();
      }
    }

    Array<te::Tensor> all_args = Array<te::Tensor>(inputs);
    for (te::Tensor arg : outputs) {
      all_args.push_back(arg);
    }

    using tvm::transform::PassContext;
    With<PassContext> fresh_pass_ctx_scope(PassContext::Create());

    std::unordered_map<te::Tensor, tir::Buffer> binds;
    IRModule lowered_module =
        tvm::LowerSchedule(schedule, all_args, prim_fn_gvar->name_hint, binds, global_var_supply);
    return CachedFunc(target, prim_fn_gvar, inputs, outputs, schedule, tir::PrimFunc{nullptr},
                      shape_func_param_states, lowered_module);
  }

  Array<te::Tensor> VisitExpr(const Expr& expr) final {
    if (expr.as<VarNode>()) {
      // Do not memoize vars because shape functions could use either the data
      // or the shape of a var each time.
      return ExprFunctor::VisitExpr(expr);
    }
    // For other case, do memoized visit
    return backend::MemoizedExprTranslator<Array<te::Tensor>>::VisitExpr(expr);
  }

  Array<te::Tensor> VisitExpr_(const VarNode* var_node) final {
    auto var = GetRef<Var>(var_node);
    auto it = param_arg_map_.find(var);
    if (it != param_arg_map_.end()) {
      // This var is a parameter of a nested function. Visit the corresponding argument in the
      // function call site.
      return VisitExpr(it->second);
    }
    if (param_states_.find(var) == param_states_.end()) {
      LOG(FATAL) << "Unexpected free variable " << PrettyPrint(var);
      return {};
    } else {
      ICHECK(data_dependents_per_input_.size());
      auto data_dependent = data_dependents_per_input_.back();
      if (data_dependent) {
        param_states_[var] |= kNeedInputData;
        return param_data_[var];
      } else {
        param_states_[var] |= kNeedInputShape;
        return param_shapes_[var];
      }
    }
  }

  Array<te::Tensor> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    ICHECK(data_dependents_per_input_.size());
    bool data_dependent = data_dependents_per_input_.back();
    if (!op->is_scalar()) {
      // This is a constant weight, extract the shape of the weight tensor.
      // This can not be data dependent.
      CHECK(!data_dependent);
      auto ttype = op->checked_type().as<TensorTypeNode>();
      int ndim = static_cast<int>(ttype->shape.size());
      Array<PrimExpr> out_shape{ndim};
      te::Tensor value = tvm::te::compute(
          out_shape,
          [&](const Array<tvm::tir::Var>& indices) {
            auto idx = indices[0];
            PrimExpr ret = make_const(DataType::Int(64), 0);
            for (int i = 0; i < ndim; i++) {
              ret = tvm::if_then_else(idx == i, ttype->shape[i], ret);
            }
            return ret;
          },
          "shape_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    }
    if (data_dependent) {
      void* data = op->data->data;
      DataType dtype = DataType(op->data->dtype);
      auto value = tvm::te::compute(
          {},
          [&](const Array<tvm::tir::Var>&) {
            if (dtype == DataType::Int(32)) {
              return make_const(dtype, static_cast<const int32_t*>(data)[0]);
            } else if (dtype == DataType::Int(64)) {
              return make_const(dtype, static_cast<const int64_t*>(data)[0]);
            } else if (dtype == DataType::Float(32)) {
              return make_const(dtype, static_cast<const float*>(data)[0]);
            } else if (dtype == DataType::Float(64)) {
              return make_const(dtype, static_cast<const double*>(data)[0]);
            } else if (dtype == DataType::Bool()) {
              return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
            } else {
              LOG(FATAL) << "not handled";
              return tvm::PrimExpr();
            }
          },
          "data_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    } else {
      auto value = tvm::te::compute(
          {}, [&](const Array<tvm::tir::Var>&) { return tir::make_const(DataType::Int(64), 0); },
          "shape_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    }
  }

  Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
    VLOG(1) << "considering call:" << std::endl << PrettyPrint(GetRef<Call>(call_node));
    if (auto* func = call_node->op.as<FunctionNode>()) {
      VLOG(1) << "user function";
      for (size_t i = 0; i < func->params.size(); ++i) {
        param_arg_map_[func->params[i]] = call_node->args[i];
      }
      return VisitExpr(func->body);
    }

    static auto fshape_func = Op::GetAttrMap<FShapeFunc>("FShapeFunc");
    static auto tshape_data_dependent = Op::GetAttrMap<TShapeDataDependent>("TShapeDataDependent");
    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);
    ICHECK(data_dependents_per_input_.empty() || !data_dependents_per_input_.back())
        << "Error in op fusion: output of the shape func is fed to a "
        << "data-dependent shape func";
    ICHECK_GT(fshape_func.count(op), 0) << "Internal error, cannot find ShapeFunc for " << op->name;
    ICHECK_GT(tshape_data_dependent.count(op), 0)
        << "Internal error, cannot find TShapeDataDependent for " << op->name;

    Array<Integer> dep_spec = tshape_data_dependent[op];
    if (dep_spec.size() == 1) {
      // This is for cases when data dependence is specified per op
      // Replicate 0 or 1 flag to all arguments
      for (size_t i = 1; i < call_node->args.size(); ++i) {
        dep_spec.push_back(dep_spec[0]);
      }
    }

    // Visit all inputs
    Array<te::Tensor> inputs;
    int count_tuple = 0;
    for (size_t i = 0; i < call_node->args.size(); ++i) {
      Expr arg = call_node->args[i];
      if (arg->checked_type().as<TupleTypeNode>()) {
        ++count_tuple;
      }
      data_dependents_per_input_.push_back(dep_spec[i]->value != 0);
      for (te::Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
      data_dependents_per_input_.pop_back();
    }
    if (count_tuple) {
      ICHECK_EQ(call_node->args.size(), 1U) << "Only allow function with a single tuple input";
    }
    // Get output ndims
    auto ret_type = call_node->checked_type();
    Array<IndexExpr> out_ndims;
    for (const auto& ttype : FlattenTupleType(ret_type)) {
      out_ndims.push_back(IntImm(DataType::Int(32), ttype->shape.size()));
    }

    // Call shape function
    Array<te::Tensor> outputs = fshape_func[op](call_node->attrs, inputs, out_ndims);
    VLOG(1) << "shape function for '" << op->name << "' with inputs:" << std::endl
            << inputs << std::endl
            << "yielded outputs:" << std::endl
            << outputs;
    readable_name_stream_ << "_" << op->name;
    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Nested functions are not allowed to be visited.";
    return Array<te::Tensor>();
  }

  Array<te::Tensor> VisitExpr_(const LetNode* op) final {
    Array<te::Tensor> val = VisitExpr(op->value);
    ICHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<te::Tensor> VisitExpr_(const TupleNode* op) final {
    Array<te::Tensor> fields;
    for (Expr field : op->fields) {
      ICHECK(field->checked_type().as<TensorTypeNode>())
          << "Expected a Tuple of Tensor, but got " << PrettyPrint(field->checked_type());
      Array<te::Tensor> res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<te::Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    Array<te::Tensor> input_shapes = VisitExpr(op->tuple);
    Array<te::Tensor> out;
    out.push_back(input_shapes[op->index]);
    return out;
  }

 private:
  /*! \brief String stream for function name */
  std::ostringstream readable_name_stream_;
  /*! \brief Map from parameter to its shape function usage state */
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> param_states_;
  /*! \brief Map from parameter to list of data placeholder */
  std::unordered_map<Expr, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual> param_data_;
  /*! \brief Map from parameter to list of shape placeholder */
  std::unordered_map<Expr, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual> param_shapes_;
  /*! \brief Stack of data dependencies for shape function, specified per each op input */
  std::vector<bool> data_dependents_per_input_;
  /*! \brief Scalars used in the shape function */
  Array<te::Tensor> scalars_;
  /*! \brief Map from parameters of a nested function to corresponding arguments in a function
   * call site.
   */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> param_arg_map_;
};

CachedFunc ShapeFuncFor(const Function& prim_func, const Target& target,
                        GlobalVarSupply global_var_supply) {
  return MakeShapeFunc().Create(prim_func, target, global_var_supply);
}

std::tuple<Array<te::Tensor>, Array<runtime::NDArray>, std::string> LowerTECompute(
    const Function& source_func, Target target, NameSupply constant_name_supply,
    bool return_inputs) {
  LowerToTECompute lower_te_compute(target, constant_name_supply);
  Array<te::Tensor> outputs = lower_te_compute.Lower(source_func);
  // Following ScheduleBuilder, remove placeholder ops from outputs.
  tvm::Array<te::Tensor> tensor_outs;
  for (const auto& tensor : outputs) {
    if (!tensor->op.as<te::PlaceholderOpNode>()) {
      tensor_outs.push_back(tensor);
    }
  }

  tvm::Array<runtime::NDArray> constants;
  for (auto [const_node, te_tensor] : lower_te_compute.constant_tensors_) {
    tensor_outs.push_back(te_tensor);
    constants.push_back(const_node->data);
  }

  if (return_inputs) {
    return std::make_tuple(Concat(lower_te_compute.fn_inputs_, tensor_outs), constants,
                           lower_te_compute.candidate_name_);
  }
  return std::make_tuple(tensor_outs, constants, lower_te_compute.candidate_name_);
}

TVM_REGISTER_GLOBAL("relay.backend.LowerToTE").set_body_typed([](Function prim_func) {
  auto tgt = tvm::Target("ext_dev");
  LowerToTECompute lower_te_compute(tgt, NameSupply(""));
  auto outputs = lower_te_compute.Lower(prim_func);
  return CachedFunc(tgt, GlobalVar(lower_te_compute.candidate_name_), lower_te_compute.fn_inputs_,
                    outputs, te::Schedule(), tir::PrimFunc(), {},
                    IRModule(Map<GlobalVar, BaseFunc>({})), lower_te_compute.constant_tensors_);
});

}  // namespace tec
}  // namespace relay
}  // namespace tvm
