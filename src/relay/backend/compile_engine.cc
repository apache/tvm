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
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/compile_engine.cc
 * \brief Internal compialtion engine.
 */
#include <tvm/schedule.h>
#include <tvm/packed_func_ext.h>
#include <tvm/operation.h>
#include <tvm/runtime/registry.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <topi/tags.h>
#include <utility>
#include <limits>
#include <mutex>
#include <functional>
#include <unordered_map>
#include "compile_engine.h"

namespace tvm {
namespace relay {

CCacheKey CCacheKeyNode::make(Function source_func, Target target) {
  auto n = make_node<CCacheKeyNode>();
  n->source_func = std::move(source_func);
  n->target = std::move(target);
  return CCacheKey(n);
}

// The getter to get schedule from compile engine.
// Get schedule from functor.
class ScheduleGetter :
      public ExprFunctor<Array<Tensor>(const Expr&)> {
 public:
  explicit ScheduleGetter(Target target)
      : target_(target) {}

  Array<IndexExpr> GetShape(const Array<IndexExpr>& shape) {
    // for now, we always use int32 shape when possible
    // even if the result of shape inference becomes int64.
    Array<IndexExpr> res;
    for (IndexExpr val : shape) {
      const int64_t* pval = as_const_int(val);
      if (pval != nullptr) {
        CHECK_LE(pval[0], std::numeric_limits<int32_t>::max());
        CHECK_GE(pval[0], std::numeric_limits<int32_t>::min());
        res.push_back(ir::IntImm::make(Int(32), *pval));
      } else {
        res.push_back(val);
      }
    }
    return res;
  }

  std::pair<Schedule, CachedFunc> Create(const Function& prim_func) {
    static auto fschedule =
        Op::GetAttr<FTVMSchedule>("FTVMSchedule");
    auto cache_node = make_node<CachedFuncNode>();
    cache_node->target = target_;
    for (Var param : prim_func->params) {
      Array<tvm::Tensor> inputs;
      if (const auto* ttype = param->checked_type().as<TensorTypeNode>()) {
        tvm::Tensor tensor = tvm::placeholder(
            GetShape(ttype->shape), ttype->dtype);
        cache_node->inputs.push_back(tensor);
        inputs.push_back(tensor);
      } else {
        // flatten tuple of tensor type.
        const auto* tuple_type = param->type_as<TupleTypeNode>();
        for (Type field : tuple_type->fields) {
          const auto* ttype = field.as<TensorTypeNode>();
          CHECK(ttype != nullptr);
          tvm::Tensor tensor = tvm::placeholder(
              GetShape(ttype->shape), ttype->dtype);
          cache_node->inputs.push_back(tensor);
          inputs.push_back(tensor);
        }
      }
      memo_[param] = inputs;
    }
    readable_name_stream_ << "fused";
    cache_node->outputs = this->VisitExpr(prim_func->body);
    auto candidate_name = readable_name_stream_.str();
    constexpr static size_t kMaxFuncNameLength = 80;
    if (candidate_name.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name <<  candidate_name.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hash<std::string>{}(candidate_name) << "_";
      candidate_name = truncated_name.str();
    }
    cache_node->func_name = candidate_name;

    CachedFunc cfunc(cache_node);
    CHECK(master_op_.defined());
    // Fusion over tupled results may leave identity relationships
    // between inputs and outputs, and those should not be scheduled.
    // Hence schedule only non PlaceholderOp outputs.
    tvm::Array<Tensor> tensor_outs;
    for (const auto& tensor : cache_node->outputs) {
      if (!tensor->op.as<PlaceholderOpNode>()) {
        tensor_outs.push_back(tensor);
      }
    }
    Schedule schedule;
    // No need to register schedule for device copy op.
    if (master_attrs_.as<DeviceCopyAttrs>() == nullptr) {
      schedule =
          fschedule[master_op_](master_attrs_, tensor_outs, target_);
      for (const auto& scalar : scalars_) {
        if (schedule->Contain(scalar)) {
          schedule[scalar].compute_inline();
        }
      }
    }
    return std::make_pair(schedule, cfunc);
  }

  Array<Tensor> VisitExpr(const Expr& expr) {
    auto it = memo_.find(expr);
    if (it != memo_.end()) {
      return it->second;
    } else {
      Array<Tensor> res = ExprFunctor::VisitExpr(expr);
      memo_[expr] = res;
      return res;
    }
  }

  Array<Tensor> VisitExpr_(const VarNode* op) final {
    LOG(FATAL) << "Free variable " << op->name_hint();
    return {};
  }

  Array<Tensor> VisitExpr_(const ConstantNode* op) final {
    CHECK(op->is_scalar());
    void* data = op->data->data;
    DataType dtype = TVMType2Type(op->data->dtype);
    Tensor value = tvm::compute({}, [&](const Array<tvm::Var>&) {
        if (dtype == Int(32)) {
          return make_const(dtype, static_cast<const int32_t*>(data)[0]);
        } else if (dtype == Int(64)) {
          return make_const(dtype, static_cast<const int64_t*>(data)[0]);
        } else if (dtype == Float(32)) {
          return make_const(dtype, static_cast<const float*>(data)[0]);
        } else if (dtype == Float(64)) {
          return make_const(dtype, static_cast<const double*>(data)[0]);
        } else if (dtype == Bool()) {
          return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
        } else {
          LOG(FATAL) << "not handled";
          return tvm::Expr();
        }
      }, "compile_engine_const", topi::kBroadcast);
    scalars_.push_back(value->op);
    return {value};
  }

  Array<Tensor> VisitExpr_(const CallNode* call_node) final {
    static auto fcompute =
        Op::GetAttr<FTVMCompute>("FTVMCompute");
    static auto fpattern =
        Op::GetAttr<TOpPattern>("TOpPattern");

    Array<Tensor> inputs;
    int count_tuple = 0;
    for (Expr arg : call_node->args) {
      if (arg->checked_type().as<TupleTypeNode>()) {
        ++count_tuple;
      }
      for (Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
    }
    if (count_tuple) {
      CHECK_EQ(call_node->args.size(), 1U)
          << "Only allow function with a single tuple input";
    }
    CHECK(call_node->op.as<OpNode>())
        << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);
    // Check if the op is a device copy op.
    bool is_copy_op = op.same_as(Op::Get("device_copy"));
    Array<Tensor> outputs;
    // Skip fcompute for device copy operators as it is not registered.
    if (is_copy_op) {
      const auto* copy_input = inputs[0].operator->();
      outputs.push_back(TensorNode::make(copy_input->shape, copy_input->dtype,
                                         Operation(), 0));
    } else {
      outputs = fcompute[op](call_node->attrs, inputs,
                             call_node->checked_type(), target_);
    }

    int op_pattern = fpattern[op];
    if (op_pattern >= kCommReduce) {
      CHECK(!master_op_.defined() || master_op_pattern_ < kCommReduce)
          << "Two complicated op in a primitive function "
          << " master=" << master_op_ << " current=" << op;
    }
    if (op_pattern >= master_op_pattern_) {
      master_op_ = op;
      master_attrs_ = call_node->attrs;
      master_op_pattern_ = op_pattern;
    }
    if (outputs.size() != 1) {
      const auto* tuple_type =
          call_node->checked_type().as<TupleTypeNode>();
      CHECK(tuple_type) << "Expect output to be a tuple type";
      CHECK_EQ(tuple_type->fields.size(), outputs.size());
    }
    // Set the name to `__copy`. It will be detected in graph runtime to perform
    // data copy across devices.
    if (is_copy_op) {
      readable_name_stream_.str(std::string());
      readable_name_stream_ << "__copy";
    } else {
      readable_name_stream_ << '_' << op->name;
    }
    return outputs;
  }

  Array<Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Do not support sub function";
    return Array<Tensor>();
  }

  Array<Tensor> VisitExpr_(const LetNode* op) final {
    Array<Tensor> val = VisitExpr(op->value);
    CHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<Tensor> VisitExpr_(const TupleNode* op) final {
    Array<Tensor> fields;
    for (Expr field : op->fields) {
      CHECK(field->checked_type().as<TensorTypeNode>())
          << "Only allow Tuple of Tensor";
      Array<Tensor> res = VisitExpr(field);
      CHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    const auto* tuple_type = op->tuple->type_as<TupleTypeNode>();
    Array<Tensor> tuple = VisitExpr(op->tuple);
    CHECK_EQ(tuple_type->fields.size(), tuple.size());
    CHECK_GE(op->index, 0);
    CHECK_LT(static_cast<size_t>(op->index), tuple.size());
    return {tuple[op->index]};
  }

 private:
  tvm::Target target_;
  Op master_op_;
  Attrs master_attrs_;
  int master_op_pattern_{0};
  std::ostringstream readable_name_stream_;
  std::unordered_map<Expr, Array<Tensor>, NodeHash, NodeEqual> memo_;
  Array<Operation> scalars_;
};


class CompileEngineImpl : public CompileEngineNode {
 public:
  // Lower the function.
  CachedFunc Lower(const CCacheKey& key)  {
    return LowerInternal(key)->cached_func;
  }

  // For now, build one module per function.
  PackedFunc JIT(const CCacheKey& key) final {
    CCacheValue value = LowerInternal(key);
    if (value->packed_func != nullptr) return value->packed_func;
    // build the function.
    if (const auto* f = runtime::Registry::Get("relay.backend.build")) {
      tvm::runtime::Module m = (*f)(value->cached_func->funcs, key->target);
      value->packed_func = m.GetFunction(value->cached_func->func_name);
    } else {
      LOG(FATAL) << "relay.backend.build is not registered";
    }
    return value->packed_func;
  }
  void Clear() final {
    cache_.clear();
  }
  // List all items in the cache.
  Array<NodeRef> ListItems() {
    std::lock_guard<std::mutex> lock(mutex_);
    Array<NodeRef> items;
    for (auto& kv : cache_) {
      items.push_back(kv.first);
      items.push_back(kv.second);
    }
    return items;
  }
  /*!
   * \brief Create schedule for target.
   * \param source_func The primitive function to be lowered.
   * \param target The target we want to create schedule for.
   * \return Pair of schedule and cache.
   *  The funcs field in cache is not yet populated.
   */
  std::pair<Schedule, CachedFunc> CreateSchedule(
      const Function& source_func, const Target& target) {
    return ScheduleGetter(target).Create(source_func);
  }

 private:
  // implement lowered func
  CCacheValue LowerInternal(const CCacheKey& key)  {
    std::lock_guard<std::mutex> lock(mutex_);
    CCacheValue value;
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second->use_count += 1;
      if (it->second->cached_func.defined()) return it->second;
      value = it->second;
    } else {
      value = CCacheValue(make_node<CCacheValueNode>());
      value->use_count = 0;
      cache_[key] = value;
    }
    // Enforce use the target.
    With<Target> target_scope(key->target);

    CHECK(!value->cached_func.defined());
    auto spair = CreateSchedule(key->source_func, key->target);
    auto cache_node = make_node<CachedFuncNode>(
        *(spair.second.operator->()));

    // Skip lowering for device copy node.
    const Expr body = (key->source_func)->body;
    if (const CallNode* call_node = body.as<CallNode>()) {
      if (call_node->attrs.as<DeviceCopyAttrs>()) {
        value->cached_func = CachedFunc(cache_node);
        return value;
      }
    }

    cache_node->func_name = GetUniqueName(cache_node->func_name);
    // NOTE: array will copy on write.
    Array<Tensor> all_args = cache_node->inputs;
    for (Tensor arg : cache_node->outputs) {
      all_args.push_back(arg);
    }
    // lower the function
    if (const auto* f = runtime::Registry::Get("relay.backend.lower")) {
      cache_node->funcs = (*f)(
          spair.first, all_args, cache_node->func_name, key->source_func);
    } else {
      tvm::BuildConfig bcfg = BuildConfig::Create();
      std::unordered_map<Tensor, Buffer> binds;
      cache_node->funcs = tvm::lower(spair.first, all_args, cache_node->func_name, binds, bcfg);
    }
    value->cached_func = CachedFunc(cache_node);
    return value;
  }
  /*!
   * \brief Get unique name from name.
   * \param name The orginal name.
   * \return Updated name which is unique.
   */
  std::string GetUniqueName(std::string name) {
    for (size_t i = 0; i < name.length(); ++i) {
      if (name[i] == '.') name[i] = '_';
    }
    while (true) {
      auto it = name_map_.find(name);
      if (it == name_map_.end()) {
        name_map_[name] = 1;
        return name;
      } else {
        std::ostringstream os;
        os << name << "_" << it->second;
        ++(it->second);
        name = os.str();
      }
    }
    return name;
  }
  /*! \brief compiler cache lock*/
  std::mutex mutex_;
  /*! \brief internal name map to get an unique name */
  std::unordered_map<std::string, int> name_map_;
  /*! \brief internal compiler cache */
  std::unordered_map<CCacheKey, CCacheValue> cache_;
};

/*! \brief The global compile engine */
const CompileEngine& CompileEngine::Global() {
  // intentionally allocate raw pointer to avoid
  // free during destructuion.
  static CompileEngine* inst = new CompileEngine(
      make_node<CompileEngineImpl>());
  return *inst;
}


TVM_REGISTER_GLOBAL("relay.backend._make_CCacheKey")
.set_body_typed<CCacheKey(Function, Target)>(CCacheKeyNode::make);

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineGlobal")
.set_body_typed<CompileEngine()>([]() {
    return CompileEngine::Global();
  });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineClear")
.set_body_typed<void(const CompileEngine&)>([](CompileEngine self) {
    self->Clear();
  });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineLower")
.set_body_typed<CachedFunc(CompileEngine, CCacheKey)>(
    [](CompileEngine self, CCacheKey key) {
      return self->Lower(key);
    });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineJIT")
.set_body_typed<PackedFunc(CompileEngine, CCacheKey)>(
    [](CompileEngine self, CCacheKey key) {
      return self->JIT(key);
    });

TVM_REGISTER_GLOBAL("relay.backend._CompileEngineListItems")
.set_body_typed<Array<NodeRef>(CompileEngine)>(
    [](CompileEngine self){
      return static_cast<CompileEngineImpl*>(self.operator->())->ListItems();
    });
}  // namespace relay
}  // namespace tvm
