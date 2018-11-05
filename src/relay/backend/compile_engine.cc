/*!
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/compile_engine.cc
 * \brief Internal compialtion engine.
 */
#include <tvm/schedule.h>
#include <tvm/packed_func_ext.h>
#include <tvm/operation.h>
#include <tvm/runtime/registry.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <utility>
#include <limits>
#include <mutex>
#include <functional>
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

    if (prim_func->params.size() == 1 &&
        prim_func->params[0]->checked_type().as<TupleTypeNode>()) {
      // Handle tuple input type by flattening them.
      // This is the current calling convention of tuple input.
      Array<tvm::Tensor> inputs;
      for (Type field : prim_func->params[0]->type_as<TupleTypeNode>()->fields) {
        const auto* ttype = field.as<TensorTypeNode>();
        CHECK(ttype != nullptr);
        tvm::Tensor tensor = tvm::placeholder(
            GetShape(ttype->shape), ttype->dtype);
        cache_node->inputs.push_back(tensor);
        inputs.push_back(tensor);
      }
      memo_[prim_func->params[0]] = inputs;

    } else {
      for (Var param : prim_func->params) {
        const auto* ttype = param->type_as<TensorTypeNode>();
        tvm::Tensor tensor = tvm::placeholder(
            GetShape(ttype->shape), ttype->dtype);
        cache_node->inputs.push_back(tensor);
        memo_[param] = Array<Tensor>({tensor});
      }
    }
    readable_name_stream_ << "fused";
    // enter the target context
    TargetContext target_ctx(target_);
    cache_node->outputs = this->VisitExpr(prim_func->body);
    cache_node->func_name = readable_name_stream_.str();
    CachedFunc cfunc(cache_node);
    CHECK(master_op_.defined());
    Schedule schedule = fschedule[master_op_](
        cache_node->outputs, target_);
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
    LOG(FATAL) << "Free variable " << op->name_hint;
    return {};
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
    Array<Tensor> outputs = fcompute[op](
        call_node->attrs,
        inputs,
        call_node->checked_type(),
        target_);

    int op_pattern = fpattern[op];
    if (op_pattern >= kCommReduce) {
      CHECK(!master_op_.defined())
          << "Two complicated op in a primitive function";
    }
    if (op_pattern >= master_op_patetrn_) {
      master_op_ = op;
      master_op_patetrn_ = op_pattern;
    }
    if (outputs.size() != 1) {
      const auto* tuple_type =
          call_node->checked_type().as<TupleTypeNode>();
      CHECK(tuple_type) << "Expect output to be a tuple type";
      CHECK_EQ(tuple_type->fields.size(), outputs.size());
    }
    readable_name_stream_ << '_' << op->name;
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
  int master_op_patetrn_{0};
  std::ostringstream readable_name_stream_;
  std::unordered_map<Expr, Array<Tensor>, NodeHash, NodeEqual> memo_;
};


class CompileEngineImpl : public CompileEngineNode {
 public:
  // Lower the fucntion.
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
    CHECK(!value->cached_func.defined());
    auto spair = CreateSchedule(key->source_func, key->target);
    auto cache_node = make_node<CachedFuncNode>(
        *(spair.second.operator->()));
    cache_node->func_name = GetUniqeName(cache_node->func_name);
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
      LOG(FATAL) << "relay.backend._lower is not registred";
    }
    value->cached_func = CachedFunc(cache_node);
    return value;
  }
  /*!
   * \brief Get unique name from name.
   * \param name The orginal name.
   * \return Updated name which is unique.
   */
  std::string GetUniqeName(std::string name) {
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
