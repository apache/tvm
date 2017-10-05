/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Compute Op.
 * \file compute_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "./compute_op.h"
#include "./op_util.h"
#include "../schedule/message_passing.h"

namespace tvm {

using namespace ir;

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ComputeOpNode>([](const ComputeOpNode *op, IRPrinter *p) {
    p->stream << "compute(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

inline bool ReduceEqual(const ir::Reduce* a, const ir::Reduce* b) {
  return (a->combiner.same_as(b->combiner)) &&
         (a->source.same_as(b->source)) &&
         (a->axis.same_as(b->axis)) &&
         (a->condition.same_as(b->condition));
}

int ComputeOpNode::num_outputs() const {
  return body.size();
}

Array<IterVar> ComputeOpNode::root_iter_vars() const {
  if (reduce_axis.size() == 0) return axis;
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

Type ComputeOpNode::output_dtype(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  return body[idx].type();
}

Array<Expr> ComputeOpNode::output_shape(size_t idx) const {
  CHECK_LT(idx, num_outputs());
  // for now, all outputs of ComputeOp have the same shape
  std::vector<Expr> shape;
  for (size_t i = 0; i < axis.size(); ++i) {
    const Range& r = axis[i]->dom;
    shape.push_back(r->extent);
  }
  return Array<Expr>(shape);
}

Tensor compute(Array<Expr> shape,
               FCompute fcompute,
               std::string name,
               std::string tag) {
  auto op_node = std::make_shared<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVarNode::make(
        Range(0, shape[i]), Var(os.str(), shape[i].type()), kDataPar));
    args.push_back(axis.back()->var);
  }

  return ComputeOpNode::make(name, tag, axis, {fcompute(args)}).output(0);
}

Array<Tensor> compute(Array<Expr> shape,
                      FBatchCompute fcompute,
                      std::string name,
                      std::string tag) {
  auto op_node = std::make_shared<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVarNode::make(
        Range(0, shape[i]), Var(os.str(), shape[i].type()), kDataPar));
    args.push_back(axis.back()->var);
  }

  Operation op = ComputeOpNode::make(name, tag, axis, fcompute(args));
  Array<Tensor> outputs;
  for (int idx = 0; idx < op->num_outputs(); ++idx) {
    outputs.push_back(op.output(idx));
  }
  return outputs;
}

Operation ComputeOpNode::make(std::string name,
                              std::string tag,
                              Array<IterVar> axis,
                              Array<Expr> body) {
  auto n = std::make_shared<ComputeOpNode>();
  n->name = name;
  n->tag = tag;
  n->axis = axis;
  n->body = body;
  if (n->body[0]->is_type<ir::Reduce>()) {
    const ir::Reduce* reduce = n->body[0].as<ir::Reduce>();
    for (size_t i = 1; i < n->body.size(); ++i) {
      const ir::Reduce* reduce_ = n->body[i].as<ir::Reduce>();
      CHECK(reduce_);
      CHECK(ReduceEqual(reduce_, reduce))
        << "The Reduce inputs of ComputeOp should "
        << "have the same attribute except value_index";
    }
    n->reduce_axis = reduce->axis;
  }
  return Operation(n);
}

// The schedule related logics
Array<Tensor> ComputeOpNode::InputTensors() const {
  Array<Tensor> ret;
  std::unordered_set<Tensor> visited;
  for (auto& e : body) {
    ir::PostOrderVisit(e, [&ret, &visited](const NodeRef& n) {
        const ir::Call *call = n.as<ir::Call>();
        if (call != nullptr && call->func.defined()) {
          Tensor t = Operation(call->func.node_).output(call->value_index);
          if (!visited.count(t)) {
            ret.push_back(t);
            visited.insert(t);
          }
        }
      });
  }
  return ret;
}

Operation ComputeOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  Array<Expr> arr;
  if (this->body[0]->is_type<ir::Reduce>()) {
    // Specially handle reduce so the replaced op
    // still share all the components
    const ir::Reduce* reduce = this->body[0].as<ir::Reduce>();
    for (size_t i = 1; i < this->body.size(); ++i) {
      const ir::Reduce* reduce_ = this->body[i].as<ir::Reduce>();
      CHECK(reduce_);
      CHECK(ReduceEqual(reduce_, reduce))
        << "The Reduce inputs of ComputeOp should "
        << "have the same attribute except value_index";
    }\
    Expr new_reduce = op::ReplaceTensor(this->body[0], rmap);
    if (!new_reduce.same_as(this->body[0])) {
      const ir::Reduce* r = new_reduce.as<ir::Reduce>();
      for (size_t k = 0; k < this->body.size(); ++k) {
        std::shared_ptr<ir::Reduce> n = std::make_shared<ir::Reduce>(*r);
        n->value_index = static_cast<int>(k);
        n->type = r->source[k].type();
        arr.push_back(Expr(n));
      }
    } else {
      arr = this->body;
    }
  } else {
    arr = UpdateArray(this->body, [&rmap] (const Expr& e) {
        return op::ReplaceTensor(e, rmap);
      });
  }
  if (!arr.same_as(this->body)) {
    return ComputeOpNode::make(name, tag, axis, arr);
  } else {
    return self;
  }
}

void ComputeOpNode::PropBoundToInputs(
    const Operation& self,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  CHECK_EQ(self.operator->(), this);
  auto fvisit = [&dom_map, out_dom_map](const NodeRef& n) {
    auto *call = n.as<ir::Call>();
    if (call != nullptr && call->func.defined()) {
      Tensor t = Operation(call->func.node_).output(call->value_index);
      if (t->op.defined() && out_dom_map->count(t)) {
        TensorDom& dom = out_dom_map->at(t);
        for (size_t i = 0; i < t.ndim(); ++i) {
          dom.data[i].push_back(EvalSet(call->args[i], dom_map));
        }
      }
    }
  };
  for (auto& e : body) ir::PostOrderVisit(e, fvisit);
}

void ComputeOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  const TensorDom& tdom = tensor_dom.at(self.output(0));
  for (size_t i = 0; i < this->axis.size(); ++i) {
    Range r = arith::Union(tdom.data.at(i)).cover_range(this->axis[i]->dom);
    CHECK(!out_dom_map->count(this->axis[i]));
    (*out_dom_map)[this->axis[i]] = r;
  }
  for (size_t i = 0; i < this->reduce_axis.size(); ++i) {
    CHECK(!out_dom_map->count(this->reduce_axis[i]));
    (*out_dom_map)[this->reduce_axis[i]] = this->reduce_axis[i]->dom;
  }
}

Stmt ComputeOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& realize_map,
    const Stmt& realize_body) const {
  CHECK_EQ(stage->op.get(), this);
  Halide::Internal::Region bounds;
  for (IterVar iv : this->axis) {
    bounds.push_back(realize_map.at(iv));
  }
  Stmt realize = realize_body;
  for (int i = this->num_outputs(); i > 0; --i) {
    Tensor t = stage->op.output(i-1);
    realize = ir::Realize::make(t->op, t->value_index,
      t->dtype, bounds, const_true(), realize);
    // alignment requirement, only useful for compute
    for (size_t i = 0; i < this->axis.size(); ++i) {
      auto it = stage->iter_var_attrs.find(this->axis[i]);
      if (it != stage->iter_var_attrs.end()) {
        IterVarAttr attr = (*it).second;
        if (attr->dim_align_factor != 0) {
          Array<Expr> tuple = {static_cast<int>(i),
                               attr->dim_align_factor,
                               attr->dim_align_offset};
          realize = ir::AttrStmt::make(
              t, ir::attr::buffer_dim_align,
              Call::make(Handle(), ir::intrinsic::tvm_tuple, tuple, Call::Intrinsic),
              realize);
        }
      }
    }
  }
  return realize;
}

// Build a reduction body.
void MakeReduction(const ComputeOpNode* op,
                   const Array<Tensor>& tensors,
                   Stmt* init,
                   Stmt* provide) {
  Array<Expr>  args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  std::vector<Stmt> inits, provides;

  size_t size = op->body.size();
  const Reduce* reduce = op->body[0].as<Reduce>();
  CHECK(reduce);
  const CommReducerNode* combiner = reduce->combiner.as<CommReducerNode>();
  CHECK(combiner);
  Array<Expr> lhs;
  for (size_t i = 0; i < size; ++i) {
    lhs.push_back(tensors[i](args));
  }
  Array<Expr> init_value = combiner->identity_element;
  Array<Expr> update_value = (*combiner)(lhs, reduce->source);
  for (size_t i = 0; i < size; ++i) {
    Tensor t = tensors[i];
    inits.emplace_back(Provide::make(
          t->op, t->value_index, init_value[i], args));
    provides.emplace_back(Provide::make(
          t->op, t->value_index, update_value[i], args));
  }
  *init = Block::make(inits);
  *provide = Block::make(provides);
  if (!is_one(reduce->condition)) {
    *provide = IfThenElse::make(reduce->condition, *provide);
  }
}

// Normal computation.
Stmt MakeProvide(const ComputeOpNode* op,
                 const Tensor& t) {
  Array<Expr> args;
  for (IterVar iv : op->axis) {
    args.push_back(iv->var);
  }
  return Provide::make(t->op, t->value_index, op->body[t->value_index], args);
}

Stmt MakeComputeStmt(const ComputeOpNode* self,
                     const Stage& stage,
                     const std::unordered_map<IterVar, Range>& dom_map) {
  // grab the nest structure
  ComputeLoopNest n = ComputeLoopNest::make(self, stage, dom_map);
  // Normal loop structure
  n.init_nest.emplace_back(op::MakeIfNest(n.init_predicates));
  n.main_nest.emplace_back(op::MakeIfNest(n.main_predicates));
  if (self->reduce_axis.size() != 0) {
    // make reduction.
    Stmt init, provide;
    Array<Tensor> source;
    for (size_t i = 0; i < self->body.size(); ++i) {
      source.push_back(stage->op.output(i));
    }
    MakeReduction(self, source, &init, &provide);
    init = op::Substitute(init, n.init_vmap);
    init = MergeNest(n.init_nest, init);
    // common nest
    std::vector<std::vector<Stmt> > common(
        n.main_nest.begin(), n.main_nest.begin() + n.num_common_loop + 1);
    std::vector<std::vector<Stmt> > reduce(
        n.main_nest.begin() + n.num_common_loop + 1, n.main_nest.end());
    provide = op::Substitute(provide, n.main_vmap);
    provide = MergeNest(reduce, provide);
    return MergeNest(common, Block::make(init, provide));
  } else {
    std::vector<Stmt> provides;
    for (size_t i = 0; i < self->body.size(); ++i) {
      provides.emplace_back(MakeProvide(self, stage->op.output(i)));
    }
    Stmt provide = op::Substitute(Block::make(provides), n.main_vmap);
    return MergeNest(n.main_nest, provide);
  }
}

enum class ComputeType {
  kNormal,
  kCrossThreadReduction,
  kTensorize
};

ComputeType DetectComputeType(const ComputeOpNode* self,
                              const Stage& stage) {
  // Verify correctness of leaf nest.
  int normal_red = 0, thread_red = 0, tensorize = 0;

  for (IterVar iv : stage->leaf_iter_vars) {
    IterVarAttr attr;
    auto it = stage->iter_var_attrs.find(iv);
    if (it != stage->iter_var_attrs.end()) {
      attr = (*it).second;
    }
    if (attr.defined() && attr->iter_type == kTensorized) {
      ++tensorize;
    }
    if (iv->iter_type == kCommReduce) {
      if (attr.defined() && attr->bind_thread.defined()) {
        ++thread_red;
      } else {
        ++normal_red;
      }
    } else {
      CHECK_EQ(thread_red, 0)
          << "Cross thread reduce cannot swap with normal data axis";
    }
  }
  if (tensorize != 0) {
    CHECK(thread_red == 0)
        << "Cannot mix cross thread reduction with Tensorize";
    return ComputeType::kTensorize;
  }
  CHECK(normal_red == 0 || thread_red == 0)
      << "Cannot mix normal reduction with thread reduce";
  if (thread_red != 0) {
    return ComputeType::kCrossThreadReduction;
  } else {
    return ComputeType::kNormal;
  }
}

// implement the provide utility.
Stmt ComputeOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map) const {
  CHECK_EQ(stage->op.operator->(), this);
  ComputeType ctype = DetectComputeType(this, stage);
  if (ctype == ComputeType::kCrossThreadReduction) {
    // specially handle cross thread reduction.
    return MakeCrossThreadReduction(this, stage, dom_map);
  } else if (ctype == ComputeType::kTensorize) {
    return MakeTensorize(this, stage, dom_map);
  } else {
    return MakeComputeStmt(this, stage, dom_map);
  }
}

ComputeLoopNest ComputeLoopNest::make(
    const ComputeOpNode* self,
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map) {
  CHECK_EQ(stage->op.operator->(), self);
  ComputeLoopNest ret;
  // make main loop nest
  ret.main_nest = op::MakeLoopNest(
      stage, dom_map, 0, false, std::unordered_set<IterVar>(), &ret.main_vmap);
  ret.main_predicates = schedule::MakeBoundCheck(
      stage, dom_map, ret.main_vmap, false,
      std::unordered_set<IterVar>());
  for (auto& e : ret.main_predicates) {
    e = likely(e);
  }
  if (stage->store_predicate.defined()) {
    ret.main_predicates.push_back(stage->store_predicate);
  }
  if (self->reduce_axis.size() != 0) {
    // try to find the location to insert the initialization.
    // Fuse the initialization and provide loop when possible.
    std::unordered_map<IterVar, int> update_state;
    for (IterVar iv : self->reduce_axis) {
      update_state[iv] = 2;
    }
    for (IterVar iv : self->axis) {
      update_state[iv] = 1;
    }
    // find which iter var is related to reduction and which is related to axis.
    schedule::PassDownBitMaskOr(stage, &update_state);
    auto leaf_iter_vars = stage->leaf_iter_vars;
    // first first loop that is related to reduction.
    size_t begin_loop = leaf_iter_vars.size();
    for (size_t i = 0; i < leaf_iter_vars.size(); ++i) {
      auto iv = leaf_iter_vars[i];
      int flag = update_state.at(iv);
      if ((flag & 2) != 0) {
        begin_loop = i; break;
      }
      ret.init_vmap[iv] = ret.main_vmap.at(iv);
    }
    ret.num_common_loop = begin_loop;
    // skip loops that does not relates to axis.
    std::unordered_set<IterVar> skip_iter;
    for (auto kv : update_state) {
      int flag = kv.second;
      if ((flag & 1) == 0) skip_iter.insert(kv.first);
    }
    ret.init_nest = op::MakeLoopNest(
        stage, dom_map, begin_loop, true,
        skip_iter, &(ret.init_vmap));
    ret.init_predicates = schedule::MakeBoundCheck(
        stage, dom_map, ret.init_vmap, true, skip_iter);
    for (auto& e : ret.init_predicates) {
      e = likely(e);
    }
  } else {
    CHECK_EQ(ret.main_nest.size(), stage->leaf_iter_vars.size() + 1);
    ret.num_common_loop = stage->leaf_iter_vars.size();
  }
  // copy elison here.
  return ret;
}
}  // namespace tvm
