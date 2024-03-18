/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file layout/layout.cc
 * \brief Define Layout used in MMA and other operations.
 *
 */

#include "layout.h"

#include <tvm/arith/pattern.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../arith/pattern_match.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

static Var getPlaceholder(const std::string& s) {
  static std::unordered_map<std::string, Var> map;
  if (map.find(s) == map.end()) {
    map[s] = Var(s);
  }
  return map[s];
}

static Var ReplicationPlaceholder() { return getPlaceholder("_rep"); }
static Var InputPlaceholder(size_t idx) {
  return getPlaceholder(std::string{'_', char('i' + idx)});
}

Map<Var, Range> LayoutNode::getVarMap() const {
  Map<Var, Range> map;
  for (size_t i = 0; i < InputDim(); i++) {
    map.Set(InputPlaceholder(i), {0, input_size_[i]});
  }
  return map;
}

Map<Var, Range> FragmentNode::getVarMap() const {
  auto map = LayoutNode::getVarMap();
  map.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  return map;
}

IterVar make_itervar(std::string name, PrimExpr dom) {
  Var var = Var(name);
  return IterVar(Range(0, dom), var, IterVarType::kDataPar);
}

LayoutNode::LayoutNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index) {
  input_size_ = input_size;
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  forward_index_ = forward_index.Map([&](const PrimExpr& e) { return analyzer.Simplify(e); });
}

Layout::Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index) {
  Map<Var, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    CHECK(is_zero(forward_var[i]->dom->min));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  forward_index = forward_index.Map([&](const PrimExpr& e) { return Substitute(e, vmap); });

  auto n = make_object<LayoutNode>(input_size, forward_index);
  data_ = std::move(n);
}

Layout::Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index) {
  auto n = make_object<LayoutNode>(input_size, forward_index);
  data_ = std::move(n);
}

void LayoutNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("input_size", &input_size_);
  v->Visit("forward_index", &forward_index_);
}

void LayoutNode::UpdateAnalyzer(arith::Analyzer* analyzer) const {
  for (const auto& [var, dom] : getVarMap()) {
    analyzer->Bind(var, dom);
  }
}

Array<PrimExpr> LayoutNode::OutputShape() const {
  Array<PrimExpr> ret(OutputDim(), 1);
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  for (size_t i = 0; i < ret.size(); i++) {
    auto ist = analyzer.int_set(forward_index_[i] + 1);
    CHECK(is_one(ist.min())) << ist.min();
    ret.Set(i, ist.max());
  }
  return ret;
}

Array<PrimExpr> LayoutNode::Forward(const Array<PrimExpr>& vars) const {
  if (vars.empty()) return forward_index_;
  ICHECK_EQ(vars.size(), InputDim());
  Map<Var, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(InputPlaceholder(i), vars[i]);
  }
  return forward_index_.Map([&](const PrimExpr& e) { return Substitute(e, vmap); });
}

Fragment FragmentNode::Repeat(const Array<PrimExpr>& repeats, bool repeat_on_thread,
                              bool lower_dim_first) const {
  ICHECK_EQ(repeats.size(), InputDim());
  Array<PrimExpr> new_input_size;
  Map<Var, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    new_input_size.push_back(input_size_[i] * repeats[i]);
    vmap.Set(InputPlaceholder(i), FloorMod(InputPlaceholder(i), InputShape()[i]));
  }

  PrimExpr repeats_index = 0, repeat_stride = 1;
  if (lower_dim_first) {
    for (int i = InputDim() - 1; i >= 0; i--) {
      repeats_index += repeat_stride * FloorDiv(InputPlaceholder(i), InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  } else {
    for (size_t i = 0; i < InputDim(); i++) {
      repeats_index += repeat_stride * FloorDiv(InputPlaceholder(i), InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  }

  if (repeat_on_thread) {
    PrimExpr thread_size = ThreadExtent();
    auto new_forward_index =
        forward_index_.Map([&](const PrimExpr& e) { return Substitute(e, vmap); });
    auto new_forward_thread = Substitute(forward_thread_, vmap) + thread_size * repeats_index;
    return Fragment(new_input_size, new_forward_index, new_forward_thread, replicate_size_,
                    std::nullopt);
  } else {
    ICHECK(OutputDim() == 1);
    PrimExpr frag_len = OutputShape()[0];
    Array<PrimExpr> new_forward_index = {Substitute(forward_index_[0], vmap) +
                                         frag_len * repeats_index};
    PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
    return Fragment(new_input_size, new_forward_index, new_forward_thread, replicate_size_,
                    std::nullopt);
  }
}

Fragment FragmentNode::Replicate(int repeats) const {
  ICHECK(repeats >= 1);
  Map<Var, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(), FloorMod(ReplicationPlaceholder(), ReplicateExtent()));
  PrimExpr new_forward_thread =
      Substitute(forward_thread_, vmap) +
      ThreadExtent() * FloorDiv(ReplicationPlaceholder(), ReplicateExtent());
  return Fragment(input_size_, forward_index_, new_forward_thread, ReplicateExtent() * repeats,
                  std::nullopt);
}

Fragment FragmentNode::DeReplicate() const {
  ICHECK(OutputDim() == 1);
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  int factor = 1;
  auto rep_size = as_const_int(ReplicateExtent());
  auto idx_size = as_const_int(OutputShape()[0]);
  if (rep_size && idx_size) {
    factor = arith::ZeroAwareGCD(*rep_size, *idx_size);
  }
  if (factor == 1) return GetRef<Fragment>(this);

  Map<Var, PrimExpr> vmap;
  vmap.Set(ReplicationPlaceholder(),
           ReplicationPlaceholder() * factor + FloorMod(forward_index_[0], factor));
  PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
  Array<PrimExpr> new_forward_index = {FloorDiv(forward_index_[0], factor)};
  return Fragment(input_size_, new_forward_index, new_forward_thread, int(*rep_size) / factor,
                  std::nullopt);
}

Layout LayoutNode::Inverse() const {
  arith::Analyzer analyzer;
  arith::IterMapResult res = arith::DetectIterMap(forward_index_, getVarMap(), 1,
                                                  arith::IterMapLevel::Bijective, &analyzer);
  ICHECK(res->errors.empty()) << res->errors;

  auto outputs_shape = OutputShape();
  Array<PrimExpr> outputs;
  for (size_t i = 0; i < OutputDim(); i++) {
    outputs.push_back(InputPlaceholder(i));
  }

  auto inv = arith::InverseAffineIterMap(res->indices, outputs);

  Array<PrimExpr> backward_index;
  for (size_t i = 0; i < InputDim(); i++) {
    if (inv.find(InputPlaceholder(i)) != inv.end()) {
      backward_index.push_back(inv[InputPlaceholder(i)]);
    } else {
      backward_index.push_back(0);
    }
  }

  return Layout(outputs_shape, backward_index);
}

int LayoutNode::VectorSize() const {
  auto last_dim = OutputShape().back().as<IntImm>();
  if (!last_dim.defined()) return 1;
  int vector_size = 2;
  PrimExpr last_index = forward_index_.back();
  Var last_var = InputPlaceholder(InputDim() - 1);
  arith::Analyzer analyzer;
  auto iter_sum = arith::NormalizeToIterSum(last_index, getVarMap(), &analyzer);
  while ((last_dim.value()->value % vector_size) == 0) {
    bool can_vector_load = true;

    for (const auto& split : iter_sum->args) {
      int scale = split->scale.as<IntImm>().value()->value;
      int lower_factor = split->lower_factor.as<IntImm>().value()->value;
      if (split->source->source.same_as(last_var) && lower_factor < vector_size) {
        if (lower_factor != scale) {
          can_vector_load = false;
          break;
        }
      } else {
        int scale = split->scale.as<IntImm>().value()->value;
        if ((scale % vector_size) != 0) {
          can_vector_load = false;
          break;
        }
      }
    }
    if (!can_vector_load) break;
    vector_size *= 2;
  }
  return vector_size / 2;
}

PrimExpr infer_fragment_index(const Map<Var, Range>& input_iters, const PrimExpr& forward_thread,
                              arith::Analyzer* analyzer) {
  Array<arith::IterSplitExpr> splits =
      DivideUnusedIterators({forward_thread}, ToIterVars(input_iters), analyzer);

  Array<arith::IterSplitExpr> split_without_rep;
  for (const auto& split : splits) {
    CHECK(split->source->source.as<Var>());
    if (split->source->source.as<Var>().value().same_as(ReplicationPlaceholder())) continue;
    split_without_rep.push_back(split);
  }
  return MakeFlattenedExpression(split_without_rep);
}

FragmentNode::FragmentNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                           PrimExpr forward_thread, PrimExpr replicate_size) {
  input_size_ = input_size;
  replicate_size_ = replicate_size;
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  forward_thread_ = analyzer.Simplify(forward_thread);
  if (forward_index.empty()) {
    forward_index = {infer_fragment_index(getVarMap(), forward_thread_, &analyzer)};
  }
  forward_index_ = forward_index.Map([&](const PrimExpr& e) { return analyzer.Simplify(e); });
}

Fragment::Fragment(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, IterVar thread_replicate) {
  Map<Var, PrimExpr> vmap;
  Array<PrimExpr> input_size;
  PrimExpr replicate_size = 1;
  for (size_t i = 0; i < forward_var.size(); i++) {
    vmap.Set(forward_var[i]->var, InputPlaceholder(i));
    CHECK(is_zero(forward_var[i]->dom->min));
    input_size.push_back(forward_var[i]->dom->extent);
  }
  if (thread_replicate.defined()) {
    ICHECK(is_zero(thread_replicate->dom->min));
    replicate_size = thread_replicate->dom->extent;
    vmap.Set(thread_replicate->var, ReplicationPlaceholder());
  }
  forward_index = forward_index.Map([&](const PrimExpr& e) { return Substitute(e, vmap); });
  forward_thread = Substitute(forward_thread, vmap);

  auto n = make_object<FragmentNode>(input_size, forward_index, forward_thread, replicate_size);
  data_ = std::move(n);
}

Fragment::Fragment(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, PrimExpr replicate_size,
                   std::optional<Var> replicate_var) {
  if (replicate_var.has_value()) {
    forward_thread =
        Substitute(forward_thread, {{replicate_var.value(), ReplicationPlaceholder()}});
  }
  auto n = make_object<FragmentNode>(input_size, forward_index, forward_thread, replicate_size);
  data_ = std::move(n);
}

void FragmentNode::VisitAttrs(tvm::AttrVisitor* v) {
  LayoutNode::VisitAttrs(v);
  v->Visit("forward_thread", &forward_thread_);
  v->Visit("replicate_size", &replicate_size_);
}

PrimExpr FragmentNode::ThreadExtent() const {
  Array<PrimExpr> ret(OutputDim(), 1);
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  auto ist = analyzer.int_set(forward_thread_ + 1);
  CHECK(is_one(ist.min()));
  return ist.max();
}

PrimExpr FragmentNode::ForwardThread(const Array<PrimExpr>& vars,
                                     const Optional<PrimExpr>& rep_var) const {
  Map<Var, PrimExpr> vmap;
  ICHECK_EQ(vars.size(), InputDim());
  for (size_t i = 0; i < InputDim(); i++) {
    vmap.Set(InputPlaceholder(i), vars[i]);
  }
  if (rep_var.defined()) vmap.Set(ReplicationPlaceholder(), rep_var.value());

  return Substitute(forward_thread_, vmap);
}

Layout FragmentNode::Inverse() const {
  auto input_size_copy = input_size_;
  input_size_copy.push_back(ReplicateExtent());
  auto forward_index_copy = forward_index_;
  forward_index_copy.push_back(
      Substitute(forward_thread_, {{ReplicationPlaceholder(), InputPlaceholder(InputDim())}}));
  auto fwd = Layout(input_size_copy, forward_index_copy);
  auto bwd = fwd->Inverse();
  return bwd;
}

Fragment FragmentNode::CondenseReplicateVar() const {
  arith::Analyzer analyzer;
  auto input_iters = getVarMap();
  input_iters.Set(ReplicationPlaceholder(), {0, ReplicateExtent()});
  PrimExpr new_forward_thread;
  IterVar new_thread_replicate;
  std::tie(new_forward_thread, new_thread_replicate) = CompressIterator(
      forward_thread_, ToIterVars(input_iters), ReplicationPlaceholder(), &analyzer);
  return Fragment(input_size_, forward_index_, new_forward_thread,
                  new_thread_replicate->dom->extent, new_thread_replicate->var);
}

void LayoutNode::DebugOutput() const {
  LOG_DEBUG << "Layout Shape: " << InputShape() << " -> " << OutputShape();
  LOG_DEBUG << "Layout Index: " << forward_index_;
}

void FragmentNode::DebugOutput() const {
  LOG_DEBUG << "Fragment Shape: " << InputShape() << " -> " << OutputShape();
  LOG_DEBUG << "Fragment Replicate: " << ReplicateExtent();
  LOG_DEBUG << "Fragment ThreadExtent: " << ThreadExtent();
  LOG_DEBUG << "Fragment Index: " << forward_index_;
  LOG_DEBUG << "Fragment ThreadIndex: " << forward_thread_;
}

bool LayoutNode::SEqualReduce(const LayoutNode* other, SEqualReducer equal) const {
  return this->InputDim() == other->InputDim() && equal(this->InputShape(), other->InputShape()) &&
         equal(this->forward_index_, other->forward_index_);
}

bool FragmentNode::SEqualReduce(const FragmentNode* other, SEqualReducer equal) const {
  return this->InputDim() == other->InputDim() &&
         equal(this->ReplicateExtent(), other->ReplicateExtent()) &&
         equal(this->InputShape(), other->InputShape()) &&
         equal(this->ThreadExtent(), other->ThreadExtent()) &&
         equal(this->forward_index_, other->forward_index_) &&
         equal(this->forward_thread_, other->forward_thread_);
}

Fragment makeGemmFragment8x8() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 2) + 4 * i;
  PrimExpr index = FloorMod(j->var, 2);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragment8x8Transposed() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(i->var, 2) + 4 * j;
  PrimExpr index = FloorMod(i->var, 2);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentC_F64(const int block_m, const int block_n, const int warp_m,
                               const int warp_n) {
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(warp_n % 16 == 0);
  auto base_layout = makeGemmFragment8x8();
  auto warp_layout = base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  auto block_layout = warp_layout->Repeat({warp_m / 8, warp_n / 8}, false, false);
  return block_layout;
}

Fragment makeGemmFragmentC(const int block_m, const int block_n, const int warp_m, const int warp_n,
                           const int element_size) {
  if (element_size == 64) return makeGemmFragmentC_F64(block_m, block_n, warp_m, warp_n);
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(warp_n % 16 == 0);
  auto base_layout = makeGemmFragment8x8()->Repeat({2, 1}, false);
  auto warp_layout = base_layout->Repeat({block_m / warp_m, block_n / warp_n}, true, false);
  auto block_layout = warp_layout->Repeat({warp_m / 16, warp_n / 8}, false, false);
  return block_layout;
}

Fragment makeGemmFragmentA(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(block_k % 16 == 0);
  auto base_layout = makeGemmFragment8x8()->Repeat({2, 2}, false, false);
  auto warp_layout = base_layout->Repeat({block_m / warp_m, 1}, true)->Replicate(block_n / warp_n);
  auto block_layout = warp_layout->Repeat({warp_m / 16, block_k / 16}, false, false);
  return block_layout;
}

Fragment makeGemmFragmentB(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n) {
  // transposed
  ICHECK(warp_n % 8 == 0);
  ICHECK(block_k % 16 == 0);
  auto base_layout = makeGemmFragment8x8Transposed()->Repeat({2, 1}, false, false);
  auto warp_layout = base_layout->Replicate(block_m / warp_m)->Repeat({1, block_n / warp_n}, true);
  auto block_layout = warp_layout->Repeat({block_k / 16, warp_n / 8}, false, true);
  return block_layout;
}

Fragment makeGemmFragment32x32(int element_size) {
  IterVar i = make_itervar("i", 32);
  IterVar j = make_itervar("j", 32);
  IterVar rep = make_itervar("rep", 1);
  ICHECK(element_size == 16 || element_size == 32);
  if (element_size == 16) {
    PrimExpr thd = FloorMod(i, 4) + FloorDiv(FloorMod(i, 16), 8) * 4 +
                   FloorDiv(FloorMod(j, 16), 8) * 8 + FloorDiv(i, 16) * 16;
    PrimExpr idx = FloorMod(j, 4) + FloorDiv(j, 16) * 4 + FloorDiv(FloorMod(i, 8), 4) * 8 +
                   FloorDiv(FloorMod(j, 8), 4) * 16;
    return Fragment({i, j}, {idx}, thd, rep);
  } else {
    PrimExpr thd = FloorMod(i, 2) + 2 * FloorDiv(FloorMod(j, 4), 2) +
                   FloorDiv(FloorMod(i, 16), 8) * 4 + FloorDiv(FloorMod(j, 16), 8) * 8 +
                   FloorDiv(i, 16) * 16;
    PrimExpr idx = FloorMod(j, 2) + 2 * FloorDiv(FloorMod(i, 4), 2) + FloorDiv(j, 16) * 4 +
                   FloorDiv(FloorMod(i, 8), 4) * 8 + FloorDiv(FloorMod(j, 8), 4) * 16;
    return Fragment({i, j}, {idx}, thd, rep);
  }
}

Fragment makeGemmVoltaFragmentC(const int block_m, const int block_n, const int warp_m,
                                const int warp_n, int element_size) {
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 32 == 0);
  ICHECK(warp_n % 32 == 0);
  auto base_layout = makeGemmFragment32x32(element_size);
  auto warp_layout = base_layout->Repeat({warp_m / 32, warp_n / 32}, false, false);
  auto block_layout = warp_layout->Repeat({block_m / warp_m, block_n / warp_n}, true);
  return block_layout;
}

Fragment makeGemmVoltaFragmentA(const int block_m, const int block_n, const int block_k,
                                const int warp_m, const int warp_n) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 32 == 0);
  ICHECK(block_k % 4 == 0);
  // this is a special case
  IterVar i = make_itervar("i", 32);
  IterVar j = make_itervar("j", 4);
  IterVar rep = make_itervar("rep", 2);
  PrimExpr thd = FloorDiv(FloorMod(i, 16), 8) * 4 + 16 * FloorDiv(i, 16) + FloorMod(i, 4) + 8 * rep;
  PrimExpr idx = j + FloorDiv(FloorMod(i, 8), 4) * 4;
  Fragment base_layout = Fragment({i, j}, {idx}, thd, rep);
  auto warp_layout = base_layout->Repeat({warp_m / 32, block_k / 4}, false, false);
  auto block_layout = warp_layout->Replicate(block_n / warp_n)->Repeat({block_m / warp_m, 1}, true);
  return block_layout;
}

PrimExpr xor2x2(const PrimExpr& i, const PrimExpr& j) { return FloorMod(i + j, 2); }

PrimExpr xor4x4(const PrimExpr& i, const PrimExpr& j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor2x2(i1, j1) + xor2x2(i0, j0);
}

PrimExpr xor8x8(const PrimExpr& i, const PrimExpr j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor4x4(i1, j1) + xor2x2(i0, j0);
}

Layout makeGemmABLayoutHalfBank(int stride, int continuous, int element_size) {
  // Swizzle 2 bit
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  int vector_size = 128 / element_size;
  ICHECK(stride % 8 == 0);
  ICHECK(continuous % (vector_size * 4) == 0);
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 4);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 4);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor4x4(c, FloorDiv(s, 2));
  PrimExpr index = vec + (c_swizzle + s * 4) * vector_size;
  return Layout(Array{i, j}, {tc, ts, index});
}

Layout makeGemmABLayoutFullBank(int stride, int continuous, int element_size) {
  // Swizzle 3 bit
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  int vector_size = 128 / element_size;
  ICHECK(stride % 8 == 0);
  ICHECK(continuous % (vector_size * 8) == 0);
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 8);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor8x8(c, s);
  PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;
  return Layout(Array{i, j}, {tc, ts, index});
}

Layout makeGemmABLayoutF64_Kinner(int stride, int continuous) {
  // Swizzle<2, 0, 4>
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr swizzled_c = FloorDiv(c, 4) * 4 + xor4x4(FloorMod(c, 4), s);
  PrimExpr index = swizzled_c + s * 16;
  return Layout(Array{i, j}, {tc, ts, index});
}

Layout makeGemmABLayoutF64_Kouter(int stride, int continuous) {
  // Swizzle<2, 2, 2>
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr swizzled_c = FloorMod(c, 4) + xor4x4(FloorDiv(c, 4), s) * 4;
  PrimExpr index = swizzled_c + s * 16;
  return Layout(Array{i, j}, {tc, ts, index});
}

Layout makeGemmABLayoutPadded(int stride, int continuous, int element_size) {
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  int padded = continuous;
  // Add 128 bits padding when the last dim is a multiple of 256 bits
  if ((element_size * continuous) % 256 == 0) padded += 128 / element_size;
  return Layout(Array{i, j}, {i * padded + j});
}

Layout MakeGemmVoltaABLayoutCrosswise(int stride, int continuous) {
  ICHECK(stride % 32 == 0 && continuous % 32 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 4);
  PrimExpr vec_strided_within_tile = FloorMod(vec_contiguous_idx, 8);

  PrimExpr bit2 = FloorMod(FloorDiv(FloorMod(i, 32), 16) + FloorDiv(FloorMod(i, 16), 8) +
                               FloorDiv(vec_strided_within_tile, 4),
                           2);
  PrimExpr bit1 =
      xor2x2(FloorDiv(FloorMod(i, 8), 4), FloorDiv(FloorMod(vec_strided_within_tile, 4), 2));
  PrimExpr permuted_vec_contiguous = FloorDiv(i, 16) * 16 + FloorMod(i, 4) * 4 + bit2 * 2 + bit1;

  PrimExpr offset = FloorMod(j, 4) + permuted_vec_contiguous * 4 + vec_contiguous_idx * stride * 4;
  return Layout(Array{i, j}, {offset});
}

Layout MakeGemmVoltaALayoutCongruous(int stride, int continuous) {
  ICHECK(stride % 4 == 0 && continuous % 64 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 8);
  PrimExpr vec_strided_idx = i;
  PrimExpr tile_contiguous_idx = FloorDiv(vec_contiguous_idx, 8);
  PrimExpr tile_strided_idx = FloorDiv(vec_strided_idx, 4);
  PrimExpr tile_contiguous_residual = FloorMod(vec_contiguous_idx, 8);
  PrimExpr tile_strided_residual = FloorMod(vec_strided_idx, 4);

  PrimExpr permuted_strided_within_tile = FloorDiv(tile_contiguous_residual, 2);
  PrimExpr permuted_contiguous_within_tile =
      FloorMod(tile_contiguous_residual, 2) * 4 +
      xor4x4(tile_strided_residual, permuted_strided_within_tile);

  PrimExpr element_strided = permuted_strided_within_tile + tile_strided_idx * 4;
  PrimExpr element_contiguous =
      FloorMod(j, 8) + (permuted_contiguous_within_tile + tile_contiguous_idx * 8) * 8;
  PrimExpr offset = element_strided * continuous + element_contiguous;
  return Layout(Array{i, j}, {offset});
}

Layout MakeGemmVoltaBLayoutCongruous(int stride, int continuous) {
  ICHECK(stride % 4 == 0 && continuous % 64 == 0);
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, 8);
  PrimExpr vec_strided_idx = i;
  PrimExpr tile_contiguous_idx = FloorDiv(vec_contiguous_idx, 8);
  PrimExpr tile_strided_idx = FloorDiv(vec_strided_idx, 4);
  PrimExpr tile_contiguous_residual = FloorMod(vec_contiguous_idx, 8);
  PrimExpr tile_strided_residual = FloorMod(vec_strided_idx, 4);

  PrimExpr permuted_strided_within_tile = FloorMod(tile_contiguous_residual, 4);
  PrimExpr permuted_contiguous_within_tile =
      FloorDiv(tile_contiguous_residual, 4) * 4 +
      xor4x4(tile_strided_residual, permuted_strided_within_tile);

  PrimExpr element_strided = permuted_strided_within_tile + tile_strided_idx * 4;
  PrimExpr element_contiguous =
      FloorMod(j, 8) + (permuted_contiguous_within_tile + tile_contiguous_idx * 8) * 8;
  PrimExpr offset = element_strided * continuous + element_contiguous;
  return Layout(Array{i, j}, {offset});
}

Layout makeGemmVoltaABLayout(int stride, int continuous, bool is_a, int kfactor) {
  if (kfactor == 2) return MakeGemmVoltaABLayoutCrosswise(stride, continuous);
  if (is_a && continuous % 64 == 0) return MakeGemmVoltaALayoutCongruous(stride, continuous);
  if (!is_a && continuous % 64 == 0) return MakeGemmVoltaBLayoutCongruous(stride, continuous);
  return makeGemmABLayoutPadded(stride, continuous, 16);
}

Layout makeGemmABLayout(int stride, int continuous, int element_size, int kfactor) {
  if (element_size == 64) {
    if (kfactor == 1 && continuous % 16 == 0)  // float64 KxN
      return makeGemmABLayoutF64_Kouter(stride, continuous);
    if (kfactor == 2 && continuous % 16 == 0)  // float64 NxK
      return makeGemmABLayoutF64_Kinner(stride, continuous);
    return makeGemmABLayoutPadded(stride, continuous, element_size);
  }
  int vector_size = 128 / element_size;
  if (kfactor == 1 && element_size == 8)  // int8 KxN
    return makeGemmABLayoutPadded(stride, continuous, element_size);
  else if (continuous % (vector_size * 8) == 0)
    return makeGemmABLayoutFullBank(stride, continuous, element_size);
  else if (continuous % (vector_size * 4) == 0)
    return makeGemmABLayoutHalfBank(stride, continuous, element_size);
  else {
    return makeGemmABLayoutPadded(stride, continuous, element_size);
  }
}

TVM_REGISTER_NODE_TYPE(LayoutNode);
TVM_REGISTER_NODE_TYPE(FragmentNode);

TVM_REGISTER_GLOBAL("tl.Layout").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = Layout(Array<IterVar>(args[0]), Array<PrimExpr>(args[1]));
});

TVM_REGISTER_GLOBAL("tl.Layout_input_shape").set_body_typed([](Layout layout) {
  return layout->InputShape();
});

TVM_REGISTER_GLOBAL("tl.Layout_output_shape").set_body_typed([](Layout layout) {
  return layout->OutputShape();
});

TVM_REGISTER_GLOBAL("tl.Layout_inverse").set_body_typed([](Layout layout) {
  return layout->Inverse();
});

TVM_REGISTER_GLOBAL("tl.Layout_index").set_body_typed([](Layout layout) {
  return layout->forward_index_;
});

TVM_REGISTER_GLOBAL("tl.Layout_vector_size").set_body_typed([](Layout layout) {
  return layout->VectorSize();
});

TVM_REGISTER_GLOBAL("tl.Fragment").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = Fragment(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("tl.Fragment_thread_size").set_body_typed([](Fragment fragment) {
  return fragment->ThreadExtent();
});

TVM_REGISTER_GLOBAL("tl.Fragment_thread").set_body_typed([](Fragment fragment) {
  return fragment->forward_thread_;
});

TVM_REGISTER_GLOBAL("tl.Fragment_repeat")
    .set_body_typed([](Fragment fragment, Array<PrimExpr> repeats, bool repeat_on_thread,
                       bool lower_dim_first) {
      return fragment->Repeat(repeats, repeat_on_thread, lower_dim_first);
    });

TVM_REGISTER_GLOBAL("tl.Fragment_replicate").set_body_typed([](Fragment fragment, int repeats) {
  return fragment->Replicate(repeats);
});

TVM_REGISTER_GLOBAL("tl.Fragment_condense_rep_var").set_body_typed([](Fragment fragment) {
  return fragment->CondenseReplicateVar();
});

TVM_REGISTER_GLOBAL("tl.make_swizzled_layout")
    .set_body_typed([](int stride, int continuous, int element_size) {
      return makeGemmABLayout(stride, continuous, element_size, 0);
    });

}  // namespace tl
}  // namespace tvm
