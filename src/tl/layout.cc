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
 * \file Layout.cc
 * \brief Define Layout used in MMA and other operations.
 *
 */

#include "layout.h"

#include <tvm/arith/pattern.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../arith/pattern_match.h"
#include "arith.h"
#include "helper.h"

namespace tvm {
namespace tl {

using namespace tir;

IterVar make_itervar(std::string name, PrimExpr dom) {
  Var var = Var(name);
  return IterVar(Range(0, dom), var, IterVarType::kDataPar);
}

Layout::Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index) {
  auto n = make_object<LayoutNode>();
  n->forward_var_ = std::move(forward_var);
  arith::Analyzer analyzer;
  n->UpdateAnalyzer(&analyzer);
  n->forward_index_ = forward_index.Map([&](const PrimExpr& e) { return analyzer.Simplify(e); });
  data_ = std::move(n);
}

void LayoutNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("forward_var", &forward_var_);
  v->Visit("forward_index", &forward_index_);
}

void LayoutNode::UpdateAnalyzer(arith::Analyzer* analyzer) const {
  for (size_t i = 0; i < InputDim(); i++) {
    analyzer->Bind(forward_var_[i]->var, forward_var_[i]->dom);
  }
}

Array<PrimExpr> LayoutNode::InputShape() const {
  Array<PrimExpr> ret(InputDim(), 1);
  for (size_t i = 0; i < ret.size(); i++) {
    ret.Set(i, forward_var_[i]->dom->extent);
  }
  return ret;
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
    vmap.Set(forward_var_[i]->var, vars[i]);
  }
  return forward_index_.Map([&](const PrimExpr& e) { return Substitute(e, vmap); });
}

PrimExpr LayoutNode::GetFlattenedIndice() const {
  arith::Analyzer analyzer;
  UpdateAnalyzer(&analyzer);
  auto shape = OutputShape();
  PrimExpr result = 0, stride = 1;
  for (int i = forward_index_.size() - 1; i >= 0; i--) {
    result += forward_index_[i] * stride;
    stride *= shape[i];
  }
  result = analyzer.Simplify(result);
  return result;
}

Fragment FragmentNode::Repeat(const Array<PrimExpr>& repeats, bool repeat_on_thread,
                              bool lower_dim_first) const {
  ICHECK_EQ(repeats.size(), InputDim());
  Array<IterVar> new_forward_var;
  Map<Var, PrimExpr> vmap;
  for (size_t i = 0; i < InputDim(); i++) {
    auto v = forward_var_[i];
    new_forward_var.push_back(
        IterVar(Range(v->dom->min, v->dom->extent * repeats[i]), v->var, v->iter_type));
    vmap.Set(v->var, FloorMod(v->var, InputShape()[i]));
  }

  PrimExpr repeats_index = 0, repeat_stride = 1;
  if (lower_dim_first) {
    for (int i = InputDim() - 1; i >= 0; i--) {
      repeats_index += repeat_stride * FloorDiv(new_forward_var[i]->var, InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  } else {
    for (size_t i = 0; i < InputDim(); i++) {
      repeats_index += repeat_stride * FloorDiv(new_forward_var[i]->var, InputShape()[i]);
      repeat_stride *= repeats[i];
    }
  }

  if (repeat_on_thread) {
    PrimExpr thread_size = ThreadExtent();
    auto new_forward_index =
        forward_index_.Map([&](const PrimExpr& e) { return Substitute(e, vmap); });
    auto new_forward_thread = Substitute(forward_thread_, vmap) + thread_size * repeats_index;
    return Fragment(new_forward_var, new_forward_index, new_forward_thread, thread_replicate_);
  } else {
    ICHECK(OutputDim() == 1);
    PrimExpr frag_len = OutputShape()[0];
    Array<PrimExpr> new_forward_index = {Substitute(forward_index_[0], vmap) +
                                         frag_len * repeats_index};
    PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
    return Fragment(new_forward_var, new_forward_index, new_forward_thread, thread_replicate_);
  }
}

Fragment FragmentNode::Replicate(int repeats) const {
  ICHECK(repeats >= 1);
  IterVar new_rep = make_itervar("rep", ReplicateExtent() * repeats);
  Map<Var, PrimExpr> vmap;
  vmap.Set(thread_replicate_->var, FloorMod(new_rep->var, ReplicateExtent()));
  PrimExpr new_forward_thread = Substitute(forward_thread_, vmap) +
                                ThreadExtent() * FloorDiv(new_rep->var, ReplicateExtent());
  return Fragment(forward_var_, forward_index_, new_forward_thread, new_rep);
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

  IterVar rep = make_itervar("rep", int(*rep_size) / factor);
  Map<Var, PrimExpr> vmap;
  vmap.Set(thread_replicate_->var, rep * factor + FloorMod(forward_index_[0], factor));
  PrimExpr new_forward_thread = Substitute(forward_thread_, vmap);
  Array<PrimExpr> new_forward_index = {FloorDiv(forward_index_[0], factor)};
  return Fragment(forward_var_, new_forward_index, new_forward_thread, rep);
}

Layout LayoutNode::Inverse() const {
  Map<Var, Range> input_iters;
  arith::Analyzer analyzer;
  for (auto iv : forward_var_) {
    input_iters.Set(iv->var, iv->dom);
  }
  arith::IterMapResult res = arith::DetectIterMap(forward_index_, input_iters, 1,
                                                  arith::IterMapLevel::Bijective, &analyzer);
  ICHECK(res->errors.empty()) << res->errors;

  Array<IterVar> inverse_iter_var;
  auto outputs_shape = OutputShape();
  Array<PrimExpr> outputs;
  for (size_t i = 0; i < OutputDim(); i++) {
    IterVar iv = make_itervar("v" + std::to_string(i), outputs_shape[i]);
    inverse_iter_var.push_back(iv);
    outputs.push_back(iv->var);
  }

  auto inv = arith::InverseAffineIterMap(res->indices, outputs);

  Array<PrimExpr> backward_index;
  for (size_t i = 0; i < InputDim(); i++) {
    if (inv.find(forward_var_[i]->var) != inv.end()) {
      backward_index.push_back(inv[forward_var_[i]->var]);
    } else {
      backward_index.push_back(0);
    }
  }

  return Layout(inverse_iter_var, backward_index);
}

int LayoutNode::VectorSize() const {
  auto last_dim = OutputShape().back().as<IntImm>();
  if (!last_dim.defined()) return 1;
  int vector_size = 2;
  PrimExpr last_index = forward_index_.back();
  Var last_var = forward_var_.back()->var;
  arith::Analyzer analyzer;
  auto iter_sum = arith::NormalizeToIterSum(last_index, ToVMap(forward_var_), &analyzer);
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

PrimExpr infer_fragment_index(const Array<IterVar>& forward_var, const IterVar& thread_replicate,
                              const PrimExpr& forward_thread, arith::Analyzer* analyzer) {
  Array<IterVar> input_iters = forward_var;
  input_iters.push_back(thread_replicate);
  Array<arith::IterSplitExpr> splits =
      DivideUnusedIterators({forward_thread}, input_iters, analyzer);

  Array<arith::IterSplitExpr> split_without_rep;
  for (const auto& split : splits) {
    CHECK(split->source->source.as<Var>());
    if (split->source->source.as<Var>().value().same_as(thread_replicate->var)) continue;
    split_without_rep.push_back(split);
  }
  return MakeFlattenedExpression(split_without_rep);
}

Fragment::Fragment(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, IterVar thread_replicate) {
  if (!thread_replicate.defined()) thread_replicate = make_itervar("unused", 1);
  ICHECK(is_zero(thread_replicate->dom->min));

  auto n = make_object<FragmentNode>();
  n->forward_var_ = std::move(forward_var);
  n->thread_replicate_ = std::move(thread_replicate);

  arith::Analyzer analyzer;
  n->UpdateAnalyzer(&analyzer);
  n->forward_thread_ = analyzer.Simplify(forward_thread);

  if (forward_index.empty()) {
    forward_index = {
        infer_fragment_index(n->forward_var_, n->thread_replicate_, n->forward_thread_, &analyzer)};
  }

  n->forward_index_ = forward_index.Map([&](const PrimExpr& e) { return analyzer.Simplify(e); });
  data_ = std::move(n);
}

void FragmentNode::VisitAttrs(tvm::AttrVisitor* v) {
  LayoutNode::VisitAttrs(v);
  v->Visit("forward_thread", &forward_thread_);
  v->Visit("thread_replicate", &thread_replicate_);
}

void FragmentNode::UpdateAnalyzer(arith::Analyzer* analyzer) const {
  LayoutNode::UpdateAnalyzer(analyzer);
  analyzer->Bind(thread_replicate_->var, thread_replicate_->dom);
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
  if (!vars.empty()) {
    ICHECK_EQ(vars.size(), InputDim());
    for (size_t i = 0; i < InputDim(); i++) {
      vmap.Set(forward_var_[i]->var, vars[i]);
    }
  }
  if (rep_var.defined()) vmap.Set(thread_replicate_->var, rep_var.value());

  return Substitute(forward_thread_, vmap);
}

Layout FragmentNode::Inverse() const {
  auto new_fwd_vars = forward_var_;
  new_fwd_vars.push_back(thread_replicate_);
  auto new_fwd_index = forward_index_;
  new_fwd_index.push_back(forward_thread_);
  auto fwd = Layout(new_fwd_vars, new_fwd_index);
  auto bwd = fwd->Inverse();
  return bwd;
}

Fragment FragmentNode::CondenseReplicateVar() const {
  arith::Analyzer analyzer;
  auto input_iters = forward_var_;
  input_iters.push_back(thread_replicate_);
  PrimExpr new_forward_thread;
  IterVar new_thread_replicate;
  std::tie(new_forward_thread, new_thread_replicate) =
      CompressIterator(forward_thread_, input_iters, thread_replicate_, &analyzer);
  return Fragment(forward_var_, forward_index_, new_forward_thread, new_thread_replicate);
}

void LayoutNode::DebugOutput() const {
  LOG_DEBUG << "Layout Shape: " << InputShape() << " -> " << OutputShape();
  LOG_DEBUG << "Layout Index: " << forward_var_.Map([](const IterVar& iv) { return iv->var; })
            << " -> " << forward_index_;
}

void FragmentNode::DebugOutput() const {
  LayoutNode::DebugOutput();
  LOG_DEBUG << "Fragment Shape: " << ThreadExtent();
  LOG_DEBUG << "Fragment Replicate: " << thread_replicate_->var << " "
            << thread_replicate_->dom->extent;
  LOG_DEBUG << "Fragment Index: " << forward_thread_;
}

bool LayoutNode::SEqualReduce(const LayoutNode* other, SEqualReducer equal) const {
  Array<PrimExpr> vars;
  for (size_t i = 0; i < this->InputDim(); i++) vars.push_back(Var());
  return this->InputDim() == other->InputDim() && equal(this->InputShape(), other->InputShape()) &&
         equal(this->Forward(vars), other->Forward(vars));
}

bool FragmentNode::SEqualReduce(const FragmentNode* other, SEqualReducer equal) const {
  this->ReplicateExtent() == other->ReplicateExtent();
  equal(this->ThreadExtent(), other->ThreadExtent());
  Array<PrimExpr> vars;
  Var rep_var{};
  for (size_t i = 0; i < this->InputDim(); i++) vars.push_back(Var());
  return this->InputDim() == other->InputDim() &&
         equal(this->ReplicateExtent(), other->ReplicateExtent()) &&
         equal(this->InputShape(), other->InputShape()) &&
         equal(this->ThreadExtent(), other->ThreadExtent()) &&
         equal(this->Forward(vars), other->Forward(vars)) &&
         equal(this->ForwardThread(vars, rep_var), other->ForwardThread(vars, rep_var));
}

bool FragmentThreadEqual(const Fragment& a, const Fragment& b) {
  if (a->InputDim() != b->InputDim()) return false;
  if (!StructuralEqual()(a->ReplicateExtent(), b->ReplicateExtent())) return false;
  Var rep = Var();
  Array<PrimExpr> vars;
  for (size_t i = 0; i < a->InputDim(); i++) vars.push_back(Var());
  if (!StructuralEqual()(a->ForwardThread(vars, rep), b->ForwardThread(vars, rep))) return false;
  return true;
}

Fragment makeGemmFragment8x8() {
  IterVar i = make_itervar("i", 8);
  IterVar j = make_itervar("j", 8);
  IterVar rep = make_itervar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 2) + 4 * i;
  PrimExpr index = FloorMod(j->var, 2);
  return Fragment({i, j}, {index}, forward_thread, rep);
}

Fragment makeGemmFragmentC(const int block_m, const int block_n, const int warp_m,
                           const int warp_n) {
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 8 == 0);
  ICHECK(warp_n % 8 == 0);
  auto base_layout = makeGemmFragment8x8();
  auto warp_layout = base_layout->Repeat({warp_m / 8, warp_n / 8}, false, false);
  auto block_layout = warp_layout->Repeat({block_m / warp_m, block_n / warp_n}, true);
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

Fragment makeGemmFragmentA(const int block_m, const int block_n, const int block_k,
                           const int warp_m, const int warp_n) {
  // assume not transposed
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK(warp_m % 16 == 0);
  ICHECK(block_k % 16 == 0);
  auto base_layout = makeGemmFragment8x8()->Repeat({2, 2}, false, false);
  auto warp_layout = base_layout->Repeat({warp_m / 16, block_k / 16}, false, false);
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

Layout makeGemmABLayoutF32Congruous(int stride, int continuous) {
  //   int tc = coord.contiguous() / 32;
  //   int ts = coord.strided() / 4;
  //   int c = (coord.contiguous() % 32) / kElementsPerAccess;
  //   int s = coord.strided() % 4;
  //   LongIndex offset = (c ^ (2 * s)) * kElementsPerAccess + s * stride_[0] +
  //                      tc * 32 + ts * stride_[0] * 4 + coord.contiguous() % 4;
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);

  PrimExpr tc = FloorDiv(j, 32);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorDiv(FloorMod(j, 32), 4);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr x = FloorMod(c, 2) + 2 * xor4x4(FloorDiv(c, 2), s);
  PrimExpr final_offset = x * 4 + s * continuous + tc * 32 + ts * continuous * 4 + FloorMod(j, 4);
  return Layout({i, j}, {final_offset});
}

Layout makeGemmABLayoutF64Congruous(int stride, int continuous) {
  // int tc = coord.contiguous() / 16;
  // int ts = coord.strided() / 4;
  // int c = coord.contiguous() % 16;
  // int s = coord.strided() % 4;
  // int bank = ((((c & 1) * 4 + (c & 6) / 2)) ^ (s & 1)) * 2 + (c / 8);
  // int row = (c & 6) / 2;
  // bank ^= ((s & 2) * 2);
  // LongIndex offset = tc * 16 + bank + (ts * 4 + row) * stride_[0];
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr row = FloorMod(FloorDiv(c, 2), 4);
  PrimExpr bank = FloorMod(c, 2) * 8 + xor4x4(row, s) * 2 + FloorDiv(c, 8);
  PrimExpr offset = tc * 16 + bank + (ts * 4 + row) * continuous;
  return Layout({i, j}, {offset});
}

Layout makeGemmABLayoutF64Crosswise(int stride, int continuous) {
  // int tc = coord.contiguous() / 16;
  // int ts = coord.strided() / 16;
  // int c = coord.contiguous() % 16;
  // int s = coord.strided() % 16;
  // int k_group = c / 4;
  // int access_s = s / 2;
  // int row = access_s % 4;
  // int bank = ((k_group & 2) << 2) ^ ((s % 2) << 3) + (c % 4) * 2 + (access_s / 4) ^ (k_group &
  // 1); int smem_row = (k_group * 4 + row) + tc * 16; int smem_col = ts * 16 + bank; LongIndex
  // offset = smem_row * stride_[0] + smem_col;
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 16);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 16);
  PrimExpr k_group = FloorDiv(c, 4);
  PrimExpr access_s = FloorDiv(s, 2);
  PrimExpr row = FloorMod(access_s, 4);
  PrimExpr bank = xor2x2(FloorDiv(k_group, 2), FloorMod(s, 2)) * 8 + FloorMod(c, 4) * 2 +
                  xor2x2(FloorDiv(access_s, 4), FloorMod(k_group, 2));
  PrimExpr smem_row = (k_group * 4 + row) + tc * 16;
  PrimExpr smem_col = ts * 16 + bank;
  PrimExpr offset = smem_row * stride + smem_col;
  return Layout({i, j}, {offset});
}

Layout makeGemmABLayoutPadded(int stride, int continuous, int element_size) {
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  int padded = continuous;
  // Add 128 bits padding when the last dim is a multiple of 256 bits
  if ((element_size * continuous) % 256 == 0) padded += 128 / element_size;
  return Layout({i, j}, {i * padded + j});
}

Layout makeGemmABLayoutCommon(int stride, int continuous, int element_size, int kfactor) {
  int access_elements = 128 / element_size;
  ICHECK(kfactor == 1 || kfactor == 2);
  // can satisfy 8 x 128bits access
  ICHECK(continuous * element_size * kfactor % 1024 == 0);
  int tile_shape[2] = {8 / kfactor, std::max(8 / kfactor, 4)};
  int partition_shape[2] = {4, 4};
  IterVar i = make_itervar("i", stride);
  IterVar j = make_itervar("j", continuous);
  PrimExpr vec_contiguous_idx = FloorDiv(j, access_elements);
  PrimExpr vec_strided_idx = FloorDiv(i, kfactor);
  PrimExpr tile_contiguous_idx = FloorDiv(vec_contiguous_idx, tile_shape[1]);
  PrimExpr tile_contiguous_residual =
      FloorMod(vec_contiguous_idx, tile_shape[1]) + FloorMod(i, kfactor) * tile_shape[1];
  PrimExpr tile_strided_residual = FloorMod(vec_strided_idx, tile_shape[0]);

  PrimExpr partition_contiguous_idx = FloorDiv(tile_contiguous_residual, partition_shape[1]);
  PrimExpr partition_strided_idx = FloorDiv(tile_strided_residual, partition_shape[0]);
  PrimExpr partition_contiguous_residual = FloorMod(tile_contiguous_residual, partition_shape[1]);
  PrimExpr partition_strided_residual = FloorMod(tile_strided_residual, partition_shape[0]);

  PrimExpr permuted_vec_contiguous_within_partition =
      xor4x4(partition_contiguous_residual, partition_strided_residual);
  PrimExpr permuted_partition_contiguous_within_tile =
      xor2x2(partition_contiguous_idx, partition_strided_idx);

  PrimExpr element_contiguous =
      (permuted_vec_contiguous_within_partition + tile_contiguous_idx * tile_shape[1] * kfactor +
       permuted_partition_contiguous_within_tile * partition_shape[1]) *
          access_elements +
      FloorMod(j, access_elements);

  PrimExpr final_offset = vec_strided_idx * continuous * kfactor + element_contiguous;
  return Layout({i, j}, {final_offset});
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
  return Layout({i, j}, {offset});
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
  return Layout({i, j}, {offset});
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
  return Layout({i, j}, {offset});
}

Layout makeGemmVoltaABLayout(int stride, int continuous, bool is_a, int kfactor) {
  if (kfactor == 2) return MakeGemmVoltaABLayoutCrosswise(stride, continuous);
  if (is_a && continuous % 64 == 0) return MakeGemmVoltaALayoutCongruous(stride, continuous);
  if (!is_a && continuous % 64 == 0) return MakeGemmVoltaBLayoutCongruous(stride, continuous);
  return makeGemmABLayoutPadded(stride, continuous, 16);
}

Layout makeGemmABLayout(int stride, int continuous, int element_size, int kfactor) {
  if (element_size == 64 && kfactor == 1 && continuous % 16 == 0)
    return makeGemmABLayoutF64Congruous(stride, continuous);
  if (element_size == 64 && kfactor == 2 && stride % 16 == 0 && continuous % 16 == 0)
    return makeGemmABLayoutF64Crosswise(stride, continuous);
  if (element_size == 32 && kfactor == 1 && continuous % 32 == 0)
    return makeGemmABLayoutF32Congruous(stride, continuous);
  if (element_size == 8 && kfactor == 1)
    return makeGemmABLayoutPadded(stride, continuous, element_size);
  if (continuous * element_size * kfactor % 1024 == 0)
    return makeGemmABLayoutCommon(stride, continuous, element_size, kfactor);
  return makeGemmABLayoutPadded(stride, continuous, element_size);
}

TVM_REGISTER_NODE_TYPE(LayoutNode);
TVM_REGISTER_NODE_TYPE(FragmentNode);

TVM_REGISTER_GLOBAL("tl.Layout").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = Layout(args[0], args[1]);
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

TVM_REGISTER_GLOBAL("tl.Layout_var").set_body_typed([](Layout layout) {
  return layout->forward_var_;
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

TVM_REGISTER_GLOBAL("tl.Fragment_replicate_var").set_body_typed([](Fragment fragment) {
  return fragment->thread_replicate_;
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

}  // namespace tl
}  // namespace tvm
