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
 * \file touch_extractor.cc
 * \brief Extract feature of touch pattern of axes in lowered IR
 */

#include "touch_extractor.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_map>

namespace tvm {
namespace autotvm {

int ParallelLevel(AnnotationType ann) {
  switch (ann) {
    case kBlockX:
    case kBlockY:
    case kBlockZ:
      return 2;
    case kThreadX:
    case kThreadY:
    case kThreadZ:
    case kParallel:
      return 1;
    default:
      return 0;
  }
}

// get touch pattern from index expression
class IndexParser : public ExprVisitor {
 public:
  void Parse(PrimExpr expr) {
    pattern_map.clear();
    this->VisitExpr(expr);
  }

  void VisitExpr_(const VarNode* op) final {
    // TODO(lmzheng): handle more index types (multiple occurrence)
    if (pattern_map.count(op) == 0) {
      pattern_map[op] = TouchPattern();
      pattern_map[op].stride = next_stride_;
      next_stride_ = 1;
    }
  }

  void VisitExpr_(const MulNode* op) final {
    if (op->a.as<VarNode>()) {
      if (const auto stride = op->b.as<IntImmNode>()) {
        next_stride_ = stride->value;
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  std::unordered_map<const VarNode*, TouchPattern> pattern_map;

 private:
  int64_t next_stride_ = 1;
};

// extract iter vars and their touch pattern from ir
bool TouchExtractor::EnterItervar_(Var var, int64_t length, AnnotationType ann_type) {
  // do not insert duplicated occurrences of virtual thread
  if (ann_type == kVirtualThread && itervar_map.count(var) != 0) {
    skip_stack_size_.push_back(itervar_stack_.size());
    return true;
  } else {
    itervar_stack_.push_back(var);
    topdown_product_ *= length;

    if (itervar_map.count(var) != 0) {
      // find two duplicated axes
      // these happens when we create tvm.thread_axis("threadIdx.x") once and
      // bind it twice. Here we treat them as two axes
      // so we create a snapshot for the old one and freeze it
      Var old = Var(var.get()->name_hint);
      itervar_map.insert({old, itervar_map[var]});
      itervar_map.erase(var);
    }

    itervar_map.insert(
        {var, ItervarFeature(var, length, static_cast<int>(itervar_stack_.size()), ann_type,
                             topdown_product_, static_cast<int>(itervar_counter_++))});
  }

  return true;
}

void TouchExtractor::ExitItervar_() {
  if (!skip_stack_size_.empty() && skip_stack_size_.back() == itervar_stack_.size()) {
    skip_stack_size_.pop_back();
    return;
  }
  Var var = itervar_stack_.back();

  // update count and reuse ratio for upper iter vars (includes self)
  for (auto kv : itervar_map[var].touch_feature) {
    if (kv.second.stride != 0) {  // multiply count
      for (auto stack_var : itervar_stack_) {
        auto touch_pattern = itervar_map[stack_var].touch_feature.find(kv.first);
        ICHECK(touch_pattern != itervar_map[stack_var].touch_feature.end());
        touch_pattern->second.count *= itervar_map[var].length;
      }
    } else {  // multiply reuse ratio
      for (auto stack_var : itervar_stack_) {
        auto touch_pattern = itervar_map[stack_var].touch_feature.find(kv.first);
        ICHECK(touch_pattern != itervar_map[stack_var].touch_feature.end());
        touch_pattern->second.reuse *= itervar_map[var].length;
      }
    }
  }
  itervar_stack_.pop_back();

  int64_t length = itervar_map[var].length;
  if (length != 0) topdown_product_ /= length;
  int64_t bottomup_product = -1;
  for (auto kv : itervar_map[var].touch_feature) {
    bottomup_product = std::max(bottomup_product, kv.second.count * kv.second.reuse);
  }

  itervar_map[var].bottomup_product = bottomup_product;

  // push base to upper parallel axis
  int para_level = ParallelLevel(itervar_map[var].ann);
  // if is the separate line of parallel level, push the base to upper parallel level
  if (!itervar_stack_.empty() &&
      ParallelLevel(itervar_map[itervar_stack_.back()].ann) == para_level + 1) {
    for (auto kv : itervar_map[var].touch_feature) {
      for (auto stack_var : itervar_stack_) {
        if (ParallelLevel(itervar_map[stack_var].ann) == para_level + 1) {
          auto touch_pattern = itervar_map[stack_var].touch_feature.find(kv.first);
          ICHECK(touch_pattern != itervar_map[stack_var].touch_feature.end());
          touch_pattern->second.thread_reuse = -kv.second.reuse;
          touch_pattern->second.thread_count = -kv.second.count;
          // NOTE: use minus as a flag to denote it is a base,
          // indicating it is not the final value
        }
      }
    }
  }

  for (auto kv : itervar_map[var].touch_feature) {
    if (kv.second.thread_count < 0) {
      itervar_map[var].touch_feature[kv.first].thread_count =
          kv.second.count / (-kv.second.thread_count);
      itervar_map[var].touch_feature[kv.first].thread_reuse =
          kv.second.reuse / (-kv.second.thread_reuse);
    }
  }
}

void TouchExtractor::EnterMem_(Var buffer_var, PrimExpr index) {
  std::string name = buffer_var.get()->name_hint;
  TouchedBuffer buf = name + "_" + std::to_string(buffer_counter_[name]++);

  // extract touch pattern from index
  IndexParser parser;
  parser.Parse(index);

  // push up mem access info
  for (auto var : itervar_stack_) {
    auto x = parser.pattern_map.find(var.get());
    if (x != parser.pattern_map.end()) {
      itervar_map[var].touch_feature[buf] = x->second;
    } else {
      itervar_map[var].touch_feature[buf] = TouchPattern();
    }
  }
}

void TouchExtractor::ExitMem_() {}

/*!
 * \brief Get axis-based feature for all axes
 * \param stmt The statement to be extracted
 * \param bool Whether take log for numerical feature
 * \param ret_feature The buffer where the return value is stored
 *
 * \note The format of return value is
 * ((
 *   ('_itervar_',  var),
 *   ('_attr_',     length, nest_level, topdown, bottomup, one_hot_annotation),
 *   ('_arith_',    add_ct, mul_ct, div_ct),
 *   ('data_vec_0', stride, mod, count, reuse, thread_count, thread_reuse),
 *   ('conv_0',     stride, mod, count, reuse, thread_count, thread_reuse),
 * ),
 * (
 *   ('_itervar_',    var2),
 *   ('_attr_',       length, nest_level, one_hot_annotation),
 *   ('_arith_',      add_ct, mul_ct, div_ct),
 *   ('kernel_vec_0', stride, mod, count, reuse, thread_count, thread_reuse),
 *   ('conv_1',       stride, mod, count, reuse, thread_count, thread_reuse),
 * ))
 *
 * Itervars are sorted according to their first occurrence position in IR.
 * Buffers touched by an itervar are sorted by their unique names.
 *
 * \note If you want to flatten these features as the input of your model,
 * You can use the faster one GetItervarFeatureFlatten below.
 */
void GetItervarFeature(Stmt stmt, bool take_log, Array<Array<Array<PrimExpr>>>* ret_feature) {
  // extract
  TouchExtractor touch_analyzer;
  touch_analyzer.Analyze(stmt);

  // sort according to order
  std::vector<Var> vars;
  for (auto kv : touch_analyzer.itervar_map) {
    vars.push_back(kv.first);
  }
  std::sort(vars.begin(), vars.end(), [&](const Var& lhs, const Var& rhs) -> bool {
    return touch_analyzer.itervar_map[lhs].order < touch_analyzer.itervar_map[rhs].order;
  });

  // whether take log for numerical feature
  std::function<double(int64_t)> trans;
  if (take_log) {
    trans = [](int64_t x) {
      if (x < 0) return -std::log(-x + 1) / std::log(2);
      x = x + 1;
      return std::log(x) / std::log(2);
    };
  } else {
    trans = [](int64_t x) { return x; };
  }

  // serialize for front end
  for (auto var : vars) {
    Array<Array<PrimExpr>> feature_row;
    ItervarFeature& fea = touch_analyzer.itervar_map[var];
    feature_row.push_back(Array<PrimExpr>{tvm::tir::StringImm("_itervar_"), var});

    Array<PrimExpr> attr{
        tvm::tir::StringImm("_attr_"),
        FloatImm(DataType::Float(32), trans(fea.length)),
        IntImm(DataType::Int(32), fea.nest_level),
        FloatImm(DataType::Float(32), trans(fea.topdown_product)),
        FloatImm(DataType::Float(32), trans(fea.bottomup_product)),
    };
    // one hot annotation
    for (int i = 0; i < kNum; i++) {
      attr.push_back(i == fea.ann);
    }
    feature_row.push_back(attr);

    // arithmetic
    feature_row.push_back(Array<PrimExpr>{
        tvm::tir::StringImm("_arith_"),
        FloatImm(DataType::Float(32), trans(fea.add_ct)),
        FloatImm(DataType::Float(32), trans(fea.mul_ct)),
        FloatImm(DataType::Float(32), trans(fea.div_ct)),
    });

    // touch map
    std::vector<TouchedBuffer> bufs;
    for (auto kv : fea.touch_feature) {
      bufs.push_back(kv.first);
    }
    std::sort(bufs.begin(), bufs.end());
    for (auto k : bufs) {
      TouchPattern& v = fea.touch_feature[k];
      feature_row.push_back(Array<PrimExpr>{
          tvm::tir::StringImm(k),
          FloatImm(DataType::Float(32), trans(v.stride)),
          FloatImm(DataType::Float(32), trans(v.mod)),
          FloatImm(DataType::Float(32), trans(v.count)),
          FloatImm(DataType::Float(32), trans(v.reuse)),
          FloatImm(DataType::Float(32), trans(v.thread_count)),
          FloatImm(DataType::Float(32), trans(v.thread_reuse)),
      });
    }

    ret_feature->push_back(feature_row);
  }
}

/*!
 * \brief Get axis-based feature for all axes and flatten them into a one-dimensional vector.
 * \param stmt The statement to be extracted
 * \param bool Whether take log for numerical feature
 * \param ret_feature The buffer where the return value is stored
 *
 * \note See GetItervarFeature for more details about the return value.
 *       This is an optimized version of GetItervarFeature + Flatten. This runs much faster.
 */
void GetItervarFeatureFlatten(Stmt stmt, bool take_log, std::vector<float>* ret_feature) {
  // extract touch feature
  TouchExtractor touch_analyzer;
  touch_analyzer.Analyze(stmt);

  // sort according to order
  std::vector<Var> vars;
  for (auto kv : touch_analyzer.itervar_map) {
    vars.push_back(kv.first);
  }
  std::sort(vars.begin(), vars.end(), [&](const Var& lhs, const Var& rhs) -> bool {
    return touch_analyzer.itervar_map[lhs].order < touch_analyzer.itervar_map[rhs].order;
  });

  // whether take log for numerical feature
  std::function<float(int64_t)> trans;
  if (take_log) {
    trans = [](int64_t x) {
      if (x < 0) return -std::log(-x + 1) / std::log(2);
      x = x + 1;
      return std::log(x) / std::log(2);
    };
  } else {
    trans = [](int64_t x) { return x; };
  }

  // serialize for front end
  for (auto var : vars) {
    ItervarFeature& fea = touch_analyzer.itervar_map[var];

    ret_feature->push_back(trans(fea.length));
    ret_feature->push_back(fea.nest_level);
    ret_feature->push_back(trans(fea.topdown_product));
    ret_feature->push_back(trans(fea.bottomup_product));

    // one hot annotation
    for (int i = 0; i < kNum; i++) {
      ret_feature->push_back(i == fea.ann);
    }

    // arithmetic
    ret_feature->push_back(trans(fea.add_ct));
    ret_feature->push_back(trans(fea.mul_ct));
    ret_feature->push_back(trans(fea.div_ct));

    // touch map
    std::vector<TouchedBuffer> bufs;
    for (auto kv : fea.touch_feature) {
      bufs.push_back(kv.first);
    }
    std::sort(bufs.begin(), bufs.end());
    for (auto k : bufs) {
      TouchPattern& v = fea.touch_feature[k];
      ret_feature->push_back(trans(v.stride));
      ret_feature->push_back(trans(v.mod));
      ret_feature->push_back(trans(v.count));
      ret_feature->push_back(trans(v.reuse));
      ret_feature->push_back(trans(v.thread_count));
      ret_feature->push_back(trans(v.thread_reuse));
    }
  }
}

/*!
 * \brief Get curve sample feature (relation feature) and flatten them into a one-dimensional
 * vector. \param stmt The statement to be extracted \param sample_n The number of points used for
 * sampling a curve (along one dimension) \param ret_feature The buffer where the return value is
 * stored
 */
void GetCurveSampleFeatureFlatten(Stmt stmt, int sample_n, std::vector<float>* ret_feature) {
  // extract touch feature
  TouchExtractor touch_ext;
  touch_ext.Analyze(stmt);

  // sort according to order
  std::vector<Var> vars;
  for (auto kv : touch_ext.itervar_map) {
    vars.push_back(kv.first);
  }
  std::sort(vars.begin(), vars.end(), [&](const Var& lhs, const Var& rhs) -> bool {
    return touch_ext.itervar_map[lhs].order < touch_ext.itervar_map[rhs].order;
  });

  int max_depth = 0;
  std::map<TouchedBuffer, std::vector<double>> reuse_curve;
  std::map<TouchedBuffer, std::vector<double>> count_curve;
  std::map<TouchedBuffer, std::vector<double>> topdown_curve;
  std::map<TouchedBuffer, std::vector<double>> bottomup_curve;
  std::set<TouchedBuffer> innermost_buffers;
  std::set<std::string> added;

  // find maximum depth of loop nest
  for (auto var : vars) {
    ItervarFeature& fea = touch_ext.itervar_map[var];
    max_depth = std::max(max_depth, fea.nest_level);
  }

  // mark inner most buffer
  for (auto iter = vars.rbegin(); iter != vars.rend(); iter++) {
    auto var = *iter;
    ItervarFeature& fea = touch_ext.itervar_map[var];
    if (fea.nest_level == max_depth) {
      for (auto kv : fea.touch_feature) {
        // delete buffer no (e.g. 'A_0' -> 'A', 'A_1' -> 'A')
        std::string raw_name = kv.first.substr(0, kv.first.rfind("_"));

        // delete memory scope (e.g. 'A.local' -> 'A', 'A.shared' -> 'A')
        size_t pos = raw_name.find(".");
        if (pos < kv.first.size()) raw_name = raw_name.substr(0, pos);

        // If there are multiple innermost buffers that are derived from a same raw buffer
        // We only record the last occurrence (note the `iter` is in reverse order)
        // e.g. `A.local`, `A.shared` are derived from `A`, if they all occurred at the inner most
        // level, we will only record the last occurrence,
        if (added.find(raw_name) == added.end()) {
          innermost_buffers.insert(kv.first);
          added.insert(raw_name);
        }
      }
    }
  }

  // pad the first point (zero) for all curves
  for (auto buf : innermost_buffers) {
    reuse_curve[buf].push_back(0);
    count_curve[buf].push_back(0);
    topdown_curve[buf].push_back(0);
    bottomup_curve[buf].push_back(0);
  }

  // extract curves
  for (auto var : vars) {
    ItervarFeature& fea = touch_ext.itervar_map[var];
    for (auto kv : fea.touch_feature) {
      if (innermost_buffers.find(kv.first) != innermost_buffers.end()) {
        reuse_curve[kv.first].emplace_back(std::log(kv.second.reuse) / std::log(2));
        count_curve[kv.first].emplace_back(std::log(kv.second.count) / std::log(2));
        topdown_curve[kv.first].emplace_back(std::log(fea.topdown_product) / std::log(2));
        bottomup_curve[kv.first].emplace_back(std::log(fea.bottomup_product) / std::log(2));
      }
    }
  }

  // sample relation in the curve
  auto sample_curve = [&](const std::vector<double>& x, const std::vector<double>& y,
                          double weight) {
    for (int i = 0; i < sample_n; i++) {
      double xx = i * weight;
      for (int j = static_cast<int>(x.size()) - 1; j >= 0; j--) {
        if (xx > x[j] - 1e-6) {
          ret_feature->emplace_back(y[j]);
          ret_feature->emplace_back(xx - x[j]);
          break;
        }
      }
    }
  };

  // serialize to frontend
  for (auto k : innermost_buffers) {
    std::vector<double>& count = count_curve[k];
    std::vector<double>& reuse = reuse_curve[k];
    std::vector<double>& top_down = topdown_curve[k];

    std::sort(count.begin(), count.end());
    std::sort(reuse.begin(), reuse.end());
    std::sort(top_down.begin(), top_down.end());

    sample_curve(count, reuse, 1);
    sample_curve(reuse, count, 1);
    sample_curve(count, top_down, 1);
    sample_curve(top_down, count, 1);
  }
}

// register API for front end
TVM_REGISTER_GLOBAL("autotvm.feature.GetItervarFeature")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Stmt stmt = args[0];
      bool take_log = args[1];
      Array<Array<Array<PrimExpr>>> ret_feature;

      GetItervarFeature(stmt, take_log, &ret_feature);

      *ret = ret_feature;
    });

TVM_REGISTER_GLOBAL("autotvm.feature.GetItervarFeatureFlatten")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Stmt stmt = args[0];
      bool take_log = args[1];
      std::vector<float> ret_feature;

      GetItervarFeatureFlatten(stmt, take_log, &ret_feature);

      TVMByteArray arr;
      arr.size = sizeof(float) * ret_feature.size();
      arr.data = reinterpret_cast<char*>(ret_feature.data());
      *ret = arr;
    });

TVM_REGISTER_GLOBAL("autotvm.feature.GetCurveSampleFeatureFlatten")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Stmt stmt = args[0];
      int sample_n = args[1];
      std::vector<float> ret_feature;

      GetCurveSampleFeatureFlatten(stmt, sample_n, &ret_feature);

      TVMByteArray arr;
      arr.size = sizeof(float) * ret_feature.size();
      arr.data = reinterpret_cast<char*>(ret_feature.data());
      *ret = arr;
    });

}  // namespace autotvm
}  // namespace tvm
