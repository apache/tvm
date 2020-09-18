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
 * \file src/tvm/relay/transforms/new_quantize.cc
 * \brief Relay Quantization related passes
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../ir/dataflow_matcher.h"

namespace tvm {
namespace relay {
namespace quantize {

class PatternCalibrationInfoNode : public Object {  // Change name later
 public:
  DFPattern pattern;
  Expr expr;

  Array<Array<Var>> input_scale_zps;
  Array<Integer> input_idxs;
  Integer output_idx;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("expr", &expr);
    v->Visit("input_scale_zps", &input_scale_zps);
    v->Visit("input_idxs", &input_idxs);
    v->Visit("output_idx", &output_idx);
  }

  static constexpr const char* _type_key = "PatternCalibrationInfoNode";
  TVM_DECLARE_BASE_OBJECT_INFO(PatternCalibrationInfoNode, Object);
};

class PatternCalibrationInfo : public ObjectRef {
 public:
  TVM_DLL PatternCalibrationInfo(DFPattern pattern, Expr expr, Array<Array<Var>> input_scale_zps,
                                 Array<Integer> input_idxs, Integer output_idx);
  TVM_DEFINE_OBJECT_REF_METHODS(PatternCalibrationInfo, ObjectRef, PatternCalibrationInfoNode);
};

PatternCalibrationInfo::PatternCalibrationInfo(DFPattern pattern, Expr expr,
                                               Array<Array<Var>> input_scale_zps,
                                               Array<Integer> input_idxs, Integer output_idx) {
  ObjectPtr<PatternCalibrationInfoNode> n = make_object<PatternCalibrationInfoNode>();
  n->pattern = std::move(pattern);
  n->expr = std::move(expr);
  n->input_scale_zps = std::move(input_scale_zps);
  n->input_idxs = std::move(input_idxs);
  n->output_idx = std::move(output_idx);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PatternCalibrationInfoNode);

TVM_REGISTER_GLOBAL("relay.new_quantize.PatternCalibrationInfo")
    .set_body_typed([](DFPattern pattern, Expr expr, Array<Array<Var>> input_scale_zps,
                       Array<Integer> input_idxs, Integer output_idx) {
      return PatternCalibrationInfo(pattern, expr, input_scale_zps, input_idxs, output_idx);
    });

class PartitionOutputs : public MixedModeMutator {
 public:
  Expr GetPartitionOutputs(const Expr& expr) {
    new_outputs.clear();
    if (auto func = expr.as<FunctionNode>()) {
      new_outputs.push_back(func->body);
    } else if (auto tuple = expr.as<TupleNode>()) {
      new_outputs = tuple->fields;  // Do I need to copy this explicitly?
    } else {
      new_outputs.push_back(expr);
    }
    VisitExpr(expr);
    Expr out;
    if (auto func = expr.as<FunctionNode>()) {
      out = Function(func->params, Tuple(new_outputs), Type{}, Array<TypeVar>{}, func->attrs);
    } else {
      out = Tuple(new_outputs);
    }
    return out;
  }

 protected:
  Expr Rewrite_(const CallNode* pre, const Expr& post) {
    auto* post_node = post.as<CallNode>();
    ICHECK(post_node != nullptr);
    if (auto* func_node = post_node->op.as<FunctionNode>()) {
      if (func_node->attrs.defined() &&
          func_node->attrs->dict.count(attr::kPartitionedFromPattern) != 0) {
        for (const auto& arg : post_node->args) {
          new_outputs.push_back(arg);
        }
        new_outputs.push_back(post);
      }
    }
    return post;
  }

  Array<Expr> new_outputs;
};

class PartitionsInOrder : protected MixedModeVisitor {
 public:
  PartitionsInOrder(bool skip_first, bool skip_last)
      : skip_first_(skip_first), skip_last_(skip_last) {}
  Array<Expr> partitions;
  bool skip_first_;
  bool skip_last_;
  Array<Expr> GetPartitionsInOrder(const Expr& expr) {
    VisitExpr(expr);
    Array<Expr> out;
    if (partitions.size() > 0) {
      if (skip_first_) {
        out.push_back(partitions[0]);
      }
      if (skip_last_) {
        out.push_back(partitions.back());
      }
    }
    return out;
  }
  void VisitExpr_(const CallNode* op) override {
    if (auto func_node = op->op.as<FunctionNode>()) {
      // If it's calling a function, check to see if it has attributes that it's a been partitioned
      // from a Pattern
      if (func_node->attrs.defined() &&
          func_node->attrs->dict.count(attr::kPartitionedFromPattern) != 0) {
        // If this is a pattern function, create a matcher on it's body
        partitions.push_back(op->op);
      }
    }
  }
};

class RewritePartitions : protected MixedModeMutator {
 public:
  RewritePartitions(const Array<DFPatternCallback>& callbacks) : callbacks_(callbacks) {}
  Map<String, ObjectRef> Rewrite(const Expr& expr) {
    // Preprocessing
    if (auto* func = expr.as<FunctionNode>()) {
      if (auto* tuple = func->body.as<TupleNode>()) {
        orig_outputs_ = tuple->fields;
      } else {
        orig_outputs_.push_back(func->body);
      }
      for (auto param : func->params) {
        new_params_.push_back(param);
      }
    } else {
      if (auto* tuple = expr.as<TupleNode>()) {
        orig_outputs_ = tuple->fields;
      } else {
        orig_outputs_.push_back(expr);
      }
    }
    Expr new_out = MixedModeMutator::Mutate(expr);

    // Add new parameters to the function
    if (auto* new_out_func = new_out.as<FunctionNode>()) {
      new_out =
          Function(new_params_, new_out_func->body, Type{}, Array<TypeVar>{}, new_out_func->attrs);
    }
    // TVM object system doesn't have pairs, so we'll return new_out and infos_ in a Map
    Map<String, ObjectRef> out_pair = {{"new_out", new_out}, {"infos_", infos_}};
    return out_pair;  //{new_out, infos_};
  }

 protected:
  Array<Var> FindScaleZp(const Expr& input, const Expr& new_body) {
    Array<Var> ScaleZp;
    auto x = WildcardPattern(make_object<WildcardPatternNode>());
    auto scale = WildcardPattern(make_object<WildcardPatternNode>());
    auto zp = WildcardPattern(make_object<WildcardPatternNode>());
    DFPattern pattern = IsOp("qnn.quantize")({x, scale, zp});

    runtime::PackedFunc callback([&](TVMArgs args, TVMRetValue* ret) {
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];

      if (node_map[x][0] == input) {
        auto scale_var = node_map[scale][0].as<VarNode>();
        auto zp_var = node_map[zp][0].as<VarNode>();

        CHECK((scale_var && zp_var) || (!scale_var && !zp_var))
            << "The scale and zero point passed to a "
            << "qnn.quantize must both be expressions composed of other variables, or be variables "
               "themselves. "
            << "Please change the AST returned from your QuantizerPattern to meet this "
               "requirement.";

        // Only add them to the list of scales / zps we will set later if they are not expressions
        if (scale_var && zp_var) {
          ScaleZp.push_back(GetRef<Var>(scale_var));
          ScaleZp.push_back(GetRef<Var>(zp_var));
        }
      }

      *ret = post;
    });
    RewritePatterns({DFPatternCallback(pattern, callback, false)}, new_body);
    return ScaleZp;
  }
  Expr Rewrite_(const CallNode* pre, const Expr& post) {
    // Cast the post as a call node and assert it actually is a call
    auto* post_node = post.as<CallNode>();
    ICHECK(post_node != nullptr);
    // Check to see if the Call is calling a Function
    if (auto* func_node = post_node->op.as<FunctionNode>()) {
      // If it's calling a function, check to see if it has attributes that it's a been partitioned
      // from a Pattern
      if (func_node->attrs.defined() &&
          func_node->attrs->dict.count(attr::kPartitionedFromPattern) != 0) {
        // If this is a pattern function, create a matcher on it's body
        auto matcher = DFPatternMatcher(func_node->body);
        // Find the callback that matches this pattern
        for (const auto& callback : callbacks_) {
          if (matcher.Match(callback->pattern, func_node->body)) {
            // extract the current params and call-level args
            Array<Var> params = func_node->params;
            Array<Expr> call_args = post_node->args;

            Array<Integer> input_idx;
            // Get the indices of the arguments to this function in the output tuple
            for (auto arg : pre->args) {
              auto itr = std::find(orig_outputs_.begin(), orig_outputs_.end(), arg);
              ICHECK(itr != orig_outputs_.end())
                  << "Didn't find the arguement in the output tuple. Indicates a possible problem "
                     "in PartitionOutputs. ";
              input_idx.push_back(std::distance(orig_outputs_.begin(), itr));
            }
            // Get the index of the output of this function
            auto itr = std::find(orig_outputs_.begin(), orig_outputs_.end(), GetRef<Expr>(pre));
            ICHECK(itr != orig_outputs_.end())
                << "Didn't find the output in the output tuple. Indicates a possible problem in "
                   "PartitionOutputs. ";
            Integer output_idx(std::distance(orig_outputs_.begin(), itr));

            // create a new body based on the callback
            Expr new_body = callback->function(pre->op.as<FunctionNode>()->body, func_node->body,
                                               matcher.GetMemo());

            // FIND THE SCALE / ZPS
            Array<Array<Var>> input_scale_zps;
            for (auto param : params) {
              Array<Var> scale_zp = FindScaleZp(param, new_body);
              // If FindScaleZp returns an empty array, we don't need to provide these as parameters
              if (scale_zp.size() != 0) {
                ICHECK(scale_zp.size() == 2)
                    << "scale_zp should have two items in it, the scale variable and the zp "
                       "variable. This points to an issue with FindScaleZp. ";
                input_scale_zps.push_back(FindScaleZp(param, new_body));
              }
            }

            infos_.push_back(PatternCalibrationInfo(callback->pattern,
                                                    pre->op.as<FunctionNode>()->body,
                                                    input_scale_zps, input_idx, output_idx));
            // find parameters added to the new body that weren't there before
            // find all of the free variables in the new body
            for (const auto& param : FreeVars(new_body)) {
              // check to see if that free variable is in the old parameter list
              if (std::find(params.begin(), params.end(), param) == params.end()) {
                // if not, add it to the new parameter list
                params.push_back(param);
                // Create a new call-level arg for it
                // Make that new arg an input to the top-level function
                new_params_.push_back(Var(param->name_hint(), param->type_annotation));
                call_args.push_back(new_params_.back());
              }
            }
            // Create a new function with new params and body
            Expr new_func = Function(params, new_body, Type{}, Array<TypeVar>{}, func_node->attrs);
            // Call the new function with the new args
            return Call(new_func, call_args, Attrs{}, Array<Type>{});
          }
        }
      }
    }
    return post;
  }
  Array<DFPatternCallback> callbacks_;
  Array<PatternCalibrationInfo> infos_;
  Array<Var> new_params_;
  Array<Expr> orig_outputs_;
};

class ReplaceArgs : protected MixedModeMutator {
 public:
  Expr Rewrite(const Expr& body,
               std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> arg_map) {
    // Leverage the memoizer to replace parameters with arguments automatically
    memo_ = arg_map;
    return MixedModeMutator::Mutate(body);
  }
};

class LowerPartitions : protected MixedModeMutator {
 public:
  LowerPartitions(const Array<Expr> targets = Array<Expr>(), const bool skipping_partitions = false)
      : targets_(targets), skipping_partitions_(skipping_partitions) {}
  Expr Rewrite(const Expr& expr) {
    Expr new_out = MixedModeMutator::Mutate(expr);
    return new_out;
  }
  Expr Rewrite_(const CallNode* pre, const Expr& post) {
    // Targets is usually length 0, 1, or 2
    if ((!skipping_partitions_) ||
        (skipping_partitions_ &&
         std::find(targets_.begin(), targets_.end(), pre->op) != targets_.end())) {
      auto* post_node = post.as<CallNode>();
      ICHECK(post_node != nullptr);
      if (auto* func_node = post_node->op.as<FunctionNode>()) {
        // If the function was created by the pattern matcher, remove it
        if (func_node->attrs.defined() &&
            func_node->attrs->dict.count(attr::kPartitionedFromPattern) != 0) {
          std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> arg_map = {};
          Array<Expr> args = post_node->args;
          Array<Var> params = func_node->params;

          for (uint i = 0; i < args.size(); i++) {
            arg_map.insert({params[i], args[i]});
          }
          return ReplaceArgs().Rewrite(func_node->body, arg_map);
        }
      }
    }
    return post;
  }

 protected:
  Array<Expr> targets_;
  bool skipping_partitions_;
};

TVM_REGISTER_GLOBAL("relay.transform.quantize.partition_outputs").set_body_typed([](const Expr& expr) {
  return PartitionOutputs().GetPartitionOutputs(expr);
});
TVM_REGISTER_GLOBAL("relay.transform.quantize.rewrite_partitions")
    .set_body_typed([](const Array<DFPatternCallback>& callbacks, const Expr& expr) {
      return RewritePartitions(callbacks).Rewrite(expr);
    });
TVM_REGISTER_GLOBAL("relay.transform.quantize.lower_partitions").set_body_typed([](const Expr& expr) {
  return LowerPartitions().Rewrite(expr);
});
TVM_REGISTER_GLOBAL("relay.transform.quantize.skip_partitions")
    .set_body_typed([](const Expr& expr, bool skip_first, bool skip_last) {
      auto targets = PartitionsInOrder(skip_first, skip_last).GetPartitionsInOrder(expr);
      return LowerPartitions(targets, true).Rewrite(expr);
    });
}  // namespace quantize
}  // namespace relay
}  // namespace tvm
