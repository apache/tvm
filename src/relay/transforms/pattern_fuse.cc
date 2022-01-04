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
 * \file src/relay/transforms/pattern_fuse.cc
 * \brief A pass to apply fuse_ops based on hierarchical DFPattern-based approach.
 *
 * Currently fuse_ops (i.e., src/relay/transforms/fuse_ops.cc) are legacy pass originate from the
 * early days of TVM. At this point no helpers exist to simplify mutations in the relay IR. Also,
 * hardware constrains in terms of hardware units were constraint. Also, original fuse_ops rely on
 * OpPatternKind of each operator, as well as a hardcoded upper bound of the maximum number of ops
 * to be fused together in a single kernel.
 *
 * This pass is stepping stone to leverage a more modern TVM approach to mutate graphs in the relay
 * level. It uses the pattern language to express fusable patterns.
 *
 *
 * Now the patterns are applied in hierarchical order as in \p Composer, and eventually will be
 * split in 2 different phases respecting the legacy iterative process to fuse ops. These phases are
 * the following:
 *
 * Phase 0
 * -------
 * Apply dominant/high-performing patterns, such as kOutEWiseFusable -> BroadCast* -> Elemwise
 * (i.e., conv2->add->mul->relu) or kInjective* (i.e., exp*).
 * Note: * operator on the given state denotes an IsUpTo operator, where the upper bound limit is
 * the max number of ops to be fused to together.
 *
 * Phase 1:
 * -------
 * The second phase as TVM lowers a fused function, it expects all arguments to be a Tensor or a
 * tuple containing only Tensors. But this tuple may contain a reference or another tuple. To avoid
 * modifying codegen logic, we do not allow fusing through this node if the tuple contains such non
 * Tensor fields.
 * So, injective ops need to be fused into intermediate tuples.
 */

#include "./pattern_fuse.h"

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <memory>
#include <utility>
#include <vector>

#include "../op/tensor/transform.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {
static const Op& stop_fusion_op = Op::Get("annotation.stop_fusion");

static const int DEPTH_OF_FUSED_OPS = 256;

TVM_REGISTER_PASS_CONFIG_OPTION("relay.PatternFuse.max_depth", Integer);

/*!
 *  General fusor is to handle possible diamond shape branches, in the following graph, conv2d can
 * be fused to elemwise add.
 *
 *           conv2d
 *          /  |  \
 *         /   |   \
 *       op    op   op
 *        \    |    /
 *         \   |   /
 *        elemwise add
 *             |
 */
class GeneralFusorPattern {
 public:
  GeneralFusorPattern() {
    attrs_elemwise.Set("TOpPattern", Integer(static_cast<int>(kElemWise)));
    attrs_broad.Set("TOpPattern", Integer(static_cast<int>(kBroadcast)));
    attrs_kfusable.Set("TOpPattern", Integer(static_cast<int>(kOutEWiseFusable)));

    x = IsWildcard();
    y = IsWildcard();

    k_elemwise_fusable = IsWildcard().HasAttr(attrs_kfusable)({x, y});

    broadcast_op = IsWildcard().HasAttr(attrs_broad)({k_elemwise_fusable, IsWildcard()});

    for (int idx = 0; idx < DEPTH_OF_FUSED_OPS; idx++) {
      broadcast_op =
          broadcast_op || IsWildcard().HasAttr(attrs_broad_next)({IsWildcard(), broadcast_op});
    }

    elemwise_op = IsWildcard().HasAttr(attrs_elemwise);

    pattern_ = elemwise_op({broadcast_op});
  }

  String GetPatternID() { return "GeneralFusorPattern"; }

  DFPattern GetPattern() { return pattern_; }

 private:
  Map<String, ObjectRef> attrs_elemwise, attrs_broad, attrs_kfusable, attrs_broad_next;

  DFPattern pattern_;
  DFPattern k_elemwise_fusable;
  DFPattern broadcast_op;
  DFPattern elemwise_op;
  DFPattern x;
  DFPattern y;
};

class DomPattern {
 public:
  DomPattern() {
    attrs_elemwise.Set("TOpPattern", Integer(static_cast<int>(kElemWise)));
    attrs_broad.Set("TOpPattern", Integer(static_cast<int>(kBroadcast)));
    attrs_kfusable.Set("TOpPattern", Integer(static_cast<int>(kOutEWiseFusable)));

    x = IsWildcard();
    y = IsWildcard();

    k_elemwise_fusable = IsWildcard().HasAttr(attrs_kfusable)({x, y});

    broadcast_op = IsWildcard().HasAttr(attrs_broad)({IsWildcard(), IsWildcard()});

    for (int idx = 0; idx < DEPTH_OF_FUSED_OPS; idx++) {
      broadcast_op =
          broadcast_op || IsWildcard().HasAttr(attrs_broad_next)({IsWildcard(), broadcast_op});
    }

    elemwise_op = IsWildcard().HasAttr(attrs_elemwise)({broadcast_op});
    // elemwise_op = IsOp("nn.relu")({IsWildcard});

    pattern_ = elemwise_op;
  }

  String GetPatternID() { return "GeneralFusorPattern"; }

  DFPattern GetPattern() { return pattern_; }

 private:
  Map<String, ObjectRef> attrs_elemwise, attrs_broad, attrs_kfusable, attrs_broad_next;

  DFPattern pattern_;
  DFPattern k_elemwise_fusable;
  DFPattern broadcast_op;
  DFPattern elemwise_op;
  DFPattern x;
  DFPattern y;
};

class DomPatternKElemwise {
 public:
  /*! Trying to get into the following
  // kOutEWiseFusable --> KBroadCast* --> Elemwise* */
  DomPatternKElemwise() {
    attrs_elemwise.Set("TOpPattern", Integer(static_cast<int>(kElemWise)));
    attrs_broad.Set("TOpPattern", Integer(static_cast<int>(kBroadcast)));
    attrs_kfusable.Set("TOpPattern", Integer(static_cast<int>(kOutEWiseFusable)));

    elemwise_op = IsWildcard().HasAttr(attrs_elemwise)({IsWildcard()});

    broadcast_op = IsWildcard().HasAttr(attrs_broad)({IsWildcard(), IsWildcard()});

    pattern_ = DominatorPattern(k_elemwise_fusable, elemwise_op, broadcast_op);
  }

  String GetPatternID() { return "GeneralFusorPattern"; }

  DFPattern GetPattern() { return pattern_; }

 private:
  Map<String, ObjectRef> attrs_elemwise, attrs_broad, attrs_kfusable, attrs_broad_next;

  DFPattern pattern_;
  DFPattern k_elemwise_fusable;
  DFPattern broadcast_op;
  DFPattern elemwise_op;
  DFPattern x;
  DFPattern y;
};

class KElemewiseBr {
 public:
  KElemewiseBr() {
    attrs_elemwise.Set("TOpPattern", Integer(static_cast<int>(kElemWise)));
    attrs_broad.Set("TOpPattern", Integer(static_cast<int>(kBroadcast)));
    attrs_kfusable.Set("TOpPattern", Integer(static_cast<int>(kOutEWiseFusable)));

    // x = IsVar(IsWildcard());
    // y = IsVar(IsWildcard());

    DFPattern var = IsVar("x");

    k_elemwise_fusable = IsWildcard().HasAttr(attrs_kfusable);

    broadcast_op = IsWildcard().HasAttr(attrs_broad)({k_elemwise_fusable, IsWildcard()});

    for (int idx = 0; idx < DEPTH_OF_FUSED_OPS; idx++) {
      broadcast_op =
          broadcast_op || IsWildcard().HasAttr(attrs_broad_next)({IsWildcard(), broadcast_op});
    }

    // elemwise_op = IsWildcard().HasAttr(attrs_elemwise);

    pattern_ = broadcast_op;
  }

  String GetPatternID() { return "GeneralFusorPattern"; }

  DFPattern GetPattern() { return pattern_; }

 private:
  Map<String, ObjectRef> attrs_elemwise, attrs_broad, attrs_kfusable, attrs_broad_next;

  DFPattern pattern_;
  DFPattern k_elemwise_fusable;
  DFPattern broadcast_op;
  DFPattern elemwise_op;
  DFPattern x;
  DFPattern y;
};

class InjectiveOpsPattern {
 public:
  /*! Trying to get into the following
  // kOutEWiseFusable --> KBroadCast* --> Elemwise* */
  InjectiveOpsPattern() {
    attrs_elemwise.Set("TOpPattern", Integer(static_cast<int>(kElemWise)));
    attrs_elemwise_.Set("TOpPattern", Integer(static_cast<int>(kElemWise)));
    attrs_broad.Set("TOpPattern", Integer(static_cast<int>(kBroadcast)));

    kinjective_f = IsWildcard().HasAttr(attrs_elemwise)({IsWildcard()});
    kinjective_s = IsWildcard().HasAttr(attrs_elemwise_)({IsWildcard()});

    for (size_t i = 0; i < DEPTH_OF_FUSED_OPS; i++) {
      kinjective_f = kinjective_f || IsWildcard().HasAttr(attrs_kinjective)({kinjective_f});
      kinjective_s = kinjective_s || IsWildcard().HasAttr(attrs_kinjective_)({kinjective_s});
    }

    elemwise_op =
        IsWildcard().HasAttr(attrs_broad)({IsWildcard().HasAttr(attrs_elemwise)({IsWildcard()}),
                                           IsWildcard().HasAttr(attrs_elemwise_)({IsWildcard()})});

    pattern_ = elemwise_op;
  }

  String GetPatternID() { return "InjectivePattern"; }

  DFPattern GetPattern() { return pattern_; }

 private:
  Map<String, ObjectRef> attrs_elemwise, attrs_broad, attrs_kfusable, attrs_broad_next,
      attrs_kinjective, attrs_kinjective_, attrs_elemwise_;

  DFPattern pattern_;
  DFPattern k_elemwise_fusable;
  DFPattern broadcast_op;
  DFPattern elemwise_op;
  DFPattern kinjective_f;
  DFPattern kinjective_s;
};

class MaxDepthOfKElemwiseOps {
 public:
  MaxDepthOfKElemwiseOps() {
    attrs_elemwise.Set("TOpPattern", Integer(static_cast<int>(kElemWise)));
    attrs_elemwise_n.Set("TOpPattern", Integer(static_cast<int>(kElemWise)));

    in_var = IsWildcard();
    elemwise_op = IsWildcard().HasAttr(attrs_elemwise)({in_var});

    for (int idx = 0; idx < DEPTH_OF_FUSED_OPS; idx++) {
      elemwise_op = IsWildcard().HasAttr(attrs_elemwise_n)({elemwise_op}) || elemwise_op;
    }

    pattern_ = elemwise_op;
  }

  String GetPatternID() { return "MaxDepthOfKElemwiseOps"; }

  DFPattern GetPattern() { return pattern_; }

 private:
  Map<String, ObjectRef> attrs_elemwise, attrs_elemwise_n;
  DFPattern in_var;
  DFPattern pattern_;
  DFPattern elemwise_op;
};

Function InferType(const Function& expr, const IRModule& m) {
  IRModule mod(m);
  mod->Update(mod->GetGlobalVar("main"), expr);
  mod = transform::InferType()(mod);
  return Downcast<Function>(mod->Lookup("main"));
}

Expr FusePattern(const Function& func, const IRModule& mod) {
  Function merged_func = func;
  Map<String, ObjectRef> attrs;
  DFPatternPartitionComposer df_composer;
  // Phase 1
  df_composer.AddPattern(GeneralFusorPattern().GetPattern());
  df_composer.AddPattern(MaxDepthOfKElemwiseOps().GetPattern());
  // df_composer.AddPattern(InjectiveOpsPattern().GetPattern());
  // df_composer.AddPattern(KElemewiseBr().GetPattern());
  // Phase 2
  // Phase 3 : here we should merge into tuples

  return PartitionPattern(df_composer.GetPatterns(), merged_func, attrs, PackedFunc());
}

namespace transform {

Pass FuseWithPattern() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> fuse_pattern =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(FusePattern(f, m)); };

  auto fuse_with_pattern = CreateFunctionPass(fuse_pattern, 0, "FuseWithPattern", {"InferType"});

  return Sequential({fuse_with_pattern, AnnotatePostFuseFuncs()});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseWithPattern").set_body_typed(FuseWithPattern);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
