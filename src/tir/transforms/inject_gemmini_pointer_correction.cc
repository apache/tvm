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
 * \brief Correct pointer addresses in scratchpad and accumulator of Gemmini
 * \file inject_gemmini_pointer_correction.cc
 * \author Federico Peccia <https://fPecc.github.io/>
 */
#include <tvm/arith/analyzer.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

struct CorrectGemminisScratchpadAndAccumulatorPointersConfigNode
    : public tvm::AttrsNode<CorrectGemminisScratchpadAndAccumulatorPointersConfigNode> {
  int dim;

  TVM_DECLARE_ATTRS(CorrectGemminisScratchpadAndAccumulatorPointersConfigNode,
                    "tir.transform.CorrectGemminisScratchpadAndAccumulatorPointersConfig") {
    TVM_ATTR_FIELD(dim).describe("Systolic array DIM").set_default(16);
  }
};

class CorrectGemminisScratchpadAndAccumulatorPointersConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(
      CorrectGemminisScratchpadAndAccumulatorPointersConfig, Attrs,
      CorrectGemminisScratchpadAndAccumulatorPointersConfigNode);
};

TVM_REGISTER_NODE_TYPE(CorrectGemminisScratchpadAndAccumulatorPointersConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.CorrectGemminisScratchpadAndAccumulatorPointers",
                                CorrectGemminisScratchpadAndAccumulatorPointersConfig);

class CorrectGemminisScratchpadAndAccumulatorPointersInjector : public StmtExprMutator {
 public:
  explicit CorrectGemminisScratchpadAndAccumulatorPointersInjector(int dim) : dim_(dim) {}

  Stmt Inject(Stmt stmt) { return this->VisitStmt(stmt); }

  PrimExpr VisitExpr_(const CallNode* op) final {
    /*
    This pass is used to modify the access ptr
    */
    auto node = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (node->op.same_as(builtin::tvm_access_ptr())) {
      const VarNode* buffer = node->args[1].as<VarNode>();

      if (std::string(buffer->name_hint).find("local") != std::string::npos) {
        PrimExpr offset = this->VisitExpr(node->args[2]);
        PrimExpr extent = this->VisitExpr(node->args[3]);

        const auto* ptr_type = buffer->type_annotation.as<PointerTypeNode>();
        ICHECK(ptr_type) << "The provided variable is not of pointer type";
        auto scope = ptr_type->storage_scope;
        auto info = GetMemoryInfo(scope);
        ICHECK(info.defined()) << "Cannot find memory info of " << scope;
        DataType dtype = Downcast<PrimType>(ptr_type->element_type)->dtype;

        int div = dim_;

        PrimExpr inner_offset = indexmod(offset, extent);
        PrimExpr outer_offset = offset - inner_offset;
        PrimExpr outer_offset_corrected = indexdiv(outer_offset, div);
        PrimExpr offset_corrected = outer_offset_corrected + inner_offset;

        return Call(node->dtype, node->op,
                    {node->args[0], node->args[1], offset_corrected, extent, node->args[4]});
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 private:
  int dim_;
};

namespace transform {

Pass CorrectGemminisScratchpadAndAccumulatorPointers() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<CorrectGemminisScratchpadAndAccumulatorPointersConfig>(
        "tir.CorrectGemminisScratchpadAndAccumulatorPointers");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<CorrectGemminisScratchpadAndAccumulatorPointersConfig>();
    }
    n->body = CorrectGemminisScratchpadAndAccumulatorPointersInjector(cfg.value()->dim)
                  .Inject(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CorrectGemminisScratchpadAndAccumulatorPointers",
                            {});
}

TVM_REGISTER_GLOBAL("tir.transform.CorrectGemminisScratchpadAndAccumulatorPointers")
    .set_body_typed(CorrectGemminisScratchpadAndAccumulatorPointers);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
