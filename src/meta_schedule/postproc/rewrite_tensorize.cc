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
#include <tvm/meta_schedule/postproc.h>

#include <algorithm>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::LoopRV;

void ApplyTensorization(const tir::Schedule& sch, const String& func_name,
                        const tir::PrimFuncNode* func, bool vectorize_init_loop) {
  std::vector<std::pair<std::string, std::function<void(tir::BlockRV)>>> jobs;

  tir::PostOrderVisit(func->body, [=, &jobs](const ObjectRef& obj) {
    if (const auto* block = obj.as<tir::BlockNode>()) {
      tir::StmtSRef block_sref = sch->GetSRef(block);
      if (Optional<String> intrin_name =
              tir::GetAnn<String>(block_sref, tir::attr::meta_schedule_auto_tensorize)) {
        std::string block_name = block_sref->StmtAs<tir::BlockNode>()->name_hint;
        if (block_name.find("init") == std::string::npos) {
          jobs.emplace_back(block_name, [sch, intrin_name](tir::BlockRV block) {
            try {
              sch->Tensorize(block, intrin_name.value());
            } catch (const std::exception& e) {
              LOG(WARNING) << "Tensorize failed with error " << e.what();
            }
          });
        } else if (vectorize_init_loop) {
          jobs.emplace_back(block_name, [sch](tir::BlockRV block) {
            Array<BlockRV> child_blocks = sch->GetChildBlocks(block);
            ICHECK(child_blocks.size() == 1);
            Array<LoopRV> init_loops = sch->GetLoops(child_blocks[0]);
            ICHECK(init_loops.size() == 1);
            sch->Vectorize(init_loops[0]);
          });
        }
      }
    }
  });

  for (auto kv : jobs) {
    tir::BlockRV block = sch->GetBlock(kv.first, func_name);
    sch->Unannotate(block, tir::attr::meta_schedule_auto_tensorize);
    kv.second(block);
  }
}

class RewriteTensorizeNode : public PostprocNode {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  bool Apply(const tir::Schedule& sch) final;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  bool vectorize_init_loop = false;

  static constexpr const char* _type_key = "meta_schedule.RewriteTensorize";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteTensorizeNode, PostprocNode);
};

bool RewriteTensorizeNode::Apply(const tir::Schedule& sch) {
  for (const auto& kv : sch->mod()->functions) {
    GlobalVar g_var = kv.first;
    BaseFunc base_func = kv.second;
    if (const tir::PrimFuncNode* prim_func = base_func.as<tir::PrimFuncNode>()) {
      ApplyTensorization(sch, g_var->name_hint, prim_func, vectorize_init_loop);
    }
  }
  return true;
}

Postproc Postproc::RewriteTensorize(bool vectorize_init_loop) {
  ObjectPtr<RewriteTensorizeNode> n = make_object<RewriteTensorizeNode>();
  n->vectorize_init_loop = vectorize_init_loop;
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteTensorizeNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteTensorize")
    .set_body_typed(Postproc::RewriteTensorize);

}  // namespace meta_schedule
}  // namespace tvm
