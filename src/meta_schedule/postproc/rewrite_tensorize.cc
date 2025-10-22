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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/meta_schedule/postproc.h>

#include <algorithm>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::LoopRV;

void CollectTensorizationJobs(
    const tir::Schedule& sch, const ffi::String& func_name, const tir::PrimFuncNode* func,
    bool vectorize_init_loop,
    std::vector<std::tuple<ffi::String, ffi::String, std::function<void(tir::BlockRV)>>>* jobs) {
  tir::PostOrderVisit(func->body, [=, &jobs](const ObjectRef& obj) {
    if (const auto* block = obj.as<tir::BlockNode>()) {
      tir::StmtSRef block_sref = sch->GetSRef(block);
      std::string block_name = block_sref->StmtAs<tir::BlockNode>()->name_hint;
      if (ffi::Optional<ffi::String> intrin_name =
              tir::GetAnn<ffi::String>(block_sref, tir::attr::meta_schedule_auto_tensorize)) {
        if (intrin_name.value() != "") {
          jobs->emplace_back(block_name, func_name, [sch, intrin_name](tir::BlockRV block) {
            try {
              sch->Tensorize(block, intrin_name.value());
            } catch (const std::exception& e) {
              LOG(WARNING) << "Tensorize failed with error " << e.what();
            }
          });
        } else if (block_name.find("init") && vectorize_init_loop) {
          jobs->emplace_back(block_name, func_name, [sch](tir::BlockRV block) {
            ffi::Array<BlockRV> child_blocks = sch->GetChildBlocks(block);
            ICHECK(child_blocks.size() == 1);
            ffi::Array<LoopRV> init_loops = sch->GetLoops(child_blocks[0]);
            ICHECK(init_loops.size() == 1);
            sch->Vectorize(init_loops[0]);
          });
        }
      }
    }
  });
}

class RewriteTensorizeNode : public PostprocNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RewriteTensorizeNode>();
  }

  void InitializeWithTuneContext(const TuneContext& context) final {}

  bool Apply(const tir::Schedule& sch) final;

  Postproc Clone() const {
    ObjectPtr<RewriteTensorizeNode> n = ffi::make_object<RewriteTensorizeNode>(*this);
    return Postproc(n);
  }

  bool vectorize_init_loop = false;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.RewriteTensorize", RewriteTensorizeNode,
                                    PostprocNode);
};

bool RewriteTensorizeNode::Apply(const tir::Schedule& sch) {
  // The rewriting jobs, 3-tuple (block_name, func_name, job_func)
  std::vector<std::tuple<ffi::String, ffi::String, std::function<void(tir::BlockRV)>>> jobs;
  for (const auto& kv : sch->mod()->functions) {
    GlobalVar g_var = kv.first;
    BaseFunc base_func = kv.second;
    if (const tir::PrimFuncNode* prim_func = base_func.as<tir::PrimFuncNode>()) {
      CollectTensorizationJobs(sch, g_var->name_hint, prim_func, vectorize_init_loop, &jobs);
    }
  }
  for (const auto& job : jobs) {
    const ffi::String& block_name = std::get<0>(job);
    const ffi::String& func_name = std::get<1>(job);
    const auto& job_func = std::get<2>(job);
    BlockRV block = sch->GetBlock(block_name, func_name);
    sch->Unannotate(block, tir::attr::meta_schedule_auto_tensorize);
    job_func(block);
  }
  return true;
}

Postproc Postproc::RewriteTensorize(bool vectorize_init_loop) {
  ObjectPtr<RewriteTensorizeNode> n = ffi::make_object<RewriteTensorizeNode>();
  n->vectorize_init_loop = vectorize_init_loop;
  return Postproc(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RewriteTensorizeNode::RegisterReflection();
  refl::GlobalDef().def("meta_schedule.PostprocRewriteTensorize", Postproc::RewriteTensorize);
}

}  // namespace meta_schedule
}  // namespace tvm
