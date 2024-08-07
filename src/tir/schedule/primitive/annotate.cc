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
#include "../utils.h"

namespace tvm {
namespace tir {

void Annotate(ScheduleState self, const StmtSRef& sref, const String& ann_key,
              const ObjectRef& ann_val) {
  // Extract annotation
  const Map<String, ObjectRef>* annotations = nullptr;
  if (const auto* loop = sref->StmtAs<ForNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->StmtAs<BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Check if the annotation already exists
  if (annotations->find(ann_key) != annotations->end()) {
    return;
  }
  // Add the new annotation
  Map<String, ObjectRef> new_ann(*annotations);
  new_ann.Set(ann_key, ann_val);
  // Create the new stmt
  if (const auto* loop = sref->StmtAs<ForNode>()) {
    ObjectPtr<ForNode> n = make_object<ForNode>(*loop);
    n->annotations = std::move(new_ann);
    self->Replace(sref, For(n), {});
  } else if (const auto* block = sref->StmtAs<BlockNode>()) {
    ObjectPtr<BlockNode> n = make_object<BlockNode>(*block);
    n->annotations = std::move(new_ann);
    Block p(n);
    self->Replace(sref, p, {{GetRef<Block>(block), p}});
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

void Unannotate(ScheduleState self, const StmtSRef& sref, const String& ann_key) {
  // Extract annotation
  const Map<String, ObjectRef>* annotations = nullptr;
  if (const auto* loop = sref->StmtAs<ForNode>()) {
    annotations = &loop->annotations;
  } else if (const auto* block = sref->StmtAs<BlockNode>()) {
    annotations = &block->annotations;
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
  }
  // Remove the annotation
  ICHECK(annotations->find(ann_key) != annotations->end())
      << "IndexError: Cannot find annotation key: " << ann_key;
  Map<String, ObjectRef> new_ann(*annotations);
  new_ann.erase(ann_key);
  // Create the new stmt
  if (const auto* loop = sref->StmtAs<ForNode>()) {
    ObjectPtr<ForNode> n = make_object<ForNode>(*loop);
    n->annotations = std::move(new_ann);
    self->Replace(sref, For(n), {});
  } else if (const auto* block = sref->StmtAs<BlockNode>()) {
    ObjectPtr<BlockNode> n = make_object<BlockNode>(*block);
    n->annotations = std::move(new_ann);
    Block p(n);
    self->Replace(sref, p, {{GetRef<Block>(block), p}});
  } else {
    LOG(FATAL) << "TypeError: Unknown type of sref: " << sref->stmt->GetTypeKey();
    throw;
  }
}

struct AnnotateTraits : public UnpackedInstTraits<AnnotateTraits> {
  static constexpr const char* kName = "Annotate";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, ObjectRef block_or_loop_rv, ObjectRef ann_val,
                                      String ann_key) {
    if (auto block = block_or_loop_rv.as<BlockRV>()) {
      return sch->Annotate(block.value(), ann_key, ann_val);
    }
    if (auto loop = block_or_loop_rv.as<LoopRV>()) {
      return sch->Annotate(loop.value(), ann_key, ann_val);
    }
    LOG(FATAL) << "TypeError: Expected Block or Loop, but gets: " << block_or_loop_rv->GetTypeKey();
    throw;
  }

  static String UnpackedAsPython(Array<String> outputs, ObjectRef block_or_loop_rv,
                                 ObjectRef ann_val, String ann_key) {
    PythonAPICall py("annotate");
    py.Input("block_or_loop", block_or_loop_rv);
    py.Input("ann_key", ann_key);
    py.Input("ann_val", ann_val);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct UnannotateTraits : public UnpackedInstTraits<UnannotateTraits> {
  static constexpr const char* kName = "Unannotate";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, ObjectRef block_or_loop_rv, String ann_key) {
    if (auto block = block_or_loop_rv.as<BlockRV>()) {
      return sch->Unannotate(block.value(), ann_key);
    }
    if (auto loop = block_or_loop_rv.as<LoopRV>()) {
      return sch->Unannotate(loop.value(), ann_key);
    }
    LOG(FATAL) << "TypeError: Expected Block or Loop, but gets: " << block_or_loop_rv->GetTypeKey();
    throw;
  }

  static String UnpackedAsPython(Array<String> outputs, ObjectRef block_or_loop_rv,
                                 String ann_key) {
    PythonAPICall py("unannotate");
    py.Input("block_or_loop", block_or_loop_rv);
    py.Input("ann_key", ann_key);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(AnnotateTraits);
TVM_REGISTER_INST_KIND_TRAITS(UnannotateTraits);

}  // namespace tir
}  // namespace tvm
