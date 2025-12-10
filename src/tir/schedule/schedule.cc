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

#include "./utils.h"
namespace tvm {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ScheduleNode>();
  BlockRVNode::RegisterReflection();
  LoopRVNode::RegisterReflection();
}

/**************** Constructor ****************/

BlockRV::BlockRV() { this->data_ = ffi::make_object<BlockRVNode>(); }

LoopRV::LoopRV() { this->data_ = ffi::make_object<LoopRVNode>(); }

/**************** GetSRef ****************/

StmtSRef ScheduleNode::GetSRef(const StmtNode* stmt) const {
  ScheduleState state = this->state();
  auto it = state->stmt2ref.find(stmt);
  if (it == state->stmt2ref.end()) {
    LOG(FATAL) << "IndexError: The stmt doesn't exist in the IR";
  }
  return it->second;
}

/**************** FFI ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleGetMod", &ScheduleNode::mod)
      .def_method("tir.schedule.ScheduleGetState", &ScheduleNode::state)
      .def_method("tir.schedule.ScheduleGetTrace", &ScheduleNode::trace)
      .def_method("tir.schedule.ScheduleGetFuncWorkingOn", &ScheduleNode::func_working_on)
      .def_method("tir.schedule.ScheduleCopy", &ScheduleNode::Copy)
      .def_method("tir.schedule.ScheduleSeed", &ScheduleNode::Seed)
      .def_method("tir.schedule.ScheduleForkSeed", &ScheduleNode::ForkSeed)
      .def_method("tir.schedule.ScheduleWorkOn", &ScheduleNode::WorkOn);
}

/**************** (FFI) Constructor ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tir.schedule.BlockRV", []() { return BlockRV(); })
      .def("tir.schedule.LoopRV", []() { return LoopRV(); })
      .def("tir.schedule.ConcreteSchedule",
           [](IRModule mod, support::LinearCongruentialEngine::TRandState seed, int debug_mask,
              int error_render_level, bool enable_check) -> Schedule {
             return Schedule::Concrete(mod, debug_mask, seed,
                                       static_cast<ScheduleErrorRenderLevel>(error_render_level),
                                       enable_check);
           })
      .def("tir.schedule.TracedSchedule",
           [](IRModule mod, support::LinearCongruentialEngine::TRandState seed, int debug_mask,
              int error_render_level, bool enable_check) -> Schedule {
             return Schedule::Traced(mod, seed, debug_mask,
                                     static_cast<ScheduleErrorRenderLevel>(error_render_level),
                                     enable_check);
           });
}

/******** (FFI) Lookup random variables ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tir.schedule.ScheduleGet",
           [](Schedule self, ObjectRef obj) -> ObjectRef {
             if (auto loop_rv = obj.as<LoopRV>()) {
               return self->Get(loop_rv.value());
             }
             if (auto block_rv = obj.as<BlockRV>()) {
               return self->Get(block_rv.value());
             }
             if (auto expr_rv = obj.as<ExprRV>()) {
               return self->Get(expr_rv.value());
             }
             LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: "
                        << obj->GetTypeKey() << ". Its value is: " << obj;
             throw;
           })
      .def("tir.schedule.ScheduleGetSRef",
           [](Schedule self, ObjectRef obj) -> ffi::Optional<ObjectRef> {
             if (auto loop_rv = obj.as<LoopRV>()) {
               return self->GetSRef(loop_rv.value());
             }
             if (auto block_rv = obj.as<BlockRV>()) {
               return self->GetSRef(block_rv.value());
             }
             if (auto stmt = obj.as<Stmt>()) {
               return self->GetSRef(stmt.value());
             }
             LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
             throw;
           })
      .def("tir.schedule.ScheduleRemoveRV", [](Schedule self, ObjectRef obj) -> void {
        if (auto loop_rv = obj.as<LoopRV>()) {
          return self->RemoveRV(loop_rv.value());
        }
        if (auto block_rv = obj.as<BlockRV>()) {
          return self->RemoveRV(block_rv.value());
        }
        if (auto expr_rv = obj.as<ExprRV>()) {
          return self->RemoveRV(expr_rv.value());
        }
        LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
        throw;
      });
}

/******** (FFI) Sampling ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleSampleCategorical", &ScheduleNode::SampleCategorical)
      .def_method("tir.schedule.ScheduleSamplePerfectTile", &ScheduleNode::SamplePerfectTile)
      .def_method("tir.schedule.ScheduleSamplePartitionedTile",
                  &ScheduleNode::SamplePartitionedTile)
      .def_method("tir.schedule.ScheduleSampleComputeLocation",
                  &ScheduleNode::SampleComputeLocation);
}
/******** (FFI) Get blocks & loops ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleGetBlock", &ScheduleNode::GetBlock)
      .def_method("tir.schedule.ScheduleGetLoops", &ScheduleNode::GetLoops)
      .def("tir.schedule.ScheduleGetChildBlocks",
           [](Schedule self, ObjectRef rv) {
             if (auto block_rv = rv.as<BlockRV>()) {
               return self->GetChildBlocks(block_rv.value());
             }
             if (auto loop_rv = rv.as<LoopRV>()) {
               return self->GetChildBlocks(loop_rv.value());
             }
             LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: "
                        << rv->GetTypeKey() << ". Its value is: " << rv;
             throw;
           })
      .def_method("tir.schedule.ScheduleGetProducers", &ScheduleNode::GetProducers)
      .def_method("tir.schedule.ScheduleGetConsumers", &ScheduleNode::GetConsumers)
      .def_method("tir.schedule.ScheduleGetOutputBlocks", &ScheduleNode::GetOutputBlocks);
}
/******** (FFI) Transform loops ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleMerge", &ScheduleNode::Merge)
      .def_method("tir.schedule.ScheduleFuse", &ScheduleNode::Fuse)
      .def_method("tir.schedule.ScheduleSplit", &ScheduleNode::Split)
      .def_method("tir.schedule.ScheduleLoopPartition", &ScheduleNode::LoopPartition)
      .def_method("tir.schedule.ScheduleReorder", &ScheduleNode::Reorder)
      .def_method("tir.schedule.ScheduleReorderBlockIterVar", &ScheduleNode::ReorderBlockIterVar)
      .def("tir.schedule.ScheduleAddUnitLoop", [](Schedule self, ObjectRef rv) -> LoopRV {
        if (auto loop_rv = rv.as<LoopRV>()) {
          return self->AddUnitLoop(loop_rv.value());
        } else if (auto block_rv = rv.as<BlockRV>()) {
          return self->AddUnitLoop(block_rv.value());
        } else {
          LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: "
                     << rv->GetTypeKey() << ". Its value is: " << rv;
          throw;
        }
      });
}
/******** (FFI) Manipulate ForKind ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleParallel", &ScheduleNode::Parallel)
      .def_method("tir.schedule.ScheduleVectorize", &ScheduleNode::Vectorize)
      .def_method("tir.schedule.ScheduleBind", &ScheduleNode::Bind)
      .def_method("tir.schedule.ScheduleUnroll", &ScheduleNode::Unroll);
}
/******** (FFI) Insert cache stages ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleCacheRead", &ScheduleNode::CacheRead)
      .def_method("tir.schedule.ScheduleCacheWrite", &ScheduleNode::CacheWrite)
      .def_method("tir.schedule.ScheduleReindexCacheRead", &ScheduleNode::ReindexCacheRead)
      .def_method("tir.schedule.ScheduleReindexCacheWrite", &ScheduleNode::ReindexCacheWrite)
      .def_method("tir.schedule.ScheduleCacheInplace", &ScheduleNode::CacheInplace)
      .def_method("tir.schedule.ScheduleCacheIndex", &ScheduleNode::CacheIndex)
      .def("tir.schedule.ScheduleReIndex",
           [](Schedule self, const BlockRV& block_rv, int buffer_index, int buffer_index_type, bool skip_simplify) {
             return self->ReIndex(block_rv, buffer_index,
                                  static_cast<BufferIndexType>(buffer_index_type), skip_simplify);
           });
}
/******** (FFI) Data movement ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleReadAt", &ScheduleNode::ReadAt)
      .def_method("tir.schedule.ScheduleWriteAt", &ScheduleNode::WriteAt);
}
/******** (FFI) Compute location ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleComputeAt", &ScheduleNode::ComputeAt)
      .def_method("tir.schedule.ScheduleReverseComputeAt", &ScheduleNode::ReverseComputeAt)
      .def_method("tir.schedule.ScheduleComputeInline", &ScheduleNode::ComputeInline)
      .def_method("tir.schedule.ScheduleReverseComputeInline", &ScheduleNode::ReverseComputeInline)
      .def_method("tir.schedule.ScheduleFuseReductionEpilogue",
                  &ScheduleNode::FuseReductionEpilogue);
}
/******** (FFI) Reduction ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleDecomposeReduction", &ScheduleNode::DecomposeReduction)
      .def_method("tir.schedule.ScheduleRFactor", &ScheduleNode::RFactor);
}
/******** (FFI) Block annotation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleStorageAlign", &ScheduleNode::StorageAlign)
      .def_method("tir.schedule.ScheduleSetScope", &ScheduleNode::SetScope)
      .def_method("tir.schedule.ScheduleUnsafeSetDType", &ScheduleNode::UnsafeSetDType);
}
/******** (FFI) Blockize & Tensorize ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tir.schedule.ScheduleBlockize",
           [](Schedule self, ObjectRef target, bool preserve_unit_iters) {
             if (auto loop_rv = target.as<LoopRV>()) {
               return self->Blockize(loop_rv.value(), preserve_unit_iters);
             } else if (auto blocks = target.as<ffi::Array<BlockRV>>()) {
               return self->Blockize(blocks.value(), preserve_unit_iters);
             }
             LOG(FATAL) << "Unsupported target type: " << target->GetTypeKey();
           })
      .def("tir.schedule.ScheduleTensorize",
           [](Schedule self, ObjectRef rv, ffi::String intrin, bool preserve_unit_iters) {
             if (auto block_rv = rv.as<BlockRV>()) {
               self->Tensorize(block_rv.value(), intrin, preserve_unit_iters);
             } else if (auto loop_rv = rv.as<LoopRV>()) {
               self->Tensorize(loop_rv.value(), intrin, preserve_unit_iters);
             } else {
               LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: "
                          << rv->GetTypeKey() << ". Its value is: " << rv;
             }
           });
}

/******** (FFI) Annotation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tir.schedule.ScheduleAnnotate",
           [](Schedule self, ObjectRef rv, const ffi::String& ann_key, const Any& ann_val) {
             if (auto block_rv = rv.as<BlockRV>()) {
               return self->Annotate(block_rv.value(), ann_key, ann_val);
             }
             if (auto loop_rv = rv.as<LoopRV>()) {
               return self->Annotate(loop_rv.value(), ann_key, ann_val);
             }
             LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: "
                        << rv->GetTypeKey() << ". Its value is: " << rv;
             throw;
           })
      .def("tir.schedule.ScheduleUnannotate", [](Schedule self, ObjectRef rv,
                                                 const ffi::String& ann_key) {
        if (auto block_rv = rv.as<BlockRV>()) {
          return self->Unannotate(block_rv.value(), ann_key);
        }
        if (auto loop_rv = rv.as<LoopRV>()) {
          return self->Unannotate(loop_rv.value(), ann_key);
        }
        LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                   << ". Its value is: " << rv;
        throw;
      });
}

/******** (FFI) Layout transformation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tir.schedule.ScheduleTransformLayout",
           [](Schedule self, const BlockRV& block_rv, int buffer_index, int buffer_index_type,
              const IndexMap& index_map, const ffi::Optional<IndexMap>& pad_value,
              bool assume_injective_transform) {
             return self->TransformLayout(block_rv, buffer_index,
                                          static_cast<BufferIndexType>(buffer_index_type),
                                          index_map, pad_value, assume_injective_transform);
           })
      .def_method("tir.schedule.ScheduleTransformBlockLayout", &ScheduleNode::TransformBlockLayout)
      .def("tir.schedule.ScheduleSetAxisSeparator",
           [](Schedule self, const BlockRV& block_rv, int buffer_index, int buffer_index_type,
              const ffi::Array<IntImm>& axis_separators) {
             return self->SetAxisSeparator(block_rv, buffer_index,
                                           static_cast<BufferIndexType>(buffer_index_type),
                                           axis_separators);
           });
}

/******** (FFI) Padding decomposition ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleDecomposePadding", &ScheduleNode::DecomposePadding)
      .def_method("tir.schedule.SchedulePadEinsum", &ScheduleNode::PadEinsum);
}
/******** (FFI) Buffer transformation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_method("tir.schedule.ScheduleRollingBuffer", &ScheduleNode::RollingBuffer);
}
/******** (FFI) Misc ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("tir.schedule.ScheduleEnterPostproc", &ScheduleNode::EnterPostproc)
      .def_method("tir.schedule.ScheduleUnsafeHideBufferAccess",
                  &ScheduleNode::UnsafeHideBufferAccess);
}
/******** (FFI) Annotate buffer access ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.schedule.ScheduleAnnotateBufferAccess",
                        [](Schedule self, const BlockRV& block_rv, int buffer_index,
                           int buffer_index_type, const IndexMap& index_map) {
                          return self->AnnotateBufferAccess(
                              block_rv, buffer_index,
                              static_cast<BufferIndexType>(buffer_index_type), index_map);
                        });
}

}  // namespace tir
}  // namespace tvm
