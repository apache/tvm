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
namespace s_tir {
using namespace tvm::tir;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ScheduleNode>();
  SBlockRVNode::RegisterReflection();
  LoopRVNode::RegisterReflection();
}

/**************** Constructor ****************/

SBlockRV::SBlockRV() { this->data_ = ffi::make_object<SBlockRVNode>(); }

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
      .def_method("s_tir.schedule.ScheduleGetMod", &ScheduleNode::mod)
      .def_method("s_tir.schedule.ScheduleGetState", &ScheduleNode::state)
      .def_method("s_tir.schedule.ScheduleGetTrace", &ScheduleNode::trace)
      .def_method("s_tir.schedule.ScheduleGetFuncWorkingOn", &ScheduleNode::func_working_on)
      .def_method("s_tir.schedule.ScheduleCopy", &ScheduleNode::Copy)
      .def_method("s_tir.schedule.ScheduleSeed", &ScheduleNode::Seed)
      .def_method("s_tir.schedule.ScheduleForkSeed", &ScheduleNode::ForkSeed)
      .def_method("s_tir.schedule.ScheduleWorkOn", &ScheduleNode::WorkOn);
}

/**************** (FFI) Constructor ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.SBlockRV", []() { return SBlockRV(); })
      .def("s_tir.schedule.LoopRV", []() { return LoopRV(); })
      .def("s_tir.schedule.ConcreteSchedule",
           [](IRModule mod, support::LinearCongruentialEngine::TRandState seed, int debug_mask,
              int error_render_level, bool enable_check) -> Schedule {
             return Schedule::Concrete(mod, debug_mask, seed,
                                       static_cast<ScheduleErrorRenderLevel>(error_render_level),
                                       enable_check);
           })
      .def("s_tir.schedule.TracedSchedule",
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
      .def("s_tir.schedule.ScheduleGet",
           [](Schedule self, ObjectRef obj) -> ObjectRef {
             if (auto loop_rv = obj.as<LoopRV>()) {
               return self->Get(loop_rv.value());
             }
             if (auto block_rv = obj.as<SBlockRV>()) {
               return self->Get(block_rv.value());
             }
             if (auto expr_rv = obj.as<ExprRV>()) {
               return self->Get(expr_rv.value());
             }
             LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: "
                        << obj->GetTypeKey() << ". Its value is: " << obj;
             throw;
           })
      .def("s_tir.schedule.ScheduleGetSRef",
           [](Schedule self, ObjectRef obj) -> ffi::Optional<ObjectRef> {
             if (auto loop_rv = obj.as<LoopRV>()) {
               return self->GetSRef(loop_rv.value());
             }
             if (auto block_rv = obj.as<SBlockRV>()) {
               return self->GetSRef(block_rv.value());
             }
             if (auto stmt = obj.as<Stmt>()) {
               return self->GetSRef(stmt.value());
             }
             LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
             throw;
           })
      .def("s_tir.schedule.ScheduleRemoveRV", [](Schedule self, ObjectRef obj) -> void {
        if (auto loop_rv = obj.as<LoopRV>()) {
          return self->RemoveRV(loop_rv.value());
        }
        if (auto block_rv = obj.as<SBlockRV>()) {
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
      .def_method("s_tir.schedule.ScheduleSampleCategorical", &ScheduleNode::SampleCategorical)
      .def_method("s_tir.schedule.ScheduleSamplePerfectTile", &ScheduleNode::SamplePerfectTile)
      .def_method("s_tir.schedule.ScheduleSamplePartitionedTile",
                  &ScheduleNode::SamplePartitionedTile)
      .def_method("s_tir.schedule.ScheduleSampleComputeLocation",
                  &ScheduleNode::SampleComputeLocation);
}
/******** (FFI) Get blocks & loops ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleGetSBlock", &ScheduleNode::GetSBlock)
      .def_method("s_tir.schedule.ScheduleGetLoops", &ScheduleNode::GetLoops)
      .def("s_tir.schedule.ScheduleGetChildBlocks",
           [](Schedule self, ObjectRef rv) {
             if (auto block_rv = rv.as<SBlockRV>()) {
               return self->GetChildBlocks(block_rv.value());
             }
             if (auto loop_rv = rv.as<LoopRV>()) {
               return self->GetChildBlocks(loop_rv.value());
             }
             LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: "
                        << rv->GetTypeKey() << ". Its value is: " << rv;
             throw;
           })
      .def_method("s_tir.schedule.ScheduleGetProducers", &ScheduleNode::GetProducers)
      .def_method("s_tir.schedule.ScheduleGetConsumers", &ScheduleNode::GetConsumers)
      .def_method("s_tir.schedule.ScheduleGetOutputBlocks", &ScheduleNode::GetOutputBlocks);
}
/******** (FFI) Transform loops ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleMerge", &ScheduleNode::Merge)
      .def_method("s_tir.schedule.ScheduleFuse", &ScheduleNode::Fuse)
      .def_method("s_tir.schedule.ScheduleSplit", &ScheduleNode::Split)
      .def_method("s_tir.schedule.ScheduleLoopPartition", &ScheduleNode::LoopPartition)
      .def_method("s_tir.schedule.ScheduleReorder", &ScheduleNode::Reorder)
      .def_method("s_tir.schedule.ScheduleReorderBlockIterVar", &ScheduleNode::ReorderBlockIterVar)
      .def("s_tir.schedule.ScheduleAddUnitLoop", [](Schedule self, ObjectRef rv) -> LoopRV {
        if (auto loop_rv = rv.as<LoopRV>()) {
          return self->AddUnitLoop(loop_rv.value());
        } else if (auto block_rv = rv.as<SBlockRV>()) {
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
      .def_method("s_tir.schedule.ScheduleParallel", &ScheduleNode::Parallel)
      .def_method("s_tir.schedule.ScheduleVectorize", &ScheduleNode::Vectorize)
      .def_method("s_tir.schedule.ScheduleBind", &ScheduleNode::Bind)
      .def_method("s_tir.schedule.ScheduleUnroll", &ScheduleNode::Unroll);
}
/******** (FFI) Insert cache stages ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleCacheRead", &ScheduleNode::CacheRead)
      .def_method("s_tir.schedule.ScheduleCacheWrite", &ScheduleNode::CacheWrite)
      .def_method("s_tir.schedule.ScheduleReindexCacheRead", &ScheduleNode::ReindexCacheRead)
      .def_method("s_tir.schedule.ScheduleReindexCacheWrite", &ScheduleNode::ReindexCacheWrite)
      .def_method("s_tir.schedule.ScheduleCacheInplace", &ScheduleNode::CacheInplace)
      .def_method("s_tir.schedule.ScheduleCacheIndex", &ScheduleNode::CacheIndex)
      .def("s_tir.schedule.ScheduleReIndex",
           [](Schedule self, const SBlockRV& block_rv, int buffer_index, int buffer_index_type) {
             return self->ReIndex(block_rv, buffer_index,
                                  static_cast<BufferIndexType>(buffer_index_type));
           });
}
/******** (FFI) Data movement ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleReadAt", &ScheduleNode::ReadAt)
      .def_method("s_tir.schedule.ScheduleWriteAt", &ScheduleNode::WriteAt);
}
/******** (FFI) Compute location ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleComputeAt", &ScheduleNode::ComputeAt)
      .def_method("s_tir.schedule.ScheduleReverseComputeAt", &ScheduleNode::ReverseComputeAt)
      .def_method("s_tir.schedule.ScheduleComputeInline", &ScheduleNode::ComputeInline)
      .def_method("s_tir.schedule.ScheduleReverseComputeInline",
                  &ScheduleNode::ReverseComputeInline)
      .def_method("s_tir.schedule.ScheduleFuseReductionEpilogue",
                  &ScheduleNode::FuseReductionEpilogue);
}
/******** (FFI) Reduction ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleDecomposeReduction", &ScheduleNode::DecomposeReduction)
      .def_method("s_tir.schedule.ScheduleRFactor", &ScheduleNode::RFactor);
}
/******** (FFI) SBlock annotation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleStorageAlign", &ScheduleNode::StorageAlign)
      .def_method("s_tir.schedule.ScheduleSetScope", &ScheduleNode::SetScope)
      .def_method("s_tir.schedule.ScheduleUnsafeSetDType", &ScheduleNode::UnsafeSetDType);
}
/******** (FFI) Blockize & Tensorize ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.schedule.ScheduleBlockize",
           [](Schedule self, ObjectRef target, bool preserve_unit_iters) {
             if (auto loop_rv = target.as<LoopRV>()) {
               return self->Blockize(loop_rv.value(), preserve_unit_iters);
             } else if (auto blocks = target.as<ffi::Array<SBlockRV>>()) {
               return self->Blockize(blocks.value(), preserve_unit_iters);
             }
             LOG(FATAL) << "Unsupported target type: " << target->GetTypeKey();
           })
      .def("s_tir.schedule.ScheduleTensorize",
           [](Schedule self, ObjectRef rv, ffi::String intrin, bool preserve_unit_iters) {
             if (auto block_rv = rv.as<SBlockRV>()) {
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
      .def("s_tir.schedule.ScheduleAnnotate",
           [](Schedule self, ObjectRef rv, const ffi::String& ann_key, const Any& ann_val) {
             if (auto block_rv = rv.as<SBlockRV>()) {
               return self->Annotate(block_rv.value(), ann_key, ann_val);
             }
             if (auto loop_rv = rv.as<LoopRV>()) {
               return self->Annotate(loop_rv.value(), ann_key, ann_val);
             }
             LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: "
                        << rv->GetTypeKey() << ". Its value is: " << rv;
             throw;
           })
      .def("s_tir.schedule.ScheduleUnannotate", [](Schedule self, ObjectRef rv,
                                                   const ffi::String& ann_key) {
        if (auto block_rv = rv.as<SBlockRV>()) {
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
      .def("s_tir.schedule.ScheduleTransformLayout",
           [](Schedule self, const SBlockRV& block_rv, int buffer_index, int buffer_index_type,
              const IndexMap& index_map, const ffi::Optional<IndexMap>& pad_value,
              bool assume_injective_transform) {
             return self->TransformLayout(block_rv, buffer_index,
                                          static_cast<BufferIndexType>(buffer_index_type),
                                          index_map, pad_value, assume_injective_transform);
           })
      .def_method("s_tir.schedule.ScheduleTransformBlockLayout",
                  &ScheduleNode::TransformBlockLayout)
      .def("s_tir.schedule.ScheduleSetAxisSeparator",
           [](Schedule self, const SBlockRV& block_rv, int buffer_index, int buffer_index_type,
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
      .def_method("s_tir.schedule.ScheduleDecomposePadding", &ScheduleNode::DecomposePadding)
      .def_method("s_tir.schedule.SchedulePadEinsum", &ScheduleNode::PadEinsum);
}
/******** (FFI) Buffer transformation ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_method("s_tir.schedule.ScheduleRollingBuffer",
                               &ScheduleNode::RollingBuffer);
}
/******** (FFI) Misc ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.schedule.ScheduleEnterPostproc", &ScheduleNode::EnterPostproc)
      .def_method("s_tir.schedule.ScheduleUnsafeHideBufferAccess",
                  &ScheduleNode::UnsafeHideBufferAccess);
}
/******** (FFI) Annotate buffer access ********/
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.schedule.ScheduleAnnotateBufferAccess",
                        [](Schedule self, const SBlockRV& block_rv, int buffer_index,
                           int buffer_index_type, const IndexMap& index_map) {
                          return self->AnnotateBufferAccess(
                              block_rv, buffer_index,
                              static_cast<BufferIndexType>(buffer_index_type), index_map);
                        });
}

}  // namespace s_tir
}  // namespace tvm
