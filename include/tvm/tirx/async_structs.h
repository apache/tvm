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
 * \file tvm/tirx/async_structs.h
 * \brief Language structures for asynchronous execution in TIR+.
 */
#ifndef TVM_TIRX_ASYNC_STRUCTS_H_
#define TVM_TIRX_ASYNC_STRUCTS_H_

#include <tvm/ffi/object.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/exec_scope.h>

namespace tvm {
namespace tirx {

// Pipeline
class PipelineNode : public ffi::Object {
 public:
  /*! \brief The thread scope of this pipeline */
  ExecScope thread_scope;
  /*! \brief The pipeline depth */
  size_t depth;
  /*! \brief Whether to separate producer and consumer threads */
  bool separate_pc;
  /*! \brief The name hint of the pipeline. */
  ffi::String name_hint;

  /*! \brief The workspace of the pipeline. */
  ffi::Map<ffi::String, tvm::tirx::Buffer> workspace;
  /*! \brief The schedule config of the pipeline. */
  ffi::Map<ffi::String, ffi::Any> schedule_config;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PipelineNode>()
        .def_ro("thread_scope", &PipelineNode::thread_scope)
        .def_ro("name_hint", &PipelineNode::name_hint)
        .def_ro("depth", &PipelineNode::depth)
        .def_ro("separate_pc", &PipelineNode::separate_pc)
        .def_ro("workspace", &PipelineNode::workspace)
        .def_ro("schedule_config", &PipelineNode::schedule_config);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.Pipeline", PipelineNode, ffi::Object);
};

class Pipeline : public ffi::ObjectRef {
 public:
  TVM_DLL explicit Pipeline(ExecScope thread_scope, size_t depth = 0, bool separate_pc = false,
                            ffi::String name_hint = "",
                            ffi::Map<ffi::String, tvm::tirx::Buffer> workspace = {},
                            ffi::Map<ffi::String, ffi::Any> schedule_config = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Pipeline, ffi::ObjectRef, PipelineNode);
};

// CopyPipeline
class CopyPipelineNode : public PipelineNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<CopyPipelineNode>();
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.CopyPipeline", CopyPipelineNode, PipelineNode);
};

class CopyPipeline : public Pipeline {
 public:
  TVM_DLL explicit CopyPipeline(ExecScope thread_scope, size_t depth = 0, bool separate_pc = false,
                                ffi::String name_hint = "",
                                ffi::Map<ffi::String, tvm::tirx::Buffer> workspace = {},
                                ffi::Map<ffi::String, ffi::Any> schedule_config = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CopyPipeline, Pipeline, CopyPipelineNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CopyPipelineNode);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_ASYNC_STRUCTS_H_
