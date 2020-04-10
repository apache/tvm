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
 * \file tvm/target/target.h
 * \brief Compilation target object.
 */
#ifndef TVM_TARGET_TARGET_H_
#define TVM_TARGET_TARGET_H_

#include <tvm/support/with.h>
#include <tvm/node/container.h>
#include <tvm/ir/expr.h>

#include <string>
#include <vector>
#include <unordered_set>
#include <utility>

namespace tvm {
/*!
 * \brief Compilation target.
 * \note Use target::llvm, target::cuda etc functions.
 * \sa Target
 */
class TargetNode : public Object {
 public:
  /*! \brief The name of the target device */
  std::string target_name;
  /*! \brief The name of the target device */
  std::string device_name;
  /*! \brief The type of the target device */
  int device_type;
  /*! \brief The maximum threads that a schedule should use for this device */
  int max_num_threads = 1;
  /*! \brief The warp size that should be used by the LowerThreadAllreduce pass */
  int thread_warp_size = 1;
  /*! \brief Keys for this target */
  Array<runtime::String> keys_array;
  /*! \brief Options for this target */
  Array<runtime::String> options_array;
  /*! \brief Collection of imported libs */
  Array<runtime::String> libs_array;

  /*! \return the full device string to pass to codegen::Build */
  TVM_DLL const std::string& str() const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("target_name", &target_name);
    v->Visit("device_name", &device_name);
    v->Visit("device_type", &device_type);
    v->Visit("max_num_threads", &max_num_threads);
    v->Visit("thread_warp_size", &thread_warp_size);
    v->Visit("keys_array", &keys_array);
    v->Visit("options_array", &options_array);
    v->Visit("libs_array", &libs_array);
  }

  /*! \brief Get the keys for this target as a vector of string */
  TVM_DLL std::vector<std::string> keys() const;

  /*! \brief Get the options for this target as a vector of string */
  TVM_DLL std::vector<std::string> options() const;

  /*! \brief Get the keys for this target as an unordered_set of string */
  TVM_DLL std::unordered_set<std::string> libs() const;

  static constexpr const char* _type_key = "Target";
  TVM_DECLARE_FINAL_OBJECT_INFO(TargetNode, Object);

 private:
  /*! \brief Internal string repr. */
  mutable std::string str_repr_;
};

/*!
 * \brief Managed reference class to TargetNode.
 * \sa TargetNode
 */
class Target : public ObjectRef {
 public:
  Target() {}
  explicit Target(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
  * \brief Create a Target given a string
  * \param target_str the string to parse
  */
  TVM_DLL static Target Create(const std::string& target_str);
  /*!
   * \brief Get the current target context from thread local storage.
   * \param allow_not_defined If the context stack is empty and this is set to true, an
   *   undefined Target will be returned. Otherwise, an empty context stack will cause a
   *   runtime error.
   * \return The target that is the current context. The target may not be defined if
   * allow_not_defined is true.
   */
  TVM_DLL static tvm::Target Current(bool allow_not_defined = true);

  const TargetNode* operator->() const {
      return static_cast<const TargetNode*>(get());
  }

  using ContainerType = TargetNode;
  class Internal;
 private:
  // enable with syntax.
  friend class Internal;
  friend class With<Target>;
  /*!
   * \brief Push a new target context onto the thread local stack.
   *  The Target on top of the stack is used to determine which
   *  specialization to use when invoking a GenericFunc.
   */
  TVM_DLL void EnterWithScope();
  /*!
   * \brief Pop a target off the thread local context stack,
   *  restoring the previous target as the current context.
   */
  TVM_DLL void ExitWithScope();
};

/*! \brief This namespace provides functions to construct Target instances */
namespace target {

/*! \return A target for LLVM */
TVM_DLL Target llvm(const std::vector<std::string>& options =
                    std::vector<std::string>());

/*! \return A target for CUDA */
TVM_DLL Target cuda(const std::vector<std::string>& options =
                    std::vector<std::string>());

/*! \return A target for ROCm */
TVM_DLL Target rocm(const std::vector<std::string>& options =
                    std::vector<std::string>());

/*! \return A target for OpenCL */
TVM_DLL Target opencl(const std::vector<std::string>& options =
                      std::vector<std::string>());

/*! \return A target for Metal */
TVM_DLL Target metal(const std::vector<std::string>& options =
                     std::vector<std::string>());

/*! \return A target for rasp */
TVM_DLL Target rasp(const std::vector<std::string>& options =
                    std::vector<std::string>());

/*! \return A target for Mali */
TVM_DLL Target mali(const std::vector<std::string>& options =
                    std::vector<std::string>());

/*! \return A target for Intel Graphics */
TVM_DLL Target intel_graphics(const std::vector<std::string>& options =
                              std::vector<std::string>());

/*! \return A target for stackvm */
TVM_DLL Target stackvm(const std::vector<std::string>& options =
                       std::vector<std::string>());

/*! \return A target for external device */
TVM_DLL Target ext_dev(const std::vector<std::string>& options =
                       std::vector<std::string>());

/*! \return A target for hexagon */
TVM_DLL Target hexagon(const std::vector<std::string>& options =
                       std::vector<std::string>());
}  // namespace target

/*!
 * \brief Container for build configuration options
 */
class BuildConfigNode : public Object {
 public:
  /*!
   * \brief The data alignment to use when constructing buffers. If this is set to
   * -1, then TVM's internal default will be used
   */
  int data_alignment = -1;
  /*!
   * \brief The offset factor to use when constructing buffers. If this is set to
   * 0, then the offset field is not used.
   */
  int offset_factor = 0;

  /*!
   * \brief Splitting factor for loop splitting. If this is set to zero, no splitting will be
   * done. Otherwise, a split will be done with this factor and the inner loop will be unrolled.
   */
  int double_buffer_split_loop = 1;
  /*! \brief Threshold of number of steps in the loop to be automatically unrolled */
  int auto_unroll_max_step = 0;
  /*! \brief The maximum nested level of loops that can be automatically unrolled */
  int auto_unroll_max_depth = 8;
  /*! \brief The maximum extent of loop that will be unrolled */
  int auto_unroll_max_extent = 0;
  /*!
   * \brief Whether to explicitly unroll the loop. If set to false, the unroll hint will
   * be passed to the CodeGen phase. Set to true if CodeGen supports unroll pragma.
   */
  bool unroll_explicit = true;

  /*! \brief Set to true if buffer arguments do not overlap. This enables more optimization. */
  bool restricted_func = true;

  /*! \brief Whether to detect global barrier */
  bool detect_global_barrier = false;

  /*! \brief Whether to partition const loop */
  bool partition_const_loop = false;

  /*! \brief Whether to dump the IR of each pass (only when building from python) */
  std::vector< std::pair<int, runtime::PackedFunc> > add_lower_pass;

  /*! \brief Whether to dump the IR of each pass (only when building from python) */
  bool dump_pass_ir = false;

  /*! \brief Whether to instrument loads and stores with check for out of the bounds. */
  bool instrument_bound_checkers = false;

  /*! \brief Whether to disable select rewriting. */
  bool disable_select_rewriting = false;

  /*! \brief Whether to disable loop vectorization. */
  bool disable_vectorize = false;

  /*! \brief Whether to disable assert stmt generation. */
  bool disable_assert = false;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data_alignment", &data_alignment);
    v->Visit("offset_factor", &offset_factor);
    v->Visit("double_buffer_split_loop", &double_buffer_split_loop);
    v->Visit("auto_unroll_max_step", &auto_unroll_max_step);
    v->Visit("auto_unroll_max_depth", &auto_unroll_max_depth);
    v->Visit("auto_unroll_max_extent", &auto_unroll_max_extent);
    v->Visit("unroll_explicit", &unroll_explicit);
    v->Visit("restricted_func", &restricted_func);
    v->Visit("detect_global_barrier", &detect_global_barrier);
    v->Visit("partition_const_loop", &partition_const_loop);
    v->Visit("dump_pass_ir", &dump_pass_ir);
    v->Visit("instrument_bound_checkers", &instrument_bound_checkers);
    v->Visit("disable_select_rewriting", &disable_select_rewriting);
    v->Visit("disable_vectorize", &disable_vectorize);
    v->Visit("disable_assert", &disable_assert);
  }

  static constexpr const char* _type_key = "BuildConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(BuildConfigNode, Object);
};

/*!
 * \brief Build configuration for compilations.
 */
class BuildConfig : public ::tvm::ObjectRef {
 public:
  BuildConfig() {}
  explicit BuildConfig(ObjectPtr<Object> n) : ObjectRef(n) {}
  const BuildConfigNode* operator->() const {
    return static_cast<const BuildConfigNode*>(get());
  }
  BuildConfigNode* operator->() {
    return static_cast<BuildConfigNode*>(get_mutable());
  }
  /*!
   * \brief Construct a BuildConfig containing a empty build config node.
   * \return The new BuildConfig
   */
  TVM_DLL static BuildConfig Create();
  /*!
   * \brief Get the current BuildConfig context from thread local storage, or a default
   * configuration if a BuildConfig scope has not been entered.
   * \return The configuration that is the current context.
   */
  TVM_DLL static BuildConfig Current();

  using ContainerType = BuildConfigNode;
  class Internal;

 private:
  // Enable with syntax.
  friend class With<BuildConfig>;
  /*!
   * \brief Push a new BuildConfig context onto the thread local stack.
   */
  TVM_DLL void EnterWithScope();

  /*!
   * \brief Pop a build config off the thread local context stack,
   * restoring the previous configuration as the current context.
   */
  TVM_DLL void ExitWithScope();
};

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_H_
