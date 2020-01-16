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
 * \file tvm/build_module.h
 * \brief Functions for compiling ops.
 */
#ifndef TVM_BUILD_MODULE_H_
#define TVM_BUILD_MODULE_H_

#include <tvm/target/target.h>
#include <tvm/support/with.h>
#include <tvm/top/schedule_pass.h>

#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>

#include "runtime/packed_func.h"

#include "lowered_func.h"

namespace tvm {

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

/*!
* \brief Build a LoweredFunc given a schedule, args and binds
* \param sch The schedule to lower.
* \param args The arguments to the function.
* \param name The name of the lowered function.
* \param binds Buffer assignments.
* \param config The build configuration.
* \return The lowered function.
*/
TVM_DLL Array<LoweredFunc> lower(top::Schedule sch,
                                 const Array<top::Tensor>& args,
                                 const std::string& name,
                                 const std::unordered_map<top::Tensor, Buffer>& binds,
                                 const BuildConfig& config);
/*!
* \brief Split host/device function and running necessary pass before build
* \param funcs The functions to be built.
* \param target The target device to build for.
* \param target_host The target for building host code. To use the default, pass Target()
* \param config The build configuration.
* \return The Array<Array<LoweredFunc>> with 2 elements. First is host function Array,
          second is device function array
*/
TVM_DLL Array<Array<LoweredFunc> > split_dev_host_funcs(const Array<LoweredFunc>& funcs,
                                                        const Target& target,
                                                        const Target& target_host,
                                                        const BuildConfig& config);

/*!
* \brief Build a device and host module for a specific target from an array of lowered functions.
* \param funcs The functions to be built.
* \param target The target device to build for.
* \param target_host The target for building host code. To use the default, pass Target()
* \param config The build configuration.
* \return The built module.
*/
TVM_DLL runtime::Module build(const Array<LoweredFunc>& funcs,
                              const Target& target,
                              const Target& target_host,
                              const BuildConfig& config);

/*!
 * \brief Build a device and host module for a specific target from a map
 * contains target to a list of lowered functions pairs. This function is used
 * for heterogeneous build.
 * \param input The map contains target to a list of lowered functions pairs.
 * \param target_host The target for building host code. To use the default,
 *        pass Target().
 * \param config The build configuration.
 * \return The built module that contains code for different processors.
 */
TVM_DLL runtime::Module build(const Map<Target, Array<LoweredFunc>>& input,
                              const Target& target_host,
                              const BuildConfig& config);

/*!
 * \brief Build a device and host module for a specific target from a map
 * contains target to a list of lowered functions pairs. This function is used
 * for heterogeneous build.
 * \param input The map contains target string to a list of lowered functions
 *        pairs.
 * \param target_host The target for building host code. To use the default,
 *        pass Target().
 * \param config The build configuration.
 * \return The built module that contains code for different processors.
 */
TVM_DLL runtime::Module build(const Map<std::string, Array<LoweredFunc>>& input,
                              const Target& target_host,
                              const BuildConfig& config);

class GenericFuncNode;

/*!
 * \brief Generic function that can be specialized on a per-target basis.
 */
class GenericFunc : public ObjectRef {
 public:
  GenericFunc() {}
  explicit GenericFunc(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*!
   * \brief Set the default function implementaiton.
   * \param value The default function
   * \param allow_override If true, this call may override a previously registered function. If
   * false, an error will be logged if the call would override a previously registered function.
   * \return reference to self.
   */
  TVM_DLL GenericFunc& set_default(const runtime::PackedFunc value,
                                   bool allow_override = false);
  /*!
   * \brief Register a specialized function
   * \param tags The tags for this specialization
   * \param value The specialized function
   * \param allow_override If true, this call may override previously registered tags. If false,
   * an error will be logged if the call would override previously registered tags.
   * \return reference to self.
   */
  TVM_DLL GenericFunc& register_func(const std::vector<std::string>& tags,
                                     const runtime::PackedFunc value,
                                     bool allow_override = false);
  /*!
   * \brief Call generic function by directly passing in unpacked format.
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call generic function
   *   void CallGeneirc(GenericFunc f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template<typename... Args>
  inline runtime::TVMRetValue operator()(Args&& ...args) const;
  /*!
   * \brief Invoke the relevant function for the current target context, set by set_target_context.
   * Arguments are passed in packed format.
   * \param args The arguments to pass to the function.
   * \param ret The return value
   */
  TVM_DLL void CallPacked(runtime::TVMArgs args,
                          runtime::TVMRetValue* ret) const;

  /*!
   * \brief Find or register the GenericFunc instance corresponding to the give name
   * \param name The name of the registered GenericFunc
   * \return The GenericFunc instance
   */
  TVM_DLL static GenericFunc Get(const std::string& name);

  /*!
   * \brief Add a GenericFunc instance to the registry
   * \param func The GenericFunc instance
   * \param name The name of the registered GenericFunc
   */
  TVM_DLL static void RegisterGenericFunc(GenericFunc func, const std::string& name);

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline GenericFuncNode* operator->();

  // declare container type
  using ContainerType = GenericFuncNode;

  // Internal class.
  struct Manager;

 private:
  friend struct Manager;
};

template<typename... Args>
inline runtime::TVMRetValue GenericFunc::operator()(Args&& ...args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  runtime::detail::for_each(runtime::TVMArgsSetter(values, type_codes),
    std::forward<Args>(args)...);
  runtime::TVMRetValue rv;
  CallPacked(runtime::TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

/*!
 * \brief Represents a generic function that can be specialized on a per-target basis.
 */
class GenericFuncNode : public Object {
 public:
  /*! \brief name of the function */
  std::string name_;
  /* \brief the generic builder */
  runtime::PackedFunc generic_func_;
  /* \brief map from keys to registered functions */
  std::unordered_map<std::string, runtime::PackedFunc> dispatch_dict_;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "GenericFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(GenericFuncNode, Object);
};

inline GenericFuncNode* GenericFunc::operator->() {
  return static_cast<GenericFuncNode*>(get_mutable());
}

#define TVM_GENERIC_FUNC_REG_VAR_DEF                            \
  static TVM_ATTRIBUTE_UNUSED ::tvm::GenericFunc& __mk_ ## TVM

/*!
 * \def TVM_REGISTER_GENERIC_FUNC
 * \brief Register a new generic function, or set a device-specific variant
 * of the corresponding function.
 *
 * \param name The name of the function
 */
#define TVM_REGISTER_GENERIC_FUNC(name)                           \
  TVM_STR_CONCAT(TVM_GENERIC_FUNC_REG_VAR_DEF, __COUNTER__) =     \
      ::tvm::GenericFunc::Get(#name)


}  // namespace tvm

#endif  // TVM_BUILD_MODULE_H_
