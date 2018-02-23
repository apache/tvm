/*!
*  Copyright (c) 2017 by Contributors
* \file build_module.h
* \brief Functions for compiling ops.
*/
#ifndef TVM_BUILD_MODULE_H_
#define TVM_BUILD_MODULE_H_

#include <string>
#include <vector>
#include "./tvm/runtime/packed_func.h"
#include "./tvm/schedule_pass.h"
#include "./tvm/lowered_func.h"

namespace tvm {
using namespace tvm::runtime;

/*!
* \brief Container for target device information.
* Use target::llvm, target::cuda etc functions instead of constructing directly.
*/
struct Target {
  /*! \brief The name of the target device */
  std::string target_name;
  /*! \brief The type of the target device */
  DLDeviceType device_type;
  /*! \brief The maximum threads that a schedule should use for this device */
  int max_num_threads = 1;
  /*! \brief The warp size that should be used by the LowerThreadAllreduce pass */
  int thread_warp_size = 1;
  /*! \brief Keys for this target */
  std::vector<std::string> keys;
  /*! \brief Options for this target */
  std::vector<std::string> options;
  /*! \brief Set of imported libs */
  std::unordered_set<std::string> libs;

  Target(const std::string& target_name,
         DLDeviceType device_type,
         int max_num_threads,
         int thread_warp_size,
         const std::vector<std::string>& keys,
         const std::vector<std::string>& options,
         const std::unordered_set<std::string>& libs =
           std::unordered_set<std::string>()) :
    target_name(target_name),
    device_type(device_type),
    max_num_threads(max_num_threads),
    thread_warp_size(thread_warp_size),
    keys(keys),
    options(options),
    libs(libs) {
  }

  /*! \return the full device string to pass to codegen::Build */
  EXPORT std::string str() const;

  /*!
   * \brief Create a Target given a string
   * \param target_str the string to parse
   */
  EXPORT static Target create(const std::string& target_str);

  /*!
   * \brief Push a new target context onto the thread local stack. The Target on top of
   * the stack is used to determine which specialization to use when invoking a GenericFunc.
   * \param target The target to set as the current context.
   */
  EXPORT static void EnterTargetScope(const tvm::Target& target);

  /*!
   * \brief Pop a target off the thread local context stack, restoring the previous target
   * as the current context.
   */
  EXPORT static void ExitTargetScope();

  /*!
   * \brief Get the current target context from thread local storage.
   * \param allow_null If the context stack is empty and this is set to true, a nullptr
   * will be returned. Otherwise, an empty context stack will cause a runtime error.
   * \return The target that is the current context. nullptr may be returned if
   * allow_null is true.
   */
  EXPORT static tvm::Target* Target::current_target(bool allow_null = true);
};

/*!
 * \brief RAII container to provide a scoped target context. Pushes a target onto the
 * context stack when constructed, and pops it when destructed.
 */
struct TargetContext {
  /*!
   * \brief Enter a new target context. The given target becomes the new current context.
   * When the TargetContext is destructed, the previous context is restored.
   * \param target The target to set as the new current context.
   */
  explicit TargetContext(const tvm::Target& target) {
    Target::EnterTargetScope(target);
  }

  /*! \brief Destructor. Pops the context off the thread local stack. */
  ~TargetContext() {
    Target::ExitTargetScope();
  }
};

/*! \brief This namespace provides functions to construct Target instances */
namespace target {
/*! \return A target for LLVM */
EXPORT Target llvm();

/*! \return A target for CUDA */
EXPORT Target cuda();

/*! \return A target for ROCm */
EXPORT Target rocm();

/*! \return A target for Metal */
EXPORT Target metal();

/*! \return A target for rasp */
EXPORT Target rasp();

/*! \return A target for Mali */
EXPORT Target mali();

/*! \return A target for stackvm */
EXPORT Target stackvm();

}  // namespace target

class BuildConfig;

/*!
* \brief Container for build configuration options
*/
class BuildConfigNode : public Node {
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

  void VisitAttrs(AttrVisitor* v) final {
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
  }

  static constexpr const char* _type_key = "BuildConfig";
  TVM_DECLARE_NODE_TYPE_INFO(BuildConfigNode, Node);
};

TVM_DEFINE_NODE_REF(BuildConfig, BuildConfigNode);

/*!
* \brief Construct a BuildConfig containing a new BuildConfigNode
* \return The new BuildConfig
*/
EXPORT BuildConfig build_config();

/*!
* \brief Build a LoweredFunc given a schedule, args and binds
* \param sch The schedule to lower.
* \param args The arguments to the function.
* \param name The name of the lowered function.
* \param binds Buffer assignments.
* \param config The build configuration.
* \return The lowered function.
*/
EXPORT Array<LoweredFunc> lower(Schedule sch,
                                const Array<Tensor>& args,
                                const std::string& name,
                                const std::unordered_map<Tensor, Buffer>& binds,
                                const BuildConfig& config);

/*!
* \brief Build a device and host module for a specific target from an array of lowered functions.
* \param funcs The functions to be built.
* \param target The target device to build for.
* \param target_host The target for building host code. If null, a suitable default will be used.
* \param config The build configuration.
* \return The built module.
*/
EXPORT runtime::Module build(const Array<LoweredFunc>& funcs,
                             const Target& target,
                             Target* target_host,
                             const BuildConfig& config);

/*!
* \brief Generic function that can be specialized on a per-target basis.
*/
class GenericFunc {
 public:
  /*!
  * \brief Set the default function implementaiton.
  * \param value The default function
  * \param allow_override If true, this call may override a previously registered function. If
  * false, an error will be logged if the call would override a previously registered function.
  * \return reference to self.
  */
  TVM_DLL GenericFunc& set_default_func(const PackedFunc value,
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
                                     const PackedFunc value,
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
  inline TVMRetValue operator()(Args&& ...args) const;
  /*!
  * \brief Invoke the relevant function for the current target context, set by set_target_context.
  * Arguments are passed in packed format.
  * \param args The arguments to pass to the function.
  * \param ret The return value
  */
  TVM_DLL void CallPacked(TVMArgs args, TVMRetValue* ret) const;

  /*!
  * \brief Find or register the GenericFunc instance corresponding to the give name
  * \param name The name of the registered GenericFunc
  * \return The GenericFunc instance
  */
  TVM_DLL static GenericFunc& Get(const std::string& name);

  // Internal class.
  struct Manager;

 private:
  /*! \brief name of the function */
  std::string name_;
  /* \brief the generic builder */
  PackedFunc generic_func_;
  /* \brief map from keys to registered functions */
  std::unordered_map<std::string, PackedFunc> dispatch_dict_;
  friend struct Manager;
};

template<typename... Args>
inline TVMRetValue GenericFunc::operator()(Args&& ...args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(TVMArgsSetter(values, type_codes),
    std::forward<Args>(args)...);
  TVMRetValue rv;
  CallPacked(TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

#define TVM_GENERIC_FUNC_REG_VAR_DEF                               \
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
