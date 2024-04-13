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
 * \file tvm/runtime/relax_vm/vm.h
 */
#ifndef TVM_RUNTIME_RELAX_VM_VM_H_
#define TVM_RUNTIME_RELAX_VM_VM_H_

#ifndef TVM_RELAX_VM_ENABLE_PROFILER
#define TVM_RELAX_VM_ENABLE_PROFILER 1
#endif

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../memory/memory_manager.h"
#include "./bytecode.h"
#include "./executable.h"

namespace tvm {
namespace runtime {

using memory::Allocator;
using memory::AllocatorType;
using memory::MemoryManager;
using memory::Storage;
using memory::StorageObj;

namespace relax_vm {

/*!
 * \brief Possible instrument actions.
 */
enum class VMInstrumentReturnKind : int {
  /*! \brief Running as normal. */
  kNoOp = 0,
  /*! \brief Skip the following run, only valid in before. */
  kSkipRun = 1,
};

/*!
 * \brief An object representing a vm closure.
 */
class VMClosureObj : public ClosureObj {
 public:
  /*!
   * \brief The function name. The function could be any
   * function object that is compatible to the VM runtime.
   */
  String func_name;

  /*!
   * \brief The implementation of the Closure.
   * \note This function takes context pointer(VirtualMachine*)
   *       as the first argument. The rest of arguments follows
   *       the same arguments as the normal function call.
   */
  PackedFunc impl;

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.Closure";
  TVM_DECLARE_FINAL_OBJECT_INFO(VMClosureObj, ClosureObj);
};

/*! \brief reference to closure. */
class VMClosure : public Closure {
 public:
  VMClosure(String func_name, PackedFunc impl);
  TVM_DEFINE_OBJECT_REF_METHODS(VMClosure, Closure, VMClosureObj);

  /*!
   * \brief Create another PackedFunc with last arguments already bound to last_args.
   *
   * This is a helper function to create captured closures.
   * \param func The input func, can be a VMClosure or PackedFunc.
   * \param last_args The arguments to bound to in the end of the function.
   * \note The new function takes in arguments and append the last_args in the end.
   */
  static PackedFunc BindLastArgs(PackedFunc func, std::vector<TVMRetValue> last_args);
};

/*!
 * \brief Represent a VM extension.
 * A VM extension allows the user to extend the VM with target specific functionalities.
 * The VM holds the reference of the extensions to ensure the extensions have the same lifetime
 * as the VM.
 *
 * This is the base class for all VM extensions and should not be used directly.
 */
class VMExtensionNode : public Object {
 protected:
  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "runtime.VMExtension";
  TVM_DECLARE_BASE_OBJECT_INFO(VMExtensionNode, Object);
};

/*! \brief Managed reference to VM extension. */
class VMExtension : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(VMExtension, ObjectRef, VMExtensionNode);
};

/*!
 * \brief The virtual machine.
 *
 * The virtual machine contains all the current execution state,
 * as well as the executable.
 *
 * The goal is to have a single self-contained object,
 * enabling one to easily pass around VMs, execute them on
 * multiple threads, or serialize them to disk or over the
 * wire.
 */
class VirtualMachine : public runtime::ModuleNode {
 public:
  /*!
   * \brief Initialize the virtual machine for a set of devices.
   * \param devices The set of TVM devices.
   * \param alloc_types The allocator types for each device.
   */
  virtual void Init(const std::vector<Device>& devices,
                    const std::vector<AllocatorType>& alloc_types) = 0;
  /*!
   * \brief Load the executable for the virtual machine.
   * \param exec The executable.
   */
  virtual void LoadExecutable(ObjectPtr<Executable> exec) = 0;
  /*!
   * \brief Get global function in the VM.
   * \param func_name The name of the function.
   * \return The closure
   */
  virtual VMClosure GetClosure(const String& func_name) = 0;
  /*!
   * \brief Invoke closure or packed function using PackedFunc convention.
   * \param closure_or_packedfunc A VM closure or a packed_func.
   * \param args The input arguments.
   * \param rv The return value.
   */
  virtual void InvokeClosurePacked(const ObjectRef& closure_or_packedfunc, TVMArgs args,
                                   TVMRetValue* rv) = 0;
  /*!
   * \brief Set an instrumentation function.
   *
   * If instrument is present, the function will be called
   * before/after each Call instruction.
   *
   * bool instrument(func, func_symbol, before_run, args...)
   *
   * - func: Union[VMClosure, PackedFunc], the function object.
   * - func_symbol: string, the symbol of the function.
   * - before_run: bool, whether it is before or after call.
   * - ret_value: Only valid in after run, otherwise it is null.
   * - args: the arguments being passed to call.
   *
   * instrument can return an int which corresponds to the action value.
   * \sa VMInstrumentAction
   *
   * \param instrument The instrument function.
   */
  virtual void SetInstrument(PackedFunc instrument) = 0;

  /*!
   * \brief Get or create a VM extension. Once created, the extension will be stored in the VM
   * and held until the VM is destructed.
   *
   * \tparam T The type of the extension
   * \return The extension instance
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of<VMExtension, T>::value>>
  T GetOrCreateExtension() {
    using ContainerType = typename T::ContainerType;
    uint32_t key = ContainerType::RuntimeTypeIndex();
    if (auto it = extensions.find(key); it != extensions.end()) {
      return Downcast<T>((*it).second);
    }
    auto [it, _] = extensions.emplace(key, T::Create());
    return Downcast<T>((*it).second);
  }

  /*!
   * \brief Create a specific instance of VM.
   * \return Created VM
   */
  static ObjectPtr<VirtualMachine> Create();
  /*!
   * \brief Create an instance of VM with the profiling feature enabled.
   * \return Created VM
   */
  static ObjectPtr<VirtualMachine> CreateProfiler();
  /*!
   * \brief Helper function for vm closure functions to get the context ptr
   * \param arg The argument value.
   */
  static VirtualMachine* GetContextPtr(TVMArgValue arg) {
    return static_cast<VirtualMachine*>(arg.operator void*());
  }

  ~VirtualMachine() {}

  //--------------------------------------------------------------------------
  // The following section contains states that other builtin can depend on
  //--------------------------------------------------------------------------
  /*! \brief The memory allocators. */
  std::vector<Allocator*> allocators;
  /*! \brief Runtime physical device list. */
  std::vector<Device> devices;
  /*! \brief The VM extensions. Mapping from the type index of the extension to the extension
   * instance. */
  std::unordered_map<uint32_t, VMExtension> extensions;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_VM_H_
