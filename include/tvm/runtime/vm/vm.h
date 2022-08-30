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
 * \file tvm/runtime/vm/vm.h
 * \brief The Relay virtual machine runtime.
 */
#ifndef TVM_RUNTIME_VM_VM_H_
#define TVM_RUNTIME_VM_VM_H_

#include <tvm/runtime/container/closure.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/bytecode.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/memory_manager.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

/*!
 * \brief An object representing a vm closure.
 */
class VMClosureObj : public ClosureObj {
 public:
  /*!
   * \brief The index into the function list. The function could be any
   * function object that is compatible to the VM runtime.
   */
  size_t func_index;
  /*! \brief The free variables of the closure. */
  std::vector<ObjectRef> free_vars;

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "vm.Closure";
  TVM_DECLARE_FINAL_OBJECT_INFO(VMClosureObj, ClosureObj);
};

/*! \brief reference to closure. */
class VMClosure : public Closure {
 public:
  VMClosure(size_t func_index, std::vector<ObjectRef> free_vars);
  TVM_DEFINE_OBJECT_REF_METHODS(VMClosure, Closure, VMClosureObj);
};

/*!
 * \brief A representation of a Relay function in the VM.
 *
 * Contains metadata about the compiled function, as
 * well as the compiled VM instructions.
 */
struct VMFunction {
  /*! \brief The function's name. */
  std::string name;
  /*! \brief The function parameter names. */
  std::vector<std::string> params;
  /*! \brief The instructions representing the function. */
  std::vector<Instruction> instructions;
  /*! \brief The size of the frame for this function */
  Index register_file_size = 0;
  /*! \brief The indexes for the device holding each function parameter. */
  std::vector<Index> param_device_indexes;

  VMFunction(std::string name, std::vector<std::string> params,
             std::vector<Instruction> instructions, Index register_file_size,
             std::vector<Index> param_device_indexes)
      : name(std::move(name)),
        params(std::move(params)),
        instructions(std::move(instructions)),
        register_file_size(register_file_size),
        param_device_indexes(std::move(param_device_indexes)) {
    ICHECK_EQ(this->params.size(), this->param_device_indexes.size());
  }

  VMFunction() = default;

  friend std::ostream& operator<<(std::ostream& os, const VMFunction&);
};

/*!
 * \brief A representation of a stack frame.
 *
 * A stack frame is a record containing the information needed
 * to restore the caller's virtual machine state after returning
 * from a function call.
 */
struct VMFrame {
  /*! \brief The return program counter. */
  Index pc;
  /*! \brief The index into the function table, points to the caller. */
  Index func_index;
  /*! \brief The number of arguments. */
  Index args;
  /*! \brief A pointer into the caller function's instructions. */
  const Instruction* code;

  /*! \brief Statically allocated space for objects */
  std::vector<ObjectRef> register_file;

  /*! \brief Register in caller's frame to put return value */
  RegName caller_return_register;

  VMFrame(Index pc, Index func_index, Index args, const Instruction* code, Index register_file_size)
      : pc(pc),
        func_index(func_index),
        args(args),
        code(code),
        register_file(register_file_size),
        caller_return_register(0) {}
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
class TVM_DLL VirtualMachine : public runtime::ModuleNode {
 public:
  /*!
   * \brief Get a PackedFunc from module.
   *
   *  The PackedFunc may not be fully initialized,
   *  there might still be first time running overhead when
   *  executing the function on certain devices.
   *  For benchmarking, use prepare to eliminate
   *
   * \param name the name of the function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   *
   * \note The function will always remain valid.
   *   If the function needs resource from the module(e.g. late linking),
   *   it should capture sptr_to_self.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  virtual ~VirtualMachine() {}

  const char* type_key() const final { return "VirtualMachine"; }

  VirtualMachine() : frames_(), func_index_(0), code_(nullptr), pc_(0), exec_(nullptr) {}

  /*!
   * \brief load the executable for the virtual machine.
   * \param exec The executable.
   */
  virtual void LoadExecutable(const ObjectPtr<Executable>& exec);

 protected:
  /*! \brief Push a call frame on to the call stack. */
  void PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func);

  /*!
   * \brief Pop a frame off the call stack.
   * \return The number of frames left.
   */
  Index PopFrame();

  /*!
   * \brief Write to a VM register.
   * \param reg The register to write to.
   * \param obj The object to write to.
   */
  inline void WriteRegister(RegName reg, const ObjectRef& obj);

  /*!
   * \brief Read a VM register.
   * \param reg The register to read from.
   * \return The read object.
   */
  ObjectRef ReadRegister(RegName reg) const;

  /*!
   * \brief Read a VM register and cast it to int32_t
   * \param reg The register to read from.
   * \return The read scalar.
   */
  int64_t LoadScalarInt(RegName reg) const;

  /*!
   * \brief Invoke a VM function.
   * \param func The function.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  ObjectRef Invoke(const VMFunction& func, const std::vector<ObjectRef>& args);

  // TODO(@jroesch): I really would like this to be a global variable.
  /*!
   * \brief Invoke a VM function by name.
   * \param name The function's name.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  ObjectRef Invoke(const std::string& name, const std::vector<ObjectRef>& args);

  /*!
   * \brief Invoke a PackedFunction
   *
   * \param packed_index The offset of the PackedFunction in all functions.
   * \param func The PackedFunction to be invoked.
   * \param arg_count The number of arguments to the PackedFunction.
   * \param output_size The number of outputs of the PackedFunction.
   * \param args Arguments to the PackedFunction.
   *
   * \note The return value will be stored in the last output_size slots of args.
   */
  virtual void InvokePacked(Index packed_index, const PackedFunc& func, Index arg_count,
                            Index output_size, const std::vector<ObjectRef>& args);

  /*!
   * \brief Initialize the virtual machine for a set of (physical) devices.
   * \param physical_devices The set of TVM devices.
   * \param alloc_types The allocator types for each device.
   */
  void Init(const std::vector<Device>& physical_devices,
            const std::vector<AllocatorType>& alloc_types);

  /*! \brief Run VM dispatch loop. */
  void RunLoop();

  /*! \brief Get device from the device list based on a given device index. */
  Device GetDevice(Index device_index) const;
  Allocator* GetAllocator(Index device_index) const;

  /*!
   * \brief Invoke a global setting up the VM state to execute.
   *
   * This does not begin execution of the VM.
   */
  void InvokeGlobal(const VMFunction& func, const std::vector<ObjectRef>& args);

  /*!
   * \brief Set inputs to a function.
   * \param name The function name
   * \param args args[offset:] are arguments to the
   * function. If the arguments are not of the correct device for the function,
   * they will be copied to the device.
   * \param offset Starting offset of the arguments in `args`.
   */
  void SetInput(std::string name, TVMArgs args, int offset);

  /*!
   * \brief Set one input tensor with index or name to a function.
   * \param name The function name.
   * \param tag index or name of the input tensor .
   * \param tensor the input tensor. If the tensor is not of the correct device for the function,
   * they will be copied to the device.
   */
  void SetOneInput(std::string name, const TVMArgValue& tag, const TVMArgValue& tensor);

  /*!
   * \brief Internal hook for profiling the start of an op.
   *
   * This hook is only called on certain ops that are likely to take a
   * significant amount of runtime (normally because they alloc or transfer to
   * device).
   *
   * \param instr Instruction that will be executed after this hook fires
   */
  virtual void OpStartHook(Instruction instr);

  /*!
   * \brief Internal hook for profiling the end of an op.
   */
  virtual void OpStopHook();

 private:
  /*!
   * \brief Get index of input tensor from its name.
   * \param func_name The function's name.
   * \param input_name The input tensor name.
   * \return The input tensor index.
   */
  int64_t GetInputIndexFromVMFunction(const std::string& func_name,
                                      const std::string& input_name) const;

  /*!
   * \brief Get index of input tensor from its name.
   * \param params parameter names.
   * \param input_name The input tensor name.
   * \return The input tensor index.
   */
  int64_t GetInputIndexFromName(const std::vector<std::string>& params,
                                const std::string& input_name) const;

  /*!
   * \brief Check executable exists and get VM function from it.
   * \param func_name The function's name.
   * \return VM function.
   */
  const VMFunction& CheckAndGetVMFunction(const std::string& func_name) const;

  /*!
   * \brief Creats inputs_ field, if it exists check its size.
   * \param func_name The function's name.
   * \param size inputs_ field size.
   * \return VM function.
   */
  void CreateInputsOrCheckSize(const std::string& func_name, size_t size);

  /*!
   * \brief Set one input tensor with given index to set of input tensors if need copy to given
   * device. \param tensors the input tensors set (destination) \param tensor some tensor (not
   * necessary DLTensor). \param index The input tensor index. \param dev device to copy if need.
   */
  void SetInputTensorWithIndex(std::vector<ObjectRef>& tensors,  // NOLINT(*)
                               const TVMArgValue& tensor, int index, Device dev);

 protected:
  /*! \brief The virtual machine's packed function table. */
  std::vector<PackedFunc> packed_funcs_;
  /*! \brief The current stack of call frames. */
  std::vector<VMFrame> frames_;
  /*! \brief The fuction table index of the current function. */
  Index func_index_;
  /*! \brief The current pointer to the code section. */
  const Instruction* code_;
  /*! \brief The virtual machine PC. */
  Index pc_;
  /*! \brief The special return register. */
  ObjectRef return_register_;
  /*! \brief The executable the VM will operate on. */
  ObjectPtr<Executable> exec_;
  /*! \brief The function name to inputs mapping. */
  std::unordered_map<std::string, std::vector<ObjectRef>> inputs_;
  /*!
   * \brief The "physical" devices the VM can execute primitives on. All "device indexes"
   * are w.r.t. this vector. Each entry in this vector must match the corresponding entry
   * in the executable's "virtual" devices vector.
   */
  std::vector<Device> devices_;
  /*! \brief The cached memory allocators, one per device. */
  std::vector<Allocator*> allocators_;
  /*!
   * \brief The constant pool for runtime. It caches the device dependent
   * object to avoid rellocation of constants during inference.
   */
  std::vector<ObjectRef> const_pool_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_VM_H_
