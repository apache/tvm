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
 * \file src/runtime/relax_vm/vm.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/relax_vm/vm.h>

#include <optional>
#include <thread>

namespace tvm {
namespace runtime {
namespace relax_vm {

//---------------------------------------------
// VM Closure object
//---------------------------------------------
TVM_REGISTER_OBJECT_TYPE(VMClosureObj);

VMClosure::VMClosure(String func_name, ffi::Function impl) {
  auto ptr = make_object<VMClosureObj>();
  ptr->func_name = func_name;
  ptr->impl = std::move(impl);
  data_ = std::move(ptr);
}

/*!
 * \brief Create another ffi::Function with last arguments already bound to last_args.
 * \param func The input func, can be a VMClosure or ffi::Function.
 * \param last_args The arguments to bound to in the end of the function.
 * \note The new function takes in arguments and append the last_args in the end.
 */
ffi::Function VMClosure::BindLastArgs(ffi::Function func, std::vector<Any> last_args) {
  return ffi::Function([func, last_args](ffi::PackedArgs args, ffi::Any* rv) {
    std::vector<AnyView> packed_args(args.size() + last_args.size());
    std::copy(args.data(), args.data() + args.size(), packed_args.data());
    for (size_t i = 0; i < last_args.size(); ++i) {
      packed_args[args.size() + i] = last_args[i];
    }
    func.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), rv);
  });
}

//-----------------------------------------------------------
// Utility functions.
//-----------------------------------------------------------
// Use the args after `starting_arg_idx` as a series of indices into `obj`,
// indexing into nested Array and returning the final indexed object.
Any IndexIntoNestedObject(Any obj, ffi::PackedArgs args, int starting_arg_idx) {
  for (int i = starting_arg_idx; i < args.size(); i++) {
    // the object must be an Array to be able to index into it
    if (!obj.as<ffi::ArrayObj>()) {
      LOG(FATAL) << "ValueError: Attempted to index into an object that is not an Array.";
    }
    int index = args[i].cast<int>();
    auto arr = Downcast<ffi::Array<Any>>(obj);
    // make sure the index is in bounds
    if (index >= static_cast<int>(arr.size())) {
      LOG(FATAL) << "IndexError: Invalid index (" << index << " >= " << arr.size() << ").";
    }
    obj = arr[index];
  }
  return obj;
}

NDArray ConvertNDArrayToDevice(NDArray src, const DLDevice& dev, Allocator* alloc) {
  if (src->device.device_type == dev.device_type && src->device.device_id == dev.device_id) {
    return src;
  } else {
    auto res = alloc->Empty(src.Shape(), src->dtype, dev);
    res.CopyFrom(src);
    return res;
  }
}

Any ConvertObjectToDevice(Any src, const Device& dev, Allocator* alloc) {
  if (src.as<NDArray::ContainerType>()) {
    return ConvertNDArrayToDevice(Downcast<NDArray>(src), dev, alloc);
  } else if (src.as<ffi::ArrayObj>()) {
    std::vector<Any> ret;
    auto arr = Downcast<ffi::Array<Any>>(src);
    for (size_t i = 0; i < arr.size(); i++) {
      ret.push_back(ConvertObjectToDevice(arr[i], dev, alloc));
    }
    return Array<Any>(ret.begin(), ret.end());
  } else {
    return src;
  }
}

ffi::Any ConvertArgToDevice(AnyView input, Device dev, Allocator* alloc) {
  // in terms of memory-behavior.
  // To be extra careful, we copy DLTensor.
  // The developer can still explicitly allocate NDArray
  // in TVM Native API or NDArray::FromDLPack to regain zero copy behavior.
  Any ret;
  if (auto opt_obj = input.as<ObjectRef>()) {
    ret = ConvertObjectToDevice(opt_obj.value(), dev, alloc);
  } else if (auto opt_dltensor = input.as<DLTensor*>()) {
    DLTensor* tensor = opt_dltensor.value();
    std::vector<int64_t> shape(tensor->shape, tensor->shape + tensor->ndim);
    auto dst = alloc->Empty(shape, tensor->dtype, dev);
    dst.CopyFrom(tensor);
    ret = dst;
  } else {
    ret = input;
  }
  return ret;
}

ffi::Any ConvertRegToDevice(ffi::Any input, Device dev, Allocator* alloc) {
  Any ret;
  if (auto opt_obj = input.as<ObjectRef>()) {
    ret = ConvertObjectToDevice(opt_obj.value(), dev, alloc);
  } else {
    ret = input;
  }
  return ret;
}

//-----------------------------------------------------------
// VM implementations.
//-----------------------------------------------------------
/*!
 * \brief The register type.
 */
using RegType = ffi::Any;

/*!
 * \brief A representation of a stack frame.
 *
 * A stack frame is a record containing the information needed
 * to restore the caller's virtual machine state after returning
 * from a function call.
 */
struct VMFrame {
  /*! \brief The return program counter. */
  Index return_pc;
  /*! \brief Statically allocated space for objects */
  std::vector<RegType> register_file;
  /*! \brief Register in caller's frame to put return value */
  RegName caller_return_register;
  // The following fields are used for ffi::Function call within
  // a single function scope. The space is reused across multiple
  // packed func calls to increase cache locality and avoid re-allocation
  /*! \brief Temporary argument value stack for packed func call. */
  std::vector<TVMValue> call_arg_values;
  /*! \brief Temporary argument tcode stack for packed func call. */
  std::vector<int> call_arg_tcodes;

  std::vector<AnyView> call_args;

  VMFrame(Index pc, Index register_file_size)
      : return_pc(pc), register_file(register_file_size), caller_return_register(0) {}

  void Clear() {
    this->caller_return_register = 0;
    this->call_args.clear();
    for (RegType& reg : register_file) {
      reg = nullptr;
    }
  }

  void ResetForRecycle(Index pc, Index register_file_size) {
    this->return_pc = pc;
    this->register_file.resize(register_file_size);
  }
};

class VirtualMachineImpl : public VirtualMachine {
 public:
  //---------------------------------------------------
  // Public facing functions overloading
  //---------------------------------------------------
  void LoadExecutable(ObjectPtr<VMExecutable> exec) final;
  void Init(const std::vector<Device>& devices,
            const std::vector<AllocatorType>& alloc_types) final;
  VMClosure GetClosure(const String& func_name) final {
    return this->GetClosureInternal(func_name, false).value();
  }
  void InvokeClosurePacked(const ObjectRef& closure_or_packedfunc, ffi::PackedArgs args,
                           ffi::Any* rv) final;
  void SetInstrument(ffi::Function instrument) final { this->instrument_ = instrument; }

  //---------------------------------------------------
  // Functions in the vtable of Module
  //---------------------------------------------------
  void _Init(ffi::PackedArgs args, ffi::Any* rv);
  void _SaveClosure(ffi::PackedArgs args, ffi::Any* rv);
  void _InvokeClosure(ffi::PackedArgs args, ffi::Any* rv);
  void _InvokeClosureStateful(std::string func_name);
  void _SetInstrument(ffi::PackedArgs args, ffi::Any* rv);
  void _GetOutputArity(ffi::PackedArgs args, ffi::Any* rv);
  void _GetOutput(ffi::PackedArgs args, ffi::Any* rv);
  void _SetInputWithoutParamModule(ffi::PackedArgs args, ffi::Any* rv);
  void _SetInputWithParamModule(ffi::PackedArgs args, ffi::Any* rv);
  int _GetFunctionArity(std::string func_name);
  std::string _GetFunctionParamName(std::string func_name, int index);
  ffi::Function _LookupFunction(const String& name);

  TVM_MODULE_VTABLE_BEGIN("relax.VirtualMachine");
  TVM_MODULE_VTABLE_ENTRY_PACKED("vm_initialization", &VirtualMachineImpl::_Init);
  TVM_MODULE_VTABLE_ENTRY_PACKED("save_function", &VirtualMachineImpl::_SaveClosure);
  TVM_MODULE_VTABLE_ENTRY_PACKED("invoke_closure", &VirtualMachineImpl::_InvokeClosure);
  TVM_MODULE_VTABLE_ENTRY("invoke_stateful", &VirtualMachineImpl::_InvokeClosureStateful);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_instrument", &VirtualMachineImpl::_SetInstrument);
  TVM_MODULE_VTABLE_ENTRY_PACKED("get_output_arity", &VirtualMachineImpl::_GetOutputArity);
  TVM_MODULE_VTABLE_ENTRY_PACKED("get_output", &VirtualMachineImpl::_GetOutput);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_input", &VirtualMachineImpl::_SetInputWithoutParamModule);
  TVM_MODULE_VTABLE_ENTRY_PACKED("set_input_with_param_module",
                                 &VirtualMachineImpl::_SetInputWithParamModule);
  TVM_MODULE_VTABLE_ENTRY("get_function_arity", &VirtualMachineImpl::_GetFunctionArity);
  TVM_MODULE_VTABLE_ENTRY("get_function_param_name", &VirtualMachineImpl::_GetFunctionParamName);
  TVM_MODULE_VTABLE_END_WITH_DEFAULT(&VirtualMachineImpl::_LookupFunction);

  //--------------------------------------------------
  // Additional support arguments functions for VM
  //--------------------------------------------------
  /*!
   * \brief Internal implementation of GetClosure which also allow none.
   * \param func_name The name of the function.
   * \param allow_missing Whether none is allowed.
   * \return The result
   */
  Optional<VMClosure> GetClosureInternal(const String& func_name, bool allow_missing);

  /*!
   * \brief Set inputs to a function.
   * \param func_name The function name.
   * \param args args[offset:] are arguments to the function. If the arguments are not of the
   * correct device for the function, they will be copied to the device.
   * \param with_param_module If set to true, the last argument will be a module and can be invoked
   *        to get the argument, this is mainly used for debugging purposes and setting composite
   * objects. \note This interface works when using VM over RPC by internally converting NDArray in
   * the arguments to DLTensor, which is supported in RPC where remote could only have a minimal C
   * runtime.
   */
  void SetInput(std::string func_name, bool with_param_module, ffi::PackedArgs args);

  /*!
   * \brief Look up whether the VM has a function by the given name.
   * \param func_name the function's name
   * \return The function, if it exists. Logs a fatal error if not.
   */
  VMFuncInfo LookupVMFuncInfo(const std::string& func_name);

  /*!
   * \brief Look up whether the VM has outputs for the given function.
   * \param func_name the function's name
   * \return The output, if it exists. Logs a fatal error if not.
   */
  RegType LookupVMOutput(const std::string& func_name);

  /*!
   * \brief Fully bind the argument of a global function and save it in the env.
   * \param func_name The global function name to be saved.
   * \param save_name The saved name of the function.
   * \param include_return Whether forward the return value, set it to false allows
   *        us to ignore forwarding return value, which can be helpful to do benchmarking
   *        in RPC environment when return value is complicated Array.
   *
   * \param args The arguments to bound to the function.
   * \note This function is used by RPC server to help benchmarking.
   */
  void SaveClosure(const String& func_name, const String& save_name, bool include_return,
                   ffi::PackedArgs args);
  /*!
   * \brief Internal function to invoke a closure.
   * \param closure_or_packed The closure to be invoked.
   * \param args The arguments to the function.
   * \return The result value.
   */
  RegType InvokeClosureInternal(const ObjectRef& closure_or_packed,
                                const std::vector<RegType>& args);
  /*!
   * \brief Invoke a VM function by interpreting bytecode.
   * \param fidx The function index.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  RegType InvokeBytecode(Index fidx, const std::vector<RegType>& args);

 protected:
  /*!
   * \brief Get function by querying all of the current module's imports.
   * \param name The name of the function.
   * \return The result function, can return ffi::Function(nullptr) if nothing is found.
   */
  ffi::Function GetFuncFromImports(const String& name) {
    for (auto& lib : this->imports_) {
      ffi::Function func = lib->GetFunction(name, true);
      if (func.defined()) return func;
    }
    return ffi::Function(nullptr);
  }
  /*!
   * \brief Initialize function pool.
   */
  void InitFuncPool();

  /*!
   * \brief A RAII wrapper that pushes and pops VM frames.
   */
  class FrameGuard {
   public:
    VirtualMachineImpl* vm;
    explicit FrameGuard(VirtualMachineImpl* vm, std::unique_ptr<VMFrame> frame) : vm(vm) {
      vm->frames_.emplace_back(std::move(frame));
    }
    ~FrameGuard() {
      ICHECK_GT(vm->frames_.size(), 0);
      vm->pc_ = vm->frames_.back()->return_pc;
      vm->frames_.back()->Clear();
      vm->frame_free_list_.emplace_back(std::move(vm->frames_.back()));
      vm->frames_.pop_back();
    }
  };
  //-------------------------------------------------
  // Instruction interpretations.
  //-------------------------------------------------
  /*!
   * \brief Push a call frame onto the call stack.
   * \param ret_pc The program counter to return to.
   * \param vm_func The function to be pushed to the call stack.
   * \return A RAII wrapper that pops the frame when going out of scope.
   */
  FrameGuard PushFrame(Index ret_pc, const VMFuncInfo& vm_func) {
    std::unique_ptr<VMFrame> new_frame;
    if (!frame_free_list_.empty()) {
      new_frame = std::move(frame_free_list_.back());
      frame_free_list_.pop_back();
      new_frame->ResetForRecycle(ret_pc, vm_func.register_file_size);
    } else {
      new_frame = std::make_unique<VMFrame>(ret_pc, vm_func.register_file_size);
    }
    return FrameGuard(this, std::move(new_frame));
  }
  /*!
   * \brief Write to a VM register.
   * \param frame current vm frame.
   * \param reg The register to write to.
   * \param obj The object to write to.
   */
  TVM_ALWAYS_INLINE void WriteRegister(VMFrame* frame, RegName reg, const RegType& obj) {
    ICHECK_LT(reg, frame->register_file.size());
    frame->register_file[reg] = obj;
  }
  /*!
   * \brief Read a VM register.
   * \param frame current vm frame.
   * \param reg The register to read from.
   * \return The value of the register.
   */
  TVM_ALWAYS_INLINE RegType ReadRegister(VMFrame* frame, RegName reg) {
    if (reg < Instruction::kBeginSpecialReg) {
      return frame->register_file[reg];
    }
    RegType ret;
    if (reg == Instruction::kVoidRegister) {
      ret = nullptr;
    } else {
      ICHECK_EQ(reg, Instruction::kVMRegister);
      // per convention, ctx ptr must be VirtualMachine* casted to void.
      // this and VirtualMachine* may or may not be the same
      // do first cast to VirtualMachine* then to void*
      ret = static_cast<void*>(static_cast<VirtualMachine*>(this));
    }
    return ret;
  }
  /*!
   * \brief Run call instruction.
   * \param curr_frame The current frame.
   * \param inst The call instruction.
   */
  virtual void RunInstrCall(VMFrame* curr_frame, Instruction inst);

  /*! \brief Run VM dispatch loop. */
  void RunLoop();

  /*!
   * \brief Retrieve the name of the function identified by the given index.
   * \param idx The index into the VM executable function table.
   * \return The name of the function.
   */
  const std::string& GetFuncName(int idx) { return exec_->func_table[idx].name; }

  /*!
   * \brief Retrieve the inputs for a function.
   * \param func_name The name of the function.
   * \return The function inputs.
   */
  const std::vector<RegType>& GetInputsFor(const std::string& func_name) {
    return inputs_[func_name];
  }

  void ClearInputsFor(const std::string& func_name) { inputs_.erase(func_name); }

  //--------------------------------------------------------
  // Internal states for execution.
  //--------------------------------------------------------
  /*! \brief The loaded executable. */
  ObjectPtr<VMExecutable> exec_;
  /*! \brief The global constant pool */
  std::vector<ffi::Any> const_pool_;
  /*!
   * \brief Function pool to cache functions in func_table
   */
  std::vector<ffi::Any> func_pool_;
  //--------------------------------------------------------
  // Executor interface support
  //--------------------------------------------------------
  /*! \brief The function name to input register mapping. */
  std::unordered_map<std::string, std::vector<RegType>> inputs_;
  /*! \brief The function name to output register. */
  std::unordered_map<std::string, RegType> outputs_;
  /*! \brief A store of closures created by `save_function`. */
  std::unordered_map<std::string, VMClosure> saved_closures_;
  //------------------------------------------------------------
  // VM Instruction execution.
  //------------------------------------------------------------
  /*!
   * \brief The current stack of call frames.
   * \note: Use unique ptr to avoid re-allocation and copy when frames_ get resized.
   */
  std::vector<std::unique_ptr<VMFrame>> frames_;
  /*!
   * \brief A free list of frame
   */
  std::vector<std::unique_ptr<VMFrame>> frame_free_list_;

  /*! \brief The virtual machine PC. */
  Index pc_{0};
  /*! \brief The special return register. */
  RegType return_value_;
  /*!\ brief instrument function. */
  ffi::Function instrument_ = nullptr;
};

void VirtualMachineImpl::LoadExecutable(ObjectPtr<VMExecutable> exec) {
  this->exec_ = exec;
  this->imports_ = exec_->imports();
}

void VirtualMachineImpl::Init(const std::vector<Device>& devices,
                              const std::vector<AllocatorType>& alloc_types) {
  ICHECK_EQ(devices.size(), alloc_types.size());

  this->devices.reserve(devices.size());
  this->allocators.reserve(alloc_types.size());
  for (size_t i = 0; i < devices.size(); i++) {
    auto alloc = MemoryManager::GetOrCreateAllocator(devices[i], alloc_types[i]);
    this->devices.push_back(devices[i]);
    this->allocators.push_back(alloc);
  }
  // Setup constant sections.
  this->const_pool_.reserve(exec_->constants.size());
  for (const auto& constant : exec_->constants) {
    if (auto opt_nd = constant.as<NDArray>()) {
      this->const_pool_.push_back(ConvertRegToDevice(opt_nd.value(), devices[0], allocators[0]));
    } else {
      this->const_pool_.push_back(constant);
    }
  }
  // Setup function sections.
  this->InitFuncPool();
}

VMFuncInfo VirtualMachineImpl::LookupVMFuncInfo(const std::string& func_name) {
  ICHECK(exec_) << "The executable is not created yet.";
  auto it = this->exec_->func_map.find(func_name);
  CHECK(it != this->exec_->func_map.end()) << "ValueError: Unknown function: " << func_name;

  return exec_->func_table[it->second];
}

RegType VirtualMachineImpl::LookupVMOutput(const std::string& func_name) {
  if (!outputs_.count(func_name)) {
    LOG(FATAL) << "ValueError: No output saved for call of \"" << func_name
               << "\"; use `invoke_stateful` to call it first.";
  }
  return outputs_[func_name];
}

void VirtualMachineImpl::SetInput(std::string func_name, bool with_param_module,
                                  ffi::PackedArgs args) {
  const auto& m = exec_->func_map;
  if (m.find(func_name) != m.end()) {
    Index gf_idx = m.at(func_name);
    const VMFuncInfo& vm_func = exec_->func_table[gf_idx];
    size_t params_num = vm_func.num_args;
    ICHECK_EQ(args.size(), params_num)
        << "The number of provided parameters doesn't match the number of arguments for";
    std::vector<RegType> func_args(params_num);
    for (int i = 0; i < args.size(); ++i) {
      if (with_param_module && i == args.size() - 1) {
        // call param func to get the arguments(usually corresponds to param pack.)
        func_args[i] = (args[i].cast<Module>()).GetFunction("get_params")();
      } else {
        func_args[i] = ConvertArgToDevice(args[i], devices[0], allocators[0]);
      }
    }
    inputs_[func_name] = func_args;
  } else {
    LOG(FATAL) << "ValueError: Unknown function: " << func_name;
  }
}

//------------------------------------------
// Closure handling
//------------------------------------------
void VirtualMachineImpl::InvokeClosurePacked(const ObjectRef& closure_or_packedfunc,
                                             ffi::PackedArgs args, ffi::Any* rv) {
  // run packed call if it is a packed func.
  if (auto* packed = closure_or_packedfunc.as<ffi::Function::ContainerType>()) {
    packed->CallPacked(args.data(), args.size(), rv);
    return;
  }
  // run closure call.
  auto* clo = closure_or_packedfunc.as<VMClosureObj>();
  ICHECK(clo != nullptr) << "Function expects a closure or ffi::Function ";

  std::vector<AnyView> packed_args(args.size() + 1);
  // per convention, ctx ptr must be VirtualMachine* casted to void.
  // this and VirtualMachine* may or maynot be the same
  // do first cast to VirtualMachine* then to void*
  packed_args[0] = static_cast<void*>(static_cast<VirtualMachine*>(this));
  std::copy(args.data(), args.data() + args.size(), packed_args.begin() + 1);
  {
    NVTXScopedRange scope("RelaxVM: " + clo->func_name);
    clo->impl.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), rv);
  }
}

// internal variant version of invoke closurepacked
RegType VirtualMachineImpl::InvokeClosureInternal(const ObjectRef& closure_or_packed,
                                                  const std::vector<RegType>& args) {
  RegType ret;
  auto* packed = closure_or_packed.as<ffi::Function::ContainerType>();
  auto* clo = closure_or_packed.as<VMClosureObj>();
  int clo_offset = clo != nullptr ? 1 : 0;

  std::vector<AnyView> packed_args(args.size() + clo_offset);

  if (clo != nullptr) {
    packed_args[0] = static_cast<void*>(static_cast<VirtualMachine*>(this));
  }
  for (size_t i = 0; i < args.size(); ++i) {
    packed_args[i + clo_offset] = args[i];
  }

  if (packed != nullptr) {
    packed->CallPacked(packed_args.data(), packed_args.size(), &ret);
  } else {
    ICHECK(clo != nullptr);
    clo->impl.CallPacked(packed_args.data(), packed_args.size(), &ret);
  }
  return ret;
}

void VirtualMachineImpl::SaveClosure(const String& func_name, const String& save_name,
                                     bool include_return, ffi::PackedArgs args) {
  VMClosure clo = this->GetClosure(func_name);
  std::vector<RegType> inputs(args.size());
  for (int i = 0; i < args.size(); ++i) {
    inputs[i] = ConvertArgToDevice(args[i], this->devices[0], this->allocators[0]);
  }
  ffi::Function impl = VMClosure::BindLastArgs(clo->impl, inputs);
  if (!include_return) {
    impl = ffi::Function([impl](ffi::PackedArgs args, ffi::Any* rv) {
      ffi::Any temp;
      impl.CallPacked(args, &temp);
    });
  }
  saved_closures_[save_name] = VMClosure(save_name, impl);
}

Optional<VMClosure> VirtualMachineImpl::GetClosureInternal(const String& func_name,
                                                           bool allow_missing) {
  // look up saved closures.
  auto saved_it = saved_closures_.find(func_name);
  if (saved_it != saved_closures_.end()) {
    return saved_it->second;
  }
  auto it = exec_->func_map.find(func_name);
  if (it == exec_->func_map.end()) {
    if (allow_missing) return std::nullopt;
    LOG(FATAL) << "ValueError: Unknown function: " << func_name;
  }

  Index gf_idx = it->second;
  const VMFuncInfo& finfo = exec_->func_table[gf_idx];

  if (finfo.kind == VMFuncInfo::FuncKind::kVMFunc) {
    // NOTE: should not capture strong ref to self and avoid cyclic ref.
    auto impl = ffi::Function([gf_idx](ffi::PackedArgs args, ffi::Any* rv) {
      // Per convention, ctx ptr is a VirtualMachine*
      VirtualMachine* ctx_ptr = static_cast<VirtualMachine*>(args[0].cast<void*>());

      std::vector<RegType> inputs(args.size() - 1);
      for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = args[i + 1];
      }
      *rv = static_cast<VirtualMachineImpl*>(ctx_ptr)->InvokeBytecode(gf_idx, inputs);
    });
    return VMClosure(func_name, impl);
  } else {
    ICHECK(finfo.kind == VMFuncInfo::FuncKind::kVMTIRFunc)
        << "Cannot support closure with function kind " << static_cast<int>(finfo.kind);
    ffi::Function tir_func = GetFuncFromImports("__vmtir__" + finfo.name);
    ICHECK(tir_func != nullptr) << "Cannot find underlying compiled tir function of VMTIRFunc "
                                << finfo.name;
    auto impl = ffi::Function([this, finfo, tir_func](ffi::PackedArgs args, ffi::Any* rv) {
      // Per convention, ctx ptr is a VirtualMachine*
      VirtualMachine* ctx_ptr = static_cast<VirtualMachine*>(args[0].cast<void*>());
      ICHECK(ctx_ptr == this);
      ICHECK_EQ(args.size() - 1, finfo.num_args)
          << "Function " << finfo.name << " expects " << finfo.num_args << " arguments";
      ICHECK_GE(finfo.register_file_size, finfo.num_args + 1);
      std::vector<ffi::Any> reg_file(finfo.register_file_size);
      for (int64_t i = 0; i < finfo.num_args; ++i) {
        reg_file[i] = args[i + 1];
      }
      void* reg_anylist_handle = reg_file.data();
      void* const_anylist_handle = this->const_pool_.data();
      void* func_anylist_handle = this->func_pool_.data();
      tir_func(static_cast<void*>(ctx_ptr), reg_anylist_handle, const_anylist_handle,
               func_anylist_handle);
      // Return value always stored after inputs.
      *rv = reg_file[finfo.num_args];
    });
    return VMClosure(func_name, impl);
  }
}

//--------------------------------------------------------------------
// Instruction interpretations.
//--------------------------------------------------------------------
RegType VirtualMachineImpl::InvokeBytecode(Index gf_idx, const std::vector<RegType>& args) {
  const VMFuncInfo& gfunc = exec_->func_table[gf_idx];
  ICHECK(gfunc.kind == VMFuncInfo::FuncKind::kVMFunc);

  // Get the curr instr which might be a potential caller.
  Instruction curr_instr = exec_->GetInstruction(pc_);
  auto guard = PushFrame(this->pc_, gfunc);
  // Get new frame and set the caller info.
  VMFrame* curr_frame = frames_.back().get();
  if (curr_instr.op == Opcode::Call) {
    curr_frame->caller_return_register = curr_instr.dst;
  }

  // load arguments to the register file
  ICHECK_EQ(static_cast<size_t>(gfunc.num_args), args.size()) << "ValueError: Invoking function "
                                                              << gfunc.name << " expects "
                                                              << gfunc.num_args << " arguments" <<
      [&]() {
        std::stringstream ss;
        if (gfunc.param_names.size()) {
          ss << " (";
          for (size_t i = 0; i < gfunc.param_names.size(); i++) {
            if (i) {
              ss << ", ";
            }
            ss << gfunc.param_names[i];
          }
          ss << ")";
        }
        return ss.str();
      }() << ", but " << args.size() << " arguments were provided.";
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(frames_.back().get(), i, args[i]);
  }
  // set program counter
  pc_ = gfunc.start_instr;
  RunLoop();
  return return_value_;
}

void VirtualMachineImpl::InitFuncPool() {
  func_pool_.resize(exec_->func_table.size());

  for (size_t func_index = 0; func_index < exec_->func_table.size(); ++func_index) {
    const VMFuncInfo& info = exec_->func_table[func_index];
    if (info.kind == VMFuncInfo::FuncKind::kPackedFunc) {
      // only look through imports first
      ffi::Function func = GetFuncFromImports(info.name);
      if (!func.defined()) {
        const auto p_func = tvm::ffi::Function::GetGlobal(info.name);
        if (p_func.has_value()) func = *(p_func);
      }
      ICHECK(func.defined())
          << "Error: Cannot find ffi::Function " << info.name
          << " in either Relax VM kernel library, or in TVM runtime ffi::Function registry, or in "
             "global Relax functions of the VM executable";
      func_pool_[func_index] = func;

    } else {
      ICHECK(info.kind == VMFuncInfo::FuncKind::kVMFunc ||
             info.kind == VMFuncInfo::FuncKind::kVMTIRFunc);
      auto clo = this->GetClosure(info.name);
      func_pool_[func_index] = clo;
    }
  }
}

void VirtualMachineImpl::RunInstrCall(VMFrame* curr_frame, Instruction instr) {
  DLOG(INFO) << "\n  pc = " << pc_ << ", execute: " << GetFuncName(instr.func_idx);
  int args_begin_offset = instrument_ != nullptr ? 4 : 0;
  // Use the call arg stack from the current frame to increase reuse
  // and avoid re-allocation
  curr_frame->call_args.resize(args_begin_offset + instr.num_args);

  // NOTE: no changes and resize to those vector ref(otherwise can leads to segfault)
  //       in the remainder part of the function.
  std::vector<AnyView>& call_args = curr_frame->call_args;

  for (Index i = 0; i < instr.num_args; ++i) {
    Instruction::Arg arg = instr.args[i];
    int arg_index = args_begin_offset + i;
    switch (arg.kind()) {
      case Instruction::ArgKind::kRegister: {
        call_args[arg_index] = ReadRegister(curr_frame, arg.value());
        break;
      }
      case Instruction::ArgKind::kImmediate: {
        call_args[arg_index] = arg.value();
        break;
      }
      case Instruction::ArgKind::kConstIdx: {
        call_args[arg_index] = this->const_pool_[arg.value()];
        break;
      }
      case Instruction::ArgKind::kFuncIdx: {
        ICHECK_LT(static_cast<size_t>(arg.value()), this->func_pool_.size());
        call_args[arg_index] = this->func_pool_[arg.value()];
        break;
      }
      default: {
        LOG(FATAL) << "ValueError: Unknown argument kind: " << int(arg.kind());
      }
    }
  }
  ffi::PackedArgs args(call_args.data() + args_begin_offset, instr.num_args);
  ffi::Any ret;

  ICHECK_LT(static_cast<size_t>(instr.func_idx), this->func_pool_.size());

  if (instrument_ == nullptr) {
    this->InvokeClosurePacked(func_pool_[instr.func_idx].cast<ObjectRef>(), args, &ret);
  } else {
    // insert light-weight instrument callback
    call_args[0] = func_pool_[instr.func_idx];
    call_args[1] = GetFuncName(instr.func_idx);
    call_args[2] = true;
    call_args[3] = nullptr;

    Any rv;
    // store dtype to str since py callback cannot handle dtype atm.
    std::vector<std::unique_ptr<std::string>> temp_dtype;
    for (int i = 0; i < instr.num_args; ++i) {
      if (call_args[i + args_begin_offset].type_index() == ffi::TypeIndex::kTVMFFIDataType) {
        std::string str_dtype =
            DLDataTypeToString(call_args[i + args_begin_offset].cast<DLDataType>());
        temp_dtype.emplace_back(std::make_unique<std::string>(str_dtype));
        call_args[i + args_begin_offset] = *temp_dtype.back();
      }
    }
    int ret_kind = static_cast<int>(VMInstrumentReturnKind::kNoOp);
    instrument_.CallPacked(call_args.data(), call_args.size(), &rv);
    if (auto opt_int = rv.as<int64_t>()) {
      ret_kind = opt_int.value();
    }
    if (ret_kind != static_cast<int>(VMInstrumentReturnKind::kSkipRun)) {
      this->InvokeClosurePacked(func_pool_[instr.func_idx].cast<ObjectRef>(), args, &ret);
      call_args[2] = false;
      call_args[3] = ret;
      instrument_.CallPacked(call_args.data(), call_args.size(), &rv);
    }
  }

  // save the return value to the register
  // saving to special register is a NOP
  if (instr.dst < Instruction::kBeginSpecialReg) {
    WriteRegister(curr_frame, instr.dst, ret);
  }
  // increment pc
  pc_++;
}

void VirtualMachineImpl::RunLoop() {
  VMFrame* curr_frame = frames_.back().get();

  while (true) {
    ICHECK_LT(static_cast<size_t>(pc_), exec_->instr_offset.size()) << "run into invalid section";
    Instruction instr = exec_->GetInstruction(pc_);
    switch (instr.op) {
      case Opcode::Call: {
        this->RunInstrCall(curr_frame, instr);
        break;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_value_ = ReadRegister(curr_frame, instr.result);
        RegName caller_return_register = curr_frame->caller_return_register;
        if (frames_.size() <= 1) {
          // directly return if no other frame in the call stack.
        } else {
          // return from a local call.
          // Update the current frame to be the parent frame.
          VMFrame* parent_frame = frames_.end()[-2].get();
          WriteRegister(parent_frame, caller_return_register, return_value_);
        }
        return;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        break;
      }
      case Opcode::If: {
        int64_t cond_val = ReadRegister(curr_frame, instr.cond).cast<int64_t>();
        if (cond_val != 0) {
          pc_++;
        } else {
          ICHECK_GT(instr.false_offset, 1);
          pc_ += instr.false_offset;
        }
        break;
      }
    }
  }
}

ObjectPtr<VirtualMachine> VirtualMachine::Create() { return make_object<VirtualMachineImpl>(); }

//--------------------------------------------------------------------
// FFI related code
//--------------------------------------------------------------------

void VirtualMachineImpl::_Init(ffi::PackedArgs args, ffi::Any* rv) {
  ICHECK_EQ(args.size() % 3, 0);
  std::vector<Device> devices;
  std::vector<AllocatorType> alloc_types;
  for (int i = 0; i < args.size(); i += 3) {
    int device_type = args[i].cast<int>();
    int device_id = args[i + 1].cast<int>();
    int alloc_type = args[i + 2].cast<int>();
    devices.push_back(Device{DLDeviceType(device_type), device_id});
    alloc_types.push_back(AllocatorType(alloc_type));
  }
  this->Init(devices, alloc_types);
}

void VirtualMachineImpl::_SaveClosure(ffi::PackedArgs args, ffi::Any* rv) {
  ICHECK_GE(args.size(), 3);
  std::string func_name = args[0].cast<std::string>();
  this->SaveClosure(func_name, args[1].cast<String>(), args[2].cast<bool>(), args.Slice(3));
}

void VirtualMachineImpl::_InvokeClosure(ffi::PackedArgs args, ffi::Any* rv) {
  this->InvokeClosurePacked(args[0].cast<ObjectRef>(), args.Slice(1), rv);
}

void VirtualMachineImpl::_InvokeClosureStateful(std::string func_name) {
  const std::unordered_map<std::string, Index>& m = this->exec_->func_map;
  if (m.find(func_name) == m.end()) {
    LOG(FATAL) << "ValueError: Unknown function: " << func_name;
  }
  if (!inputs_.count(func_name)) {
    LOG(FATAL) << "ValueError: No inputs set for stateful call of " << func_name
               << "; use `set_input` first.";
    return;
  }
  outputs_[func_name] = this->InvokeClosureInternal(func_pool_[m.at(func_name)].cast<ObjectRef>(),
                                                    inputs_[func_name]);
}

void VirtualMachineImpl::_SetInstrument(ffi::PackedArgs args, ffi::Any* rv) {
  if (args[0].as<ffi::Function>()) {
    this->SetInstrument(args[0].cast<ffi::Function>());
  } else {
    String func_name = args[0].cast<String>();
    const auto factory = tvm::ffi::Function::GetGlobal(func_name);
    CHECK(factory.has_value()) << "Cannot find factory " << func_name;
    ffi::Any rv;
    factory->CallPacked(args.Slice(1), &rv);
    this->SetInstrument(rv.cast<ffi::Function>());
  }
}

void VirtualMachineImpl::_GetOutputArity(ffi::PackedArgs args, ffi::Any* rv) {
  std::string func_name = args[0].cast<std::string>();
  RegType out = LookupVMOutput(func_name);
  Any obj = IndexIntoNestedObject(out, args, 1);
  if (const auto* arr = obj.as<ffi::ArrayObj>()) {
    *rv = static_cast<int>(arr->size());
  } else {
    *rv = -1;
  }
}

void VirtualMachineImpl::_GetOutput(ffi::PackedArgs args, ffi::Any* rv) {
  std::string func_name = args[0].cast<std::string>();
  RegType out = LookupVMOutput(func_name);
  Any obj = IndexIntoNestedObject(out, args, 1);
  if (obj.as<ffi::ArrayObj>()) {
    LOG(FATAL) << "ValueError: `get_output` cannot return a tuple for RPC compatibility. "
                  "Please specify another index argument.";
    return;
  }
  *rv = obj;
}

void VirtualMachineImpl::_SetInputWithoutParamModule(ffi::PackedArgs args, ffi::Any* rv) {
  std::string func_name = args[0].cast<std::string>();
  this->SetInput(func_name, false, args.Slice(1));
}

void VirtualMachineImpl::_SetInputWithParamModule(ffi::PackedArgs args, ffi::Any* rv) {
  std::string func_name = args[0].cast<std::string>();
  this->SetInput(func_name, true, args.Slice(1));
}

int VirtualMachineImpl::_GetFunctionArity(std::string func_name) {
  const VMFuncInfo& vm_func = LookupVMFuncInfo(func_name);
  return vm_func.param_names.size();
}

std::string VirtualMachineImpl::_GetFunctionParamName(std::string func_name, int index) {
  const VMFuncInfo& vm_func = LookupVMFuncInfo(func_name);
  if (static_cast<size_t>(index) >= vm_func.param_names.size()) {
    LOG(FATAL) << "ValueError: Invalid index for " << func_name << " (" << index << " out of "
               << vm_func.param_names.size() << ")";
  }
  return vm_func.param_names[index];
}

ffi::Function VirtualMachineImpl::_LookupFunction(const String& name) {
  if (Optional<VMClosure> opt = this->GetClosureInternal(name, true)) {
    return ffi::Function([clo = opt.value(), _self = GetRef<Module>(this)](ffi::PackedArgs args,
                                                                           ffi::Any* rv) -> void {
      auto* self = const_cast<VirtualMachineImpl*>(_self.as<VirtualMachineImpl>());
      ICHECK(self);
      self->InvokeClosurePacked(clo, args, rv);
    });
  }
  return ffi::Function(nullptr);
}

//----------------------------------------------------------------
// Profiler can be optionally disabled via a macro to reduce dep.
//----------------------------------------------------------------
#if TVM_RELAX_VM_ENABLE_PROFILER

/*!
 * \brief An extension of VirtualMachineImpl to support per-op profiling
 * It overrides RunInstrCall to add instrumentations around it.
 */
class VirtualMachineProfiler : public VirtualMachineImpl {
 public:
  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (name == "profile") {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        std::string f_name = args[0].cast<std::string>();
        VMClosure clo = this->GetClosure(f_name);

        std::vector<Device> devices;
        for (auto dev : this->devices) {
          if (dev.device_type > 0) {
            devices.push_back(dev);
          }
        }

        prof_ = profiling::Profiler(devices, {}, {{String("Executor"), String("VM")}});

        auto inputs = GetInputsFor(f_name);

        bool clear_inputs = false;
        if (inputs.size() == 0) {
          ICHECK(args.size() > 1) << "No input is provided";
          SetInput(f_name, false, args.Slice(1));
          inputs = GetInputsFor(f_name);
          clear_inputs = true;
        } else {
          ICHECK_EQ(args.size(), 1) << "Inputs are already provided by set_input.";
        }

        // warmup
        this->InvokeClosureInternal(clo, inputs);

        prof_->Start();
        this->InvokeClosureInternal(clo, inputs);
        prof_->Stop();

        // Return the report as json, since profiling::Report object is not supported by RPC
        std::string report_json = prof_->Report()->AsJSON();
        *rv = report_json;

        prof_ = std::nullopt;  // releases hardware counters
        if (clear_inputs) {
          // SetInput modifies the internal states of VM. Undo the change after profiling.
          ClearInputsFor(f_name);
        }
      });
    } else {
      return VirtualMachineImpl::GetFunction(name, sptr_to_self);
    }
  }

 protected:
  void RunInstrCall(VMFrame* curr_frame, Instruction inst) override {
    bool profiling = false;
    if (prof_ && prof_->IsRunning()) {
      auto f_name = GetFuncName(inst.func_idx);
      std::optional<Device> dev;
      std::vector<NDArray> arrs;

      auto f_check_ndarray_arg = [&dev, &arrs](const RegType& arg) {
        if (auto opt_nd = arg.as<NDArray>()) {
          NDArray arr = opt_nd.value();
          if (arr.defined()) {
            dev = arr->device;
            arrs.push_back(arr);
          }
        }
      };

      for (Index i = 0; i < inst.num_args; ++i) {
        Instruction::Arg arg = inst.args[i];
        if (arg.kind() == Instruction::ArgKind::kRegister) {
          auto reg = ReadRegister(curr_frame, arg.value());
          f_check_ndarray_arg(reg);
        } else if (arg.kind() == Instruction::ArgKind::kConstIdx) {
          const auto& const_val = this->const_pool_[arg.value()];
          f_check_ndarray_arg(const_val);
        }
      }

      std::unordered_map<std::string, ObjectRef> metrics;
      metrics["Argument Shapes"] = profiling::ShapeString(arrs);

      // If a suitable device is found, enable profiling.
      if (dev) {
        profiling = true;
        prof_->StartCall(f_name, *dev, metrics);
      }
    }

    VirtualMachineImpl::RunInstrCall(curr_frame, inst);

    if (profiling) {
      prof_->StopCall();
    }
  }

 private:
  std::optional<profiling::Profiler> prof_;
};

ObjectPtr<VirtualMachine> VirtualMachine::CreateProfiler() {
  return make_object<VirtualMachineProfiler>();
}

#else
ObjectPtr<VirtualMachine> VirtualMachine::CreateProfiler() {
  LOG(FATAL) << "Profiler support is disabled";
  return nullptr;
}
#endif  // TVM_RELAX_VM_ENABLE_PROFILER
}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
