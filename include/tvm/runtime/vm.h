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
 *  Copyright (c) 2018 by Contributors
 * \file tvm/runtime/vm.h
 * \brief A virtual machine for executing Relay programs.
 */
#ifndef TVM_RUNTIME_VM_H_
#define TVM_RUNTIME_VM_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*! \brief A register name. */
using RegName = int64_t;

/*! \brief An alias for the integer type used ubiquitously
 * in the VM.
 */
using Index = int64_t;

/*! \brief An enumeration of Relay's opcodes.
 *
 * The opcode is used to implement instruction
 * as a tagged union.
 */
enum class Opcode {
  Move = 0U,
  Ret = 1U,
  Invoke = 2U,
  InvokeClosure = 3U,
  InvokePacked = 4U,
  AllocTensor = 5U,
  AllocTensorReg = 6U,
  AllocDatatype = 7U,
  AllocClosure = 8U,
  GetField = 9U,
  If = 10U,
  LoadConst = 11U,
  Goto = 12U,
  GetTag = 13U,
  LoadConsti = 14U,
  Fatal = 15U,
};

/*! \brief A single virtual machine instruction.
 *
 * The representation of the instruction is as
 * a tagged union.
 *
 * The first field represents which instruction,
 * and by extension which field of the union
 * is active.
 */
struct Instruction {
  /*! \brief The instruction opcode. */
  Opcode op;

  /*! \brief The destination register. */
  RegName dst;

  union {
    struct /* AllocTensor Operands */ {
      /*! \brief The number of dimensions. */
      uint32_t ndim;
      /*! \brief The shape of tensor. */
      int64_t* shape;
      /*! \brief The datatype of tensor to be allocated. */
      DLDataType dtype;
    } alloc_tensor;
    struct /* AllocTensorReg Operands */ {
      /*! \brief The register to read the shape out of. */
      RegName shape_register;
      /*! \brief The datatype of tensor to be allocated. */
      DLDataType dtype;
    } alloc_tensor_reg;
    struct /* InvokeClosure Operands */ {
      /*! \brief The register containing the closure. */
      RegName closure;
      /*! \brief The number of arguments to the closure. */
      Index num_closure_args;
      /*! \brief The closure arguments as an array. */
      RegName* closure_args;
    };
    struct /* Return Operands */ {
      /*! \brief The register to return. */
      RegName result;
    };
    struct /* Move Operands */ {
      /*! \brief The source register for a move operation. */
      RegName from;
    };
    struct /* InvokePacked Operands */ {
      /*! \brief The index into the packed function table. */
      Index packed_index;
      /*! \brief The arity of the packed function. */
      Index arity;
      /*! \brief The number of outputs produced by the packed function. */
      Index output_size;
      /*! \brief The arguments to pass to the packed function. */
      RegName* packed_args;
    };
    struct /* If Operands */ {
      /*! \brief The register containing the test value. */
      RegName test;
      /*! \brief The register containing the target value. */
      RegName target;
      /*! \brief The program counter offset for the true branch. */
      Index true_offset;
      /*! \brief The program counter offset for the false branch. */
      Index false_offset;
    } if_op;
    struct /* Invoke Operands */ {
      /*! \brief The function to call. */
      Index func_index;
      /*! \brief The number of arguments to the function. */
      Index num_args;
      /*! \brief The registers containing the arguments. */
      RegName* invoke_args_registers;
    };
    struct /* LoadConst Operands */ {
      /* \brief The index into the constant pool. */
      Index const_index;
    };
    struct /* LoadConsti Operands */ {
      /* \brief The index into the constant pool. */
      Index val;
    } load_consti;
    struct /* Jump Operands */ {
      /*! \brief The jump offset. */
      Index pc_offset;
    };
    struct /* Proj Operands */ {
      /*! \brief The register to project from. */
      RegName object;
      /*! \brief The field to read out. */
      Index field_index;
    };
    struct /* GetTag Operands */ {
      /*! \brief The register to project from. */
      RegName object;
    } get_tag;
    struct /* AllocDatatype Operands */ {
      /*! \brief The datatype's constructor tag. */
      Index constructor_tag;
      /*! \brief The number of fields to store in the datatype. */
      Index num_fields;
      /*! \brief The fields as an array. */
      RegName* datatype_fields;
    };
    struct /* AllocClosure Operands */ {
      /*! \brief The index into the function table. */
      Index clo_index;
      /*! \brief The number of free variables to capture. */
      Index num_freevar;
      /*! \brief The free variables as an array. */
      RegName* free_vars;
    };
  };

  /*! \brief Construct a return instruction.
   *  \param return_reg The register containing the return value.
   *  \return The return instruction.
   * */
  static Instruction Ret(RegName return_reg);
  /*! \brief Construct a fatal instruction.
   *  \return The fatal instruction.
   * */  
  static Instruction Fatal();
  /*! \brief Construct a invoke packed instruction.
   *  \param packed_index The index of the packed function.
   *  \param arity The arity of the function.
   *  \param output_size The number of outputs of the packed function.
   *  \param args The argument registers.
   *  \return The invoke packed instruction.
   */
  static Instruction InvokePacked(Index packed_index, Index arity, Index output_size,
                                  const std::vector<RegName>& args);
  /*! \brief Construct an allocate tensor instruction with constant shape.
   *  \param shape The shape of the tensor.
   *  \param dtype The dtype of the tensor.
   *  \param dst The destination register.
   *  \return The allocate tensor instruction.
   */
  static Instruction AllocTensor(std::vector<int64_t> shape, DLDataType dtype, RegName dst);
  /*! \brief Construct an allocate tensor instruction with register.
   *  \param shape_register The register containing the shape.
   *  \param dtype The dtype of the tensor.
   *  \param dst The destination register.
   *  \return The allocate tensor instruction.
   */
  static Instruction AllocTensorReg(RegName shape_register, DLDataType dtype, RegName dst);
  /*! \brief Construct an allocate datatype instruction.
   *  \param tag The datatype tag.
   *  \param num_fields The number of fields for the datatype.
   *  \param fields The registers containing the fields.
   *  \param dst The register name of the destination.
   *  \return The allocate instruction tensor.
   */
  static Instruction AllocDatatype(Index tag, Index num_fields, const std::vector<RegName>& fields,
                                   RegName dst);
  /*! \brief Construct an allocate closure instruction.
   *  \param func_index The index of the function table.
   *  \param num_freevar The number of free variables.
   *  \param free_vars The registers of the free variables.
   *  \param dst The destination register.
   *  \return The allocate closure instruction.
   */
  static Instruction AllocClosure(Index func_index, Index num_freevar,
                                  const std::vector<RegName>& free_vars, RegName dst);
  /*! \brief Construct a get field instruction.
   *  \param object_reg The register containing the object to project from.
   *  \param field_index The field to read out of the object.
   *  \param dst The destination register.
   *  \return The get field instruction.
   */
  static Instruction GetField(RegName object_reg, Index field_index, RegName dst);
  /*! \brief Construct a get_tag instruction.
   *  \param object_reg The register containing the object to project from.
   *  \param dst The destination register.
   *  \return The get_tag instruction.
   */
  static Instruction GetTag(RegName object_reg, RegName dst);
  /*! \brief Construct an if instruction.
   *  \param test The register containing the test value.
   *  \param target The register containing the target value.
   *  \param true_branch The offset to the true branch.
   *  \param false_branch The offset to the false branch.
   *  \return The if instruction.
   */
  static Instruction If(RegName test, RegName target, Index true_branch, Index false_branch);
  /*! \brief Construct a goto instruction.
   *  \param pc_offset The offset from the current pc.
   *  \return The goto instruction.
   */
  static Instruction Goto(Index pc_offset);
  /*! \brief Construct an invoke instruction.
   *  \param func_index The index of the function to invoke.
   *  \param args The registers containing the arguments.
   *  \param dst The destination register.
   *  \return The invoke instruction.
   */
  static Instruction Invoke(Index func_index, const std::vector<RegName>& args, RegName dst);
  /*! \brief Construct an invoke closure instruction.
   *  \param closure The register of the closure to invoke.
   *  \param args The registers containing the arguments.
   *  \param dst The destination register.
   *  \return The invoke closure instruction.
   */
  static Instruction InvokeClosure(RegName closure, const std::vector<RegName>& args, RegName dst);
  /*! \brief Construct a load constant instruction.
   *  \param const_index The index of the constant.
   *  \param dst The destination register.
   *  \return The load constant instruction.
   */
  static Instruction LoadConst(Index const_index, RegName dst);
  /*! \brief Construct a load_constanti instruction.
   *  \param val The interger constant value.
   *  \param dst The destination register.
   *  \return The load_constanti instruction.
   */
  static Instruction LoadConsti(Index val, RegName dst);
  /*! \brief Construct a move instruction.
   *  \param src The source register.
   *  \param dst The destination register.
   *  \return The move instruction.
   */
  static Instruction Move(RegName src, RegName dst);

  Instruction();
  Instruction(const Instruction& instr);
  Instruction& operator=(const Instruction& instr);
  ~Instruction();

  friend std::ostream& operator<<(std::ostream& os, const Instruction&);
};

/*! \brief A representation of a Relay function in the VM.
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
  Index register_file_size;

  VMFunction(const std::string& name, std::vector<std::string> params,
             const std::vector<Instruction>& instructions,
             Index register_file_size)
      : name(name),
        params(params),
        instructions(instructions),
        register_file_size(register_file_size) {}

  VMFunction() {}

  friend std::ostream& operator<<(std::ostream& os, const VMFunction&);
};

/*! \brief A representation of a stack frame.
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
  std::vector<Object> register_file;

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

/*! \brief The virtual machine.
 *
 * The virtual machine contains all the current execution state,
 * as well as the global view of functions, the global constant
 * table, the compiled operators.
 *
 * The goal is to have a single self-contained object,
 * enabling one to easily pass around VMs, execute them on
 * multiple threads, or serialized them to disk or over the
 * wire.
 */
class VirtualMachine : public runtime::ModuleNode {
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
  virtual PackedFunc GetFunction(const std::string& name,
                                 const std::shared_ptr<ModuleNode>& sptr_to_self);

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
                            Index output_size, const std::vector<Object>& args);

  virtual ~VirtualMachine() {}

  const char* type_key() const final {
    return "VirtualMachine";
  }

  /*! \brief The runtime module/library that contains generated code. */
  runtime::Module lib;
  /*! \brief The virtual machine's packed function table. */
  std::vector<PackedFunc> packed_funcs;
  /*! \brief The virtual machine's function table. */
  std::vector<VMFunction> functions;
  /*! \brief The current stack of call frames. */
  std::vector<VMFrame> frames;
  /*! \brief The global constant pool. */
  std::vector<Object> constants;
  /*! \brief The fuction table index of the current function. */
  Index func_index;
  /*! \brief The current pointer to the code section. */
  const Instruction* code;
  /*! \brief The virtual machine PC. */
  Index pc;

  /*! \brief The special return register. */
  Object return_register;

  /*! \brief The set of TVM contexts the VM is currently executing on. */
  std::vector<TVMContext> ctxs;

  /*! \brief Push a call frame on to the call stack. */
  void PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func);
  /*! \brief Pop a frame off the call stack.
   *  \return The number of frames left.
   */
  Index PopFrame();

  /*! \brief Write to a VM register.
   *  \param reg The register to write to.
   *  \param obj The object to write to.
   */
  inline void WriteRegister(RegName reg, const Object& obj);

  /*! \brief Read a VM register.
   *  \param reg The register to read from.
   *  \return The read object.
   */
  inline Object ReadRegister(RegName reg) const;

  /*! \brief Read a VM register and cast it to int32_t
   *  \param reg The register to read from.
   *  \return The read scalar.
   */
  int32_t LoadScalarInt(RegName reg) const;

  /*! \brief Invoke a VM function.
   * \param func The function.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  Object Invoke(const VMFunction& func, const std::vector<Object>& args);

  // TODO(@jroesch): I really would like this to be a global variable.
  /*! \brief Invoke a VM function by name.
   * \param name The function's name.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  Object Invoke(const std::string& name, const std::vector<Object>& args);

  VirtualMachine() : functions(), frames(), func_index(0), code(nullptr), pc(0) {}

  /*! \brief Initialize the virtual machine for a set of contexts.
   *  \param contexts The set of TVM contexts.
   */
  void Init(const std::vector<TVMContext>& contexts);

  /*! \brief Run VM dispatch loop.
   */
  void RunLoop();

  /*! \brief Get device context for params.
   */
  TVMContext GetParamsContext() const;

  /*!
   * \brief Load parameters from the parameter bytearray.
   * \param params The binary file that contains parameters.
   */
  void LoadParams(const std::string& params);

  /*! \brief A map from globals (as strings) to their index in the function map.
   */
  std::unordered_map<std::string, Index> global_map;

  /*! \brief A mapping from the packed function (as string) to the index that
   * corresponds to the position of the `packed_funcs` list.
   */
  std::unordered_map<std::string, Index> primitive_map;

 private:
  /*! \brief Invoke a global setting up the VM state to execute.
   *
   * This does not begin execution of the VM.
   */
  void InvokeGlobal(const VMFunction& func, const std::vector<Object>& args);


  /*! \brief The parameter name to data mapping. */
  std::unordered_map<std::string, Object> params_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_H_
