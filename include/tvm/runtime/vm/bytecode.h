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
 * \file tvm/runtime/vm/bytecode.h
 * \brief The bytecode for Relay virtual machine.
 */
#ifndef TVM_RUNTIME_VM_BYTECODE_H_
#define TVM_RUNTIME_VM_BYTECODE_H_

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

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
  AllocADT = 7U,
  AllocClosure = 8U,
  GetField = 9U,
  If = 10U,
  LoadConst = 11U,
  Goto = 12U,
  GetTag = 13U,
  LoadConsti = 14U,
  Fatal = 15U,
  AllocStorage = 16U,
  ShapeOf = 17U,
  ReshapeTensor = 18U,
  DeviceCopy = 19U,
  KillRegister = 20U,
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
      /*! \brief The storage to allocate from. */
      RegName storage;
      /*! \brief The offset into the storage to allocate from. */
      Index offset;
      /*! \brief The number of dimensions. */
      uint32_t ndim;
      /*! \brief The shape of tensor. */
      int64_t* shape;
      /*! \brief The datatype of tensor to be allocated. */
      DLDataType dtype;
    } alloc_tensor;
    struct /* AllocTensorReg Operands */ {
      /*! \brief The storage to allocate from. */
      RegName storage;
      /*! \brief The offset into the storage to allocate from. */
      Index offset;
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
      /*! \brief The index of the device on which the load will be made. */
      Index device_index;
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
    struct /* AllocADT Operands */ {
      // TODO(mbs): Needs a DeviceAndScope.
      /*! \brief The datatype's constructor tag. */
      Index constructor_tag;
      /*! \brief The number of fields to store in the datatype. */
      Index num_fields;
      /*! \brief The fields as an array. */
      RegName* datatype_fields;
    };
    struct /* AllocClosure Operands */ {
      // TODO(mbs): Needs a DeviceAndScope.
      /*! \brief The index into the function table. */
      Index clo_index;
      /*! \brief The number of free variables to capture. */
      Index num_freevar;
      /*! \brief The free variables as an array. */
      RegName* free_vars;
    };
    struct /* AllocStorage Operands */ {
      /*! \brief The alignment of the allocation. */
      Index alignment;
      /*! \brief The hint of the dtype. */
      DLDataType dtype_hint;
      /*! \brief The number of dimensions. */
      uint32_t ndim;
      union {
        /*! \brief The shape of tensor. */
        int64_t* shape;
        /*! \brief The size of the allocation. */
        RegName allocation_size;
      };
      /*! \brief The index of the device on which the allocation will be made. */
      Index device_index;
    } alloc_storage;
    struct /* ShapeOf Operands */ {
      RegName tensor;
    } shape_of;
    struct /* ReshapeTensor Operands */ {
      RegName tensor;
      RegName newshape;
    } reshape_tensor;
    struct /* DeviceCopy Operands */ {
      RegName src;
      /*! \brief The index of the source device to copy from. */
      Index src_device_index;
      /*! \brief The index of the destination deviceto copy to. */
      Index dst_device_index;
    } device_copy;
  };

  /*!
   * \brief Construct a return instruction.
   * \param return_reg The register containing the return value.
   * \return The return instruction.
   */
  static Instruction Ret(RegName return_reg);
  /*!
   * \brief Construct a fatal instruction.
   * \return The fatal instruction.
   */
  static Instruction Fatal();
  /*!
   * \brief Construct a invoke packed instruction.
   * \param packed_index The index of the packed function.
   * \param arity The arity of the function.
   * \param output_size The number of outputs of the packed function.
   * \param args The argument registers.
   * \return The invoke packed instruction.
   */
  static Instruction InvokePacked(Index packed_index, Index arity, Index output_size,
                                  const std::vector<RegName>& args);
  /*!
   * \brief Construct an allocate tensor instruction with constant shape.
   * \param storage The storage to allocate out of.
   * \param offset The offset to allocate at.
   * \param shape The shape of the tensor.
   * \param dtype The dtype of the tensor.
   * \param dst The destination register.
   * \return The allocate tensor instruction.
   */
  static Instruction AllocTensor(RegName storage, Index offset, const std::vector<int64_t>& shape,
                                 DLDataType dtype, RegName dst);
  /*!
   * \brief Construct an allocate tensor instruction with register.
   * \param storage The storage to allocate out of.
   * \param offset The offset into the storage to allocate from.
   * \param shape_register The register containing the shape.
   * \param dtype The dtype of the tensor.
   * \param dst The destination register.
   * \return The allocate tensor instruction.
   */
  static Instruction AllocTensorReg(RegName storage, Index offset, RegName shape_register,
                                    DLDataType dtype, RegName dst);
  /*!
   * \brief Construct an allocate datatype instruction.
   * \param tag The datatype tag.
   * \param num_fields The number of fields for the datatype.
   * \param fields The registers containing the fields.
   * \param dst The register name of the destination.
   * \return The allocate instruction tensor.
   */
  static Instruction AllocADT(Index tag, Index num_fields, const std::vector<RegName>& fields,
                              RegName dst);
  /*!
   * \brief Construct an allocate closure instruction.
   * \param func_index The index of the function table.
   * \param num_freevar The number of free variables.
   * \param free_vars The registers of the free variables.
   * \param dst The destination register.
   * \return The allocate closure instruction.
   */
  static Instruction AllocClosure(Index func_index, Index num_freevar,
                                  const std::vector<RegName>& free_vars, RegName dst);
  /*!
   * \brief Construct a get field instruction.
   * \param object_reg The register containing the object to project from.
   * \param field_index The field to read out of the object.
   * \param dst The destination register.
   * \return The get field instruction.
   */
  static Instruction GetField(RegName object_reg, Index field_index, RegName dst);
  /*!
   * \brief Construct a get_tag instruction.
   * \param object_reg The register containing the object to project from.
   * \param dst The destination register.
   * \return The get_tag instruction.
   */
  static Instruction GetTag(RegName object_reg, RegName dst);
  /*!
   * \brief Construct an if instruction.
   * \param test The register containing the test value.
   * \param target The register containing the target value.
   * \param true_branch The offset to the true branch.
   * \param false_branch The offset to the false branch.
   * \return The if instruction.
   */
  static Instruction If(RegName test, RegName target, Index true_branch, Index false_branch);
  /*!
   * \brief Construct a goto instruction.
   * \param pc_offset The offset from the current pc.
   * \return The goto instruction.
   */
  static Instruction Goto(Index pc_offset);
  /*!
   * \brief Construct an invoke instruction.
   * \param func_index The index of the function to invoke.
   * \param args The registers containing the arguments.
   * \param dst The destination register.
   * \return The invoke instruction.
   */
  static Instruction Invoke(Index func_index, const std::vector<RegName>& args, RegName dst);
  /*!
   * \brief Construct an invoke closure instruction.
   * \param closure The register of the closure to invoke.
   * \param args The registers containing the arguments.
   * \param dst The destination register.
   * \return The invoke closure instruction.
   */
  static Instruction InvokeClosure(RegName closure, const std::vector<RegName>& args, RegName dst);
  /*!
   * \brief Construct a load constant instruction.
   * \param const_index The index of the constant.
   * \param device_index The index of the device to load on.
   * \param dst The destination register.
   * \return The load constant instruction.
   */
  static Instruction LoadConst(Index const_index, Index device_index, RegName dst);
  /*!
   * \brief Construct a load_constanti instruction.
   * \param val The interger constant value.
   * \param dst The destination register.
   * \return The load_constanti instruction.
   */
  static Instruction LoadConsti(Index val, RegName dst);
  /*!
   * \brief Construct a move instruction.
   * \param src The source register.
   * \param dst The destination register.
   * \return The move instruction.
   */
  static Instruction Move(RegName src, RegName dst);
  /*!
   * \brief Allocate a storage block.
   * \param size The size of the allocation.
   * \param alignment The allocation's alignment.
   * \param dtype_hint The data type hint for the allocator.
   * \param device_index The index of the device to allocate on.
   * \param shape The shape of the allocation.
   * \param dst The destination to place the storage.
   * \return The alloc storage instruction.
   */
  static Instruction AllocStorage(RegName size, Index alignment, DLDataType dtype_hint,
                                  Index device_index, const std::vector<int64_t>& shape,
                                  RegName dst);
  /*!
   * \brief Get the shape of an input tensor.
   * \param tensor The input tensor.
   * \param dst The destination to store the shape of the given tensor.
   * \return The shape of instruction.
   */
  static Instruction ShapeOf(RegName tensor, RegName dst);
  /*!
   * \brief Reshape the tensor given the new shape.
   * \param tensor The input tensor.
   * \param newshape The shape tensor.
   * \param dst The destination to store the output tensor with new shape.
   * \return The reshape tensor instruction.
   */
  static Instruction ReshapeTensor(RegName tensor, RegName newshape, RegName dst);
  /*!
   * \brief Copy tensor cross different devices.
   * \param src The source register.
   * \param src_device_index The index of the device holding the tensor in the source register.
   * \param dst_device_index The index of the device to hold the tensor in the destination register.
   * \param dst The destination register to store the copied tensor.
   * \return The device copy instruction.
   */
  static Instruction DeviceCopy(RegName src, Index src_device_index, Index dst_device_index,
                                RegName dst);

  static Instruction KillRegister(RegName dst);

  Instruction();
  Instruction(const Instruction& instr);
  Instruction& operator=(const Instruction& instr);
  ~Instruction();

  friend std::ostream& operator<<(std::ostream& os, const Instruction&);
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_BYTECODE_H_
