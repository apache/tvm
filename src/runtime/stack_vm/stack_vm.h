/*!
 *  Copyright (c) 2016 by Contributors
 * \file stack_vm.h
 * \brief A simple stack-based virtual machine.
 *
 *  This can be used to interepret host side code
 *  to setup calls into device functions
 *  when only Runtime compilation for device is available(via NVRTC or OpenCL).
 */
#ifndef TVM_RUNTIME_STACK_VM_STACK_VM_H_
#define TVM_RUNTIME_STACK_VM_STACK_VM_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief A simple stack-based virtual machine.
 */
class StackVM {
 public:
  /*!
   * \brief Invoke the StackVM as PackedFunc
   * \param args The arguments to the StackVM.
   */
  void operator()(const TVMArgs& args) const;
  /*!
   * \brief The opcode of stack vm
   * \note Notation
   *  - sp Stack pointer
   *  - pc Program pointer
   */
  enum OpCode {
    // integer ops
    ADD_I64,
    SUB_I64,
    MUL_I64,
    DIV_I64,
    MOD_I64,
    EQ_I64,
    LT_I64,
    LE_I64,
    // floating ops
    ADD_F64,
    SUB_F64,
    MUL_F64,
    DIV_F64,
    EQ_F64,
    LT_F64,
    LE_F64,
    // load operation
    ADDR_LOAD_UINT32,
    ADDR_LOAD_INT32,
    ADDR_LOAD_INT64,
    ADDR_LOAD_FP64,
    ADDR_LOAD_HANDLE,
    // store operations
    // *(stack[sp - 1].v_andle) = stack[sp].v_int64
    // sp = sp - 2;
    ADDR_STORE_INT64,
    /*!
     * \brief Quick routine to load uint32 from constant offset.
     * \code
     *  stack[sp].v_int64 = ((uint32_t*)stack[sp].v_handle)[code[pc + 1].v_int];
     *  pc = pc + 2;
     * \endcode
     */
    ARRAY_LOAD_UINT32,
    // logical ops
    NOT,
    /*!
     * \brief Add address by an offset.
     * \code
     *  stack[sp - 1].v_handle = ((char*)stack[sp - 1].v_handle + stack[sp].v_int64);
     *  sp = sp - 1;
     * \endcode
     */
    ADDR_ADD,
    /*!
     * \brief push integer fetched from next pc position into stack
     * \code
     *  stack[sp + 1].v_int64 = code[pc + 1].v_int;
     *  pc = pc + 2;
     *  sp = sp + 1;
     * \endcode
     */
    PUSH_I64,
    /*!
     * \brief push a value given relative index on the stack
     * \code
     *  stack[sp + 1] = stack[sp + code[pc + 1].v_int];
     *  pc = pc + 2;
     *  sp = sp + 1;
     * \endcode
     */
    PUSH_VALUE,
    /*!
     * \brief Load data from heap to top of stack
     * \code
     *  stack[sp + 1] = heap[code[pc + 1].v_int];
     *  pc = pc + 2;
     *  sp = sp + 1;
     * \endcode
     */
    LOAD_HEAP,
    /*!
     * \brief Store data to heap
     * \code
     *  heap[code[pc + 1].v_int] = stack[sp];
     *  sp = sp - 1;
     * \endcode
     */
    STORE_HEAP,
    /*! \brief pop value from top of the stack */
    POP,
    /*!
     * \brief select based on operands.
     * \code
     *  stack[sp - 2] = stack[sp].v_int64 ? stack[sp - 2] : stack[sp - 1]
     *  sp = sp - 2;
     * \endcode
     */
    SELECT,
    /*!
     * \brief call an extern packed function
     * \code
     *  num_args = stack[sp].v_int64;
     *  call_fid = code[pc + 1].v_int;
     *  f = extern_func[call_fid];
     *  int* type_codes = &(code[pc + 2].v_int)
     *  stack[sp - num_args] = f(&stack[sp - num_args], type_codes, num_args);
     *  sp = sp - num_args;
     *  // The type codes are hidden in the code space.
     *  pc = pc + 2 + num_args
     * \endcode
     */
    CALL_PACKED_FUNC,
    /*!
     * \brief Assert condition is true.
     * \code
     *  CHECK(stack[sp]) << str_data[code[pc + 1].v_int];
     *  sp = sp - 1;
     * \endcode
     */
    ASSERT,
    /*!
     * \brief Relative Jump if the condition is true,
     *  Does not change the stack status.
     * \code
     *  if (stack[sp]) {
     *    pc += code[pc + 1].v_int
     *  } else {
     *    pc = pc + 2;
     *  }
     * \endcode
     */
    RJUMP_IF_TRUE,
    /*!
     * \brief Relative Jump if the condition is true,
     *  Does not change the stack status.
     * \code
     *  if (stack[sp]) {
     *    pc += code[pc + 1].v_int
     *  } else {
     *    pc = pc + 2;
     *  }
     * \endcode
     */
    RJUMP_IF_FALSE,
    /*!
     * \brief Relative jump to a location.
     * \code
     *  pc += code[pc + 1].v_int;
     * \endcode
     */
    RJUMP,
    /*!
     * \brief debug instruction.
     * \code
     *  CHECK_EQ(sp, code[pc + 1]).v_int;
     *  pc += 2;
     * \code
     */
    ASSERT_SP,
    // Intrinsics for API function,
    TVM_LOAD_ARG_INT64,
    TVM_LOAD_ARG_FP64,
    TVM_LOAD_ARG_HANDLE,
    TVM_ARRAY_GET_DATA,
    TVM_ARRAY_GET_SHAPE,
    TVM_ARRAY_GET_STRIDES,
    TVM_ARRAY_GET_NDIM,
    TVM_ARRAY_GET_TYPE_CODE,
    TVM_ARRAY_GET_TYPE_BITS,
    TVM_ARRAY_GET_TYPE_LANES
  };
  /*! \brief The code structure */
  union Code {
    OpCode op_code;
    int v_int;
  };
  /*! \brief The state object of StackVM */
  struct State {
    /*! \brief The execution stack */
    std::vector<TVMValue> stack;
    /*! \brief The global heap space */
    std::vector<TVMValue> heap;
    /*! \brief stack pointer  */
    int64_t sp{0};
    /*! \brief program counter */
    int64_t pc{0};
  };
  /*! \brief execute the stack vm with given state */
  void Run(State* state) const;
  /*!
   * \brief Print instruction at location pc
   * \param os The ostream
   * \param pc The pc
   * \return the pc to next instruction.
   */
  int64_t PrintCode(std::ostream&os, int64_t pc) const;  // NOLINT(*)
  /*! \brief Get thread local state of the stack VM */
  static State* ThreadLocalState();
  /*! \brief The instructions */
  std::vector<Code> code;
  /*! \brief constant error messages */
  std::vector<std::string> str_data;
  /*! \brief Extern functions in packed func format */
  std::vector<runtime::PackedFunc> packed_func;
  /*! \brief name of each heap id*/
  std::vector<std::string> heap_id_name;
  /*! \brief The memory size needed */
  size_t heap_size{0};
  /*! \brief The stack size required */
  size_t stack_size{1024};
  /*!
   * \brief Convert I64 opcode to F64 Ones
   * \param code The op code.
   * \return the F64 op code.
   */
  static OpCode CodeI64ToF64(OpCode code) {
    switch (code) {
      case ADD_I64: return ADD_F64;
      case SUB_I64: return SUB_F64;
      case MUL_I64: return MUL_F64;
      case DIV_I64: return DIV_F64;
      case EQ_I64: return EQ_F64;
      case LT_I64: return LT_F64;
      case LE_I64: return LE_F64;
      case MOD_I64: LOG(FATAL) << "cannot handle mod for float";
      default: LOG(FATAL) << "cannot handle op " << code; return ADD_F64;
    }
  }
  /*!
   * \brief Get load opcode for type t
   * \param t the type code.
   * \return The load opcode
   */
  static OpCode GetLoad(TVMType t) {
    CHECK_EQ(t.lanes, 1U);
    if (t.code == kHandle) return ADDR_LOAD_HANDLE;
    if (t.code == kInt) {
      switch (t.bits) {
        case 32 : return ADDR_LOAD_INT32;
        case 64 : return ADDR_LOAD_INT64;
      }
    } else if (t.code == kUInt) {
      switch (t.bits) {
        case 32 : return ADDR_LOAD_UINT32;
      }
    } else if (t.code == kFloat) {
      switch (t.bits) {
        case 64 : return ADDR_LOAD_FP64;
      }
    }
    LOG(FATAL) << "Cannot load type " << t;
    return ADDR_LOAD_FP64;
  }
  /*!
   * \brief Get store opcode for type t
   * \param t the type code.
   * \return The load opcode
   */
  static OpCode GetStore(TVMType t) {
    CHECK_EQ(t.lanes, 1U);
    if (t.code == kInt) {
      switch (t.bits) {
        case 64 : return ADDR_STORE_INT64;
      }
    }
    LOG(FATAL) << "Cannot store type " << t;
    return ADDR_LOAD_FP64;
  }
  friend std::ostream& operator<<(std::ostream& os, const StackVM& vm);  // NOLINT(*)
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_STACK_VM_STACK_VM_H_
