/*!
 *  Copyright (c) 2016 by Contributors
 * \file stack_vm.h
 * \brief A simple stack-based virtual machine.
 *
 *  This can be used to interepret host side code
 *  to setup calls into device functions
 *  when only Runtime compilation for device is available(via NVRTC or OpenCL).
 */
#ifndef TVM_CODEGEN_STACK_VM_STACK_VM_H_
#define TVM_CODEGEN_STACK_VM_STACK_VM_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

using runtime::operator<<;
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
    // Pointer comparison
    EQ_HANDLE,
    /*!
     * \brief Routine to load data from address with const offset.
     * \code
     *  stack[sp].v_int64 = ((DType*)stack[sp].v_handle)[code[pc + 1].v_int];
     *  pc = pc + 2;
     * \endcode
     */
    ARRAY_LOAD_UINT32,
    ARRAY_LOAD_INT32,
    ARRAY_LOAD_INT64,
    ARRAY_LOAD_FP64,
    ARRAY_LOAD_HANDLE,
    ARRAY_LOAD_TVMVALUE,
    /*!
     * \brief Routine to store data from constant offset.
     * \code
     *  ((DType*)stack[sp - 1].v_handle)[code[pc + 1].v_int] = stack[sp];
     *  pc = pc + 2;
     *  sp = sp - 2;
     * \endcode
     */
    ARRAY_STORE_UINT32,
    ARRAY_STORE_INT32,
    ARRAY_STORE_INT64,
    ARRAY_STORE_FP64,
    ARRAY_STORE_HANDLE,
    ARRAY_STORE_TVMVALUE,
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
    /*!
     * \brief call an extern packed function
     * \code
     *  value_stack = stack[sp - 1].v_handle;
     *  type_stack = stack[sp - 0].v_handle;
     *  call_fid = code[pc + 1].v_int;
     *  begin = code[pc + 2].v_int;
     *  end = code[pc + 3].v_int;
     *  num_args = end - begin - 1;
     *  f = extern_func[call_fid];
     *  stack[sp - 1] = f(&value_stack[begin:end-1], type_stack[begin:end-1], num_args);
     *  sp = sp - 1;
     *  // The type codes are hidden in the code space.
     *  pc = pc + 4
     * \endcode
     */
    CALL_PACKED_LOWERED,
    // Allocate things on stack
    /*!
     * \brief allocate data from stack.
     * \code
     *  num = code[pc + 1].v_int;
     *  void* addr = &stack[sp];
     *  sp = sp + num;
     *  stack[sp].v_handle = addr;
     *  pc = pc + 1;
     * \endcode
     */
    TVM_STACK_ALLOCA_BY_8BYTE,
    /*!
     * \brief allocate data from device.
     * \code
     *  device_type = stack[sp - 2].v_int64;
     *  device_id = stack[sp - 1].v_int64;
     *  nbytes = stack[sp].v_int64;
     *  stack[sp - 2].v_handle = device_alloca(device_type, device_id, nbytes);
     *  sp = sp - 2;
     *  pc = pc + 1;
     * \endcode
     */
    TVM_DEVICE_ALLOCA,
    /*!
     * \brief free data into device.
     * \code
     *  device_type = stack[sp - 2].v_int64;
     *  device_id = stack[sp - 1].v_int64;
     *  ptr = stack[sp].v_handle;
     *  stack[sp - 2].v_int64 = device_free(device_type, device_id, ptr);
     *  sp = sp - 2;
     *  pc = pc + 1;
     * \endcode
     */
    TVM_DEVICE_FREE,
    /*!
     * \brief throw last error
     */
    TVM_THROW_LAST_ERROR,
    /*!
     * \brief get data from structure.
     * \code
     *  index = code[pc + 1].v_int;
     *  field = code[pc + 2].v_int;
     *  stack[sp] = ((StructType*)stack[sp].v_handle)[index]->field;
     *  pc = pc + 3
     * \endcode
     */
    TVM_STRUCT_GET,
    /*!
     * \brief set data into structure.
     * \code
     *  index = code[pc + 1].v_int;
     *  field = code[pc + 2].v_int;
     *  ((StructType*)stack[sp - 1].v_handle)[index]->field = stack[sp];
     *  pc = pc + 3
     *  sp = sp - 1
     * \endcode
     */
    TVM_STRUCT_SET
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
    /*! \brief extern functions */
    std::vector<PackedFunc> extern_func;
    /*! \brief stack pointer  */
    int64_t sp{0};
    /*! \brief program counter */
    int64_t pc{0};
  };
  /*! \brief The external function entries. */
  struct ExternFuncEntry {
    std::string name;
    runtime::PackedFunc func;
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
  /*! \brief The current module context of stackvm */
  runtime::ModuleNode* mod_ctx{nullptr};
  /*! \brief Extern functions */
  std::vector<std::string> extern_func_name;
  /*! \brief name of each heap id */
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
    if (t.code == kHandle) return ARRAY_LOAD_HANDLE;
    if (t.code == kInt) {
      switch (t.bits) {
        case 32 : return ARRAY_LOAD_INT32;
        case 64 : return ARRAY_LOAD_INT64;
      }
    } else if (t.code == kUInt) {
      switch (t.bits) {
        case 32 : return ARRAY_LOAD_UINT32;
      }
    } else if (t.code == kFloat) {
      switch (t.bits) {
        case 64 : return ARRAY_LOAD_FP64;
      }
    }
    LOG(FATAL) << "Cannot load type " << t;
    return ARRAY_LOAD_FP64;
  }
  /*!
   * \brief Get store opcode for type t
   * \param t the type code.
   * \return The load opcode
   */
  static OpCode GetStore(TVMType t) {
    CHECK_EQ(t.lanes, 1U);
    if (t.code == kHandle) return ARRAY_STORE_HANDLE;
    if (t.code == kInt) {
      switch (t.bits) {
        case 32 : return ARRAY_STORE_INT32;
        case 64 : return ARRAY_STORE_INT64;
      }
    } else if (t.code == kUInt) {
      switch (t.bits) {
        case 32 : return ARRAY_STORE_UINT32;
      }
    } else if (t.code == kFloat) {
      switch (t.bits) {
        case 64 : return ARRAY_STORE_FP64;
      }
    }
    LOG(FATAL) << "Cannot store type " << t;
    return ARRAY_STORE_FP64;
  }
  friend std::ostream& operator<<(std::ostream& os, const StackVM& vm);  // NOLINT(*)

 private:
  // get extern function.
  const PackedFunc& GetExtern(State* s, int fid) const;
};

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_STACK_VM_STACK_VM_H_
