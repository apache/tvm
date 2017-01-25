/*!
 *  Copyright (c) 2017 by Contributors
 * Implementation stack VM.
 * \file stack_vm.cc
 */
#include <dmlc/thread_local.h>
#include "./stack_vm.h"

namespace tvm {
namespace jit {

typedef dmlc::ThreadLocalStore<StackVM::State> StackVMStateStore;

StackVM::State* StackVM::ThreadLocalState() {
  return StackVMStateStore::Get();
}

#define STACK_VM_BINOP(OP, FIELD)                                 \
  {                                                               \
    stack[sp - 1].FIELD = stack[sp - 1].FIELD OP stack[sp].FIELD; \
    sp -= 1; pc += 1;                                             \
  }

#define STACK_VM_CMPOP(OP, FIELD)                                 \
  {                                                               \
    stack[sp - 1].v_int64 = stack[sp - 1].FIELD OP stack[sp].FIELD; \
    sp -= 1; pc += 1;                                             \
  }

#define STACK_VM_LOAD(FIELD, DST_TYPE, SRC_TYPE)                        \
  {                                                                     \
    stack[sp].FIELD = static_cast<DST_TYPE>(                            \
        *static_cast<SRC_TYPE*>(stack[sp].v_handle));                   \
    pc += 1;                                                            \
  }

#define STACK_VM_STORE(FIELD, DST_TYPE)                                 \
  {                                                                     \
    *static_cast<DST_TYPE*>(stack[sp - 1].v_handle) =                   \
        static_cast<DST_TYPE>(stack[sp].FIELD);                         \
    sp -= 2; pc += 1;                                                   \
  }

#define STACK_VM_TVM_LOAD_ARG(OP, TYPE)                                 \
  {                                                                     \
    TVMValue* args = static_cast<TVMValue*>(stack[sp - 2].v_handle);    \
    int64_t index = stack[sp].v_int64;                                  \
    int tc = static_cast<int*>(stack[sp - 1].v_handle)[index];          \
    CHECK(OP)                                                           \
        << " argument " << index << " is expected to be " << TYPE;      \
    stack[sp - 2] = args[index];                                        \
    sp -= 2;                                                            \
    pc += 1;                                                            \
  }


#define STACK_VM_TVM_ARRARY_GET(FIELD, TYPE, SFIELD)            \
  {                                                             \
    TVMArray* arr = static_cast<TVMArray*>(stack[sp].v_handle); \
    stack[sp].FIELD = (TYPE)(arr->SFIELD);                      \
    pc += 1;                                                    \
  }

#define STACK_VM_PRINT_CODE0(CODE)                          \
  case CODE:  {                                               \
    os << "[" << pc << "]\t" << #CODE << std::endl; return pc + 1;  \
  }

#define STACK_VM_PRINT_CODE1(CODE)                                      \
  case CODE:  {                                                         \
    os << "[" << pc << "]\t" << #CODE << " " << code[pc + 1].v_int << "\n" \
       <<  "[" << pc + 1 << "]" << std::endl;                               \
        return pc + 2;                                                  \
  }

#define STACK_VM_PRINT_HEAP_ACCESS(CODE)                                \
  case CODE:  {                                                         \
    os << "[" << pc << "]\t" << #CODE << " " << code[pc + 1].v_int      \
       << " " << heap_id_name[code[pc + 1].v_int] << "\n"               \
       <<  "[" << pc + 1 << "]" << std::endl;                           \
        return pc + 2;                                                  \
  }

#define STACK_VM_PRINT_JUMP(CODE)                                     \
  case CODE:  {                                                        \
    os << "[" << pc << "]\t" << #CODE << " rel=" << code[pc + 1].v_int \
       << " to " << pc + code[pc + 1].v_int << '\n'                    \
       << "[" << pc + 1 << "]" << std::endl;                         \
    return pc + 2;                                                     \
  }


int64_t StackVM::PrintCode(std::ostream& os, int64_t pc) const {
  switch (code[pc].op_code) {
    // int
    STACK_VM_PRINT_CODE0(ADD_I64);
    STACK_VM_PRINT_CODE0(SUB_I64);
    STACK_VM_PRINT_CODE0(MUL_I64);
    STACK_VM_PRINT_CODE0(MOD_I64);
    STACK_VM_PRINT_CODE0(DIV_I64);
    STACK_VM_PRINT_CODE0(EQ_I64);
    STACK_VM_PRINT_CODE0(LT_I64);
    STACK_VM_PRINT_CODE0(LE_I64);
    // floats
    STACK_VM_PRINT_CODE0(ADD_F64);
    STACK_VM_PRINT_CODE0(SUB_F64);
    STACK_VM_PRINT_CODE0(MUL_F64);
    STACK_VM_PRINT_CODE0(DIV_F64);
    STACK_VM_PRINT_CODE0(EQ_F64);
    STACK_VM_PRINT_CODE0(LT_F64);
    STACK_VM_PRINT_CODE0(LE_F64);
    // addressing load
    STACK_VM_PRINT_CODE0(ADDR_LOAD_UINT32);
    STACK_VM_PRINT_CODE0(ADDR_LOAD_INT32);
    STACK_VM_PRINT_CODE0(ADDR_LOAD_INT64);
    STACK_VM_PRINT_CODE0(ADDR_LOAD_FP64);
    STACK_VM_PRINT_CODE0(ADDR_LOAD_HANDLE);
    STACK_VM_PRINT_CODE0(ADDR_STORE_INT64);
    STACK_VM_PRINT_CODE1(ARRAY_LOAD_UINT32);
    STACK_VM_PRINT_CODE0(NOT);
    STACK_VM_PRINT_CODE0(ADDR_ADD);
    // stack ops
    STACK_VM_PRINT_CODE1(PUSH_I64);
    STACK_VM_PRINT_CODE1(PUSH_VALUE);
    STACK_VM_PRINT_CODE0(POP);
    STACK_VM_PRINT_CODE0(SELECT);
    STACK_VM_PRINT_HEAP_ACCESS(STORE_HEAP);
    STACK_VM_PRINT_HEAP_ACCESS(LOAD_HEAP);
    STACK_VM_PRINT_CODE1(CALL_EXTERN);
    STACK_VM_PRINT_CODE1(ASSERT);
    STACK_VM_PRINT_JUMP(RJUMP_IF_TRUE);
    STACK_VM_PRINT_JUMP(RJUMP_IF_FALSE);
    STACK_VM_PRINT_JUMP(RJUMP);
    STACK_VM_PRINT_CODE1(ASSERT_SP);
    // Intrinsics
    STACK_VM_PRINT_CODE0(TVM_LOAD_ARG_INT64);
    STACK_VM_PRINT_CODE0(TVM_LOAD_ARG_FP64);
    STACK_VM_PRINT_CODE0(TVM_LOAD_ARG_HANDLE);
    STACK_VM_PRINT_CODE0(TVM_ARRAY_GET_DATA);
    STACK_VM_PRINT_CODE0(TVM_ARRAY_GET_SHAPE);
    STACK_VM_PRINT_CODE0(TVM_ARRAY_GET_STRIDES);
    STACK_VM_PRINT_CODE0(TVM_ARRAY_GET_NDIM);
    STACK_VM_PRINT_CODE0(TVM_ARRAY_GET_TYPE_CODE);
    STACK_VM_PRINT_CODE0(TVM_ARRAY_GET_TYPE_BITS);
    STACK_VM_PRINT_CODE0(TVM_ARRAY_GET_TYPE_LANES);
  }
  LOG(FATAL) << "unknown op code " << code[pc].op_code;
  return 0;
}

std::ostream& operator<<(std::ostream& os, const StackVM& vm) {  // NOLINT(*)
  int64_t pc = 0;
  const int64_t code_size = static_cast<int64_t>(vm.code.size());
  os << "Program dump: code-size=" << code_size << '\n'
     << "----------begin-----------------\n";
  while (pc < code_size) {
    pc = vm.PrintCode(os, pc);
  }
  os << "----------end--------------------\n";
  return os;
}

void StackVM::Run(State* s) const {
  int64_t sp = s->sp;
  int64_t pc = s->pc;
  std::vector<TVMValue>& stack = s->stack;
  std::vector<TVMValue>& heap = s->heap;

  if (stack.size() < stack_size) {
    stack.resize(stack_size);
  }
  int64_t stack_cap = static_cast<int64_t>(stack_size - 4);
  if (heap.size() < heap_size) {
    heap.resize(heap_size);
  }
  const int64_t code_size = static_cast<int64_t>(code.size());

  while (pc < code_size) {
    switch (code[pc].op_code) {
      case ADD_I64: STACK_VM_BINOP(+, v_int64); break;
      case SUB_I64: STACK_VM_BINOP(-, v_int64); break;
      case MUL_I64: STACK_VM_BINOP(*, v_int64); break;
      case DIV_I64: STACK_VM_BINOP(/, v_int64); break;
      case MOD_I64: STACK_VM_BINOP(%, v_int64); break;
      case EQ_I64: STACK_VM_CMPOP(==, v_int64); break;
      case LT_I64: STACK_VM_CMPOP(<, v_int64); break;
      case LE_I64: STACK_VM_CMPOP(<=, v_int64); break;
      case ADD_F64: STACK_VM_BINOP(+, v_float64); break;
      case SUB_F64: STACK_VM_BINOP(-, v_float64); break;
      case MUL_F64: STACK_VM_BINOP(*, v_float64); break;
      case DIV_F64: STACK_VM_BINOP(/, v_float64); break;
      case EQ_F64: STACK_VM_CMPOP(==, v_float64); break;
      case LT_F64: STACK_VM_CMPOP(<, v_float64); break;
      case LE_F64: STACK_VM_CMPOP(<=, v_float64); break;
      // addressing
      case ADDR_LOAD_UINT32: STACK_VM_LOAD(v_int64, int64_t, uint32_t); break;
      case ADDR_LOAD_INT32: STACK_VM_LOAD(v_int64, int64_t, int32_t); break;
      case ADDR_LOAD_INT64: STACK_VM_LOAD(v_int64, int64_t, int64_t); break;
      case ADDR_LOAD_FP64: STACK_VM_LOAD(v_float64, double, double); break;
      case ADDR_LOAD_HANDLE: STACK_VM_LOAD(v_handle, void*, void*); break;
      case ADDR_STORE_INT64: STACK_VM_STORE(v_int64, int64_t); break;
      case ADDR_ADD: {
        stack[sp - 1].v_handle = (char*)(stack[sp - 1].v_handle) + stack[sp].v_int64;  // NOLINT(*)
        sp = sp - 1;
        pc = pc + 1;
        break;
      }
      case ARRAY_LOAD_UINT32: {
        stack[sp].v_int64 = ((uint32_t*)stack[sp].v_handle)[code[pc + 1].v_int];  // NOLINT(*)
        pc = pc + 2;
        break;
      }
      case NOT: {
        stack[sp].v_int64 = !stack[sp].v_int64;
        pc += 1;
        break;
      }
      case PUSH_I64: {
        stack[sp + 1].v_int64 = code[pc + 1].v_int;
        sp += 1;
        pc += 2;
        break;
      }
      case PUSH_VALUE: {
        int relpos = code[pc + 1].v_int;
        CHECK_LE(relpos, 0);
        stack[sp + 1] = stack[sp + relpos];
        sp += 1;
        pc += 2;
        break;
      }
      case POP: {
        sp -= 1;
        pc += 1;
        break;
      }
      case SELECT: {
        stack[sp - 2] = (stack[sp].v_int64 ? stack[sp - 2] : stack[sp - 1]);
        sp -= 2;
        pc += 1;
        break;
      }
      case LOAD_HEAP: {
        stack[sp + 1] = heap[code[pc + 1].v_int];
        sp += 1;
        pc += 2;
        break;
      }
      case STORE_HEAP: {
        heap[code[pc + 1].v_int] = stack[sp];
        sp -= 1;
        pc += 2;
        break;
      }
      case CALL_EXTERN: {
        int num_args = static_cast<int>(stack[sp].v_int64);
        int call_fid = code[pc + 1].v_int;
        stack[sp - num_args] = extern_func[call_fid](
            &stack[sp - num_args], num_args);
        sp = sp - num_args;
        pc += 2;
        break;
      }
      case ASSERT: {
        CHECK(stack[sp].v_int64) << str_data[code[pc + 1].v_int];
        sp -= 1;
        pc += 2;
        break;
      }
      case RJUMP_IF_TRUE: {
        if (stack[sp].v_int64) {
          pc += code[pc + 1].v_int;
        } else {
          pc += 2;
        }
        break;
      }
      case RJUMP_IF_FALSE: {
        if (!stack[sp].v_int64) {
          pc += code[pc + 1].v_int;
        } else {
          pc += 2;
        }
        break;
      }
      case RJUMP: {
        pc += code[pc + 1].v_int;
        break;
      }
      case ASSERT_SP: {
        int64_t expected = code[pc + 1].v_int;
        CHECK_EQ(sp, expected)
            << "sp assertion failed, expected="
            << expected << " now=" << sp << ", pc=" << pc;
        pc += 2;
        break;
      }
      case TVM_LOAD_ARG_INT64: {
        STACK_VM_TVM_LOAD_ARG(tc == kInt, "int"); break;
      }
      case TVM_LOAD_ARG_FP64: {
        STACK_VM_TVM_LOAD_ARG(tc == kFloat, "float"); break;
      }
      case TVM_LOAD_ARG_HANDLE: {
        STACK_VM_TVM_LOAD_ARG(
            tc == kHandle || tc == kNull || tc == kArrayHandle, "handle"); break;
      }
      case TVM_ARRAY_GET_DATA: {
        STACK_VM_TVM_ARRARY_GET(v_handle, void*, data); break;
      }
      case TVM_ARRAY_GET_SHAPE: {
        STACK_VM_TVM_ARRARY_GET(v_handle, void*, shape); break;
      }
      case TVM_ARRAY_GET_STRIDES: {
        STACK_VM_TVM_ARRARY_GET(v_handle, void*, strides); break;
      }
      case TVM_ARRAY_GET_NDIM: {
        STACK_VM_TVM_ARRARY_GET(v_int64, int64_t, ndim); break;
      }
      case TVM_ARRAY_GET_TYPE_CODE: {
        STACK_VM_TVM_ARRARY_GET(v_int64, int64_t, dtype.code); break;
      }
      case TVM_ARRAY_GET_TYPE_BITS: {
        STACK_VM_TVM_ARRARY_GET(v_int64, int64_t, dtype.bits); break;
      }
      case TVM_ARRAY_GET_TYPE_LANES: {
        STACK_VM_TVM_ARRARY_GET(v_int64, int64_t, dtype.lanes); break;
      }
    }
    CHECK_LT(sp, stack_cap) << "Stack overflow";
  }
}

}  // namespace jit
}  // namespace tvm
