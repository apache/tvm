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
 *  Copyright (c) 2017 by Contributors
 * Implementation stack VM.
 * \file stackvm.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/c_backend_api.h>
#include <algorithm>
#include "stackvm.h"

namespace tvm {
namespace runtime {

typedef dmlc::ThreadLocalStore<StackVM::State> StackVMStateStore;

StackVM::State* StackVM::ThreadLocalState() {
  return StackVMStateStore::Get();
}

#define STACK_VM_BINOP(OP, FIELD)                                 \
  {                                                               \
    stack[sp - 1].FIELD = stack[sp - 1].FIELD OP stack[sp].FIELD; \
    sp -= 1; pc += 1;                                             \
  }

#define STACK_VM_CMPOP(OP, FIELD)                                   \
  {                                                                 \
    stack[sp - 1].v_int64 = stack[sp - 1].FIELD OP stack[sp].FIELD; \
    sp -= 1; pc += 1;                                               \
  }

#define STACK_VM_LOAD(FIELD, DST_TYPE, SRC_TYPE)                        \
  {                                                                     \
    int index = code[pc + 1].v_int;                                     \
    stack[sp]FIELD = static_cast<DST_TYPE>(                             \
        static_cast<SRC_TYPE*>(stack[sp].v_handle)[index]);             \
    pc += 2;                                                            \
  }

#define STACK_VM_STORE(FIELD, DST_TYPE)                                 \
  {                                                                     \
    int index = code[pc + 1].v_int;                                     \
    static_cast<DST_TYPE*>(stack[sp - 1].v_handle)[index] =             \
        static_cast<DST_TYPE>(stack[sp]FIELD);                          \
    sp -= 2; pc += 2;                                                   \
  }

#define STACK_VM_PRINT_CODE0(CODE)                            \
  case CODE:  {                                                     \
    os << "[" << pc << "]\t" << #CODE << std::endl; return pc + 1;  \
  }

#define STACK_VM_PRINT_CODE1(CODE)                                      \
  case CODE:  {                                                         \
    os << "[" << pc << "]\t" << #CODE << " " << code[pc + 1].v_int << "\n" \
       <<  "[" << pc + 1 << "]" << std::endl;                           \
        return pc + 2;                                                  \
  }

#define STACK_VM_PRINT_CODE2(CODE)                                      \
  case CODE:  {                                                         \
    os << "[" << pc << "]\t" << #CODE                                   \
        << " " << code[pc + 1].v_int                                    \
        << " " << code[pc + 2].v_int << "\n"                            \
       <<  "[" << pc + 1 << "]" << std::endl                            \
       <<  "[" << pc + 2 << "]" << std::endl;                           \
        return pc + 3;                                                  \
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
    // handle.
    STACK_VM_PRINT_CODE0(EQ_HANDLE);
    // addressing load
    STACK_VM_PRINT_CODE1(ARRAY_LOAD_UINT32);
    STACK_VM_PRINT_CODE1(ARRAY_LOAD_INT32);
    STACK_VM_PRINT_CODE1(ARRAY_LOAD_INT64);
    STACK_VM_PRINT_CODE1(ARRAY_LOAD_FP64);
    STACK_VM_PRINT_CODE1(ARRAY_LOAD_HANDLE);
    STACK_VM_PRINT_CODE1(ARRAY_LOAD_TVMVALUE);
    STACK_VM_PRINT_CODE1(ARRAY_STORE_UINT32);
    STACK_VM_PRINT_CODE1(ARRAY_STORE_INT32);
    STACK_VM_PRINT_CODE1(ARRAY_STORE_INT64);
    STACK_VM_PRINT_CODE1(ARRAY_STORE_FP64);
    STACK_VM_PRINT_CODE1(ARRAY_STORE_HANDLE);
    STACK_VM_PRINT_CODE1(ARRAY_STORE_TVMVALUE);
    STACK_VM_PRINT_CODE0(NOT);
    STACK_VM_PRINT_CODE0(ADDR_ADD);
    // stack ops
    STACK_VM_PRINT_CODE1(PUSH_I64);
    STACK_VM_PRINT_CODE1(PUSH_VALUE);
    STACK_VM_PRINT_CODE0(POP);
    STACK_VM_PRINT_CODE0(SELECT);
    STACK_VM_PRINT_HEAP_ACCESS(STORE_HEAP);
    STACK_VM_PRINT_HEAP_ACCESS(LOAD_HEAP);
    STACK_VM_PRINT_CODE1(ASSERT);
    STACK_VM_PRINT_JUMP(RJUMP_IF_TRUE);
    STACK_VM_PRINT_JUMP(RJUMP_IF_FALSE);
    STACK_VM_PRINT_JUMP(RJUMP);
    STACK_VM_PRINT_CODE1(ASSERT_SP);
    // Intrinsics
    STACK_VM_PRINT_CODE2(TVM_STRUCT_GET);
    STACK_VM_PRINT_CODE2(TVM_STRUCT_SET);
    // Allocate data by 8 bytes.
    STACK_VM_PRINT_CODE1(TVM_STACK_ALLOCA_BY_8BYTE);
    STACK_VM_PRINT_CODE0(TVM_DEVICE_ALLOCA);
    STACK_VM_PRINT_CODE0(TVM_DEVICE_FREE);
    STACK_VM_PRINT_CODE0(TVM_THROW_LAST_ERROR);
    // packed function.
    case CALL_PACKED_LOWERED: {
      int call_fid = code[pc + 1].v_int;
      int begin = code[pc + 2].v_int;
      int end = code[pc + 3].v_int;
      os << "[" << pc << "]\tCALL_PACKED_FUNC "
         << " fid=" << call_fid
         << " begin=" << begin
         << " end=" << end;
      os << '\n';
      for (int i = 0; i < 3; ++i) {
        os << "[" << pc + 1 + i << "]" << std::endl;
      }
      return pc + 4;
    }
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

void StackVM::Run(const runtime::TVMArgs& args,
                  runtime::ModuleNode* mod_ctx) const {
  StackVM::State* s = StackVM::ThreadLocalState();
  if (s->heap.size() < heap_size) {
    s->heap.resize(heap_size);
  }
  s->sp = 0;
  s->pc = 0;
  s->mod_ctx = mod_ctx;
  s->heap[0].v_handle = (void*)args.values;  // NOLINT(*)
  s->heap[1].v_handle = (void*)args.type_codes;  // NOLINT(*)
  s->heap[2].v_int64 = args.num_args;
  this->Run(s);
}

void StackVM::InitCache() {
  extern_func_cache_.clear();
  extern_func_cache_.resize(
      extern_func_name.size(), PackedFunc(nullptr));
}

void StackVM::Save(dmlc::Stream* strm) const {
  // to be endian invariant.
  std::vector<int32_t> code_copy(code.size());
  std::transform(code.begin(), code.end(), code_copy.begin(), [](Code c) {
      return c.v_int;
    });
  strm->Write(code_copy);
  strm->Write(str_data);
  strm->Write(extern_func_name);
  strm->Write(heap_id_name);
  strm->Write(heap_size);
  strm->Write(stack_size);
}

bool StackVM::Load(dmlc::Stream* strm)  {
  // to be endian invariant.
  std::vector<int32_t> code_copy;
  if (!strm->Read(&code_copy)) return false;
  code.resize(code_copy.size());
  std::transform(code_copy.begin(), code_copy.end(), code.begin(), [](int v) {
      Code code; code.v_int = v; return code;
    });
  if (!strm->Read(&str_data)) return false;
  if (!strm->Read(&extern_func_name)) return false;
  if (!strm->Read(&heap_id_name)) return false;
  if (!strm->Read(&heap_size)) return false;
  if (!strm->Read(&stack_size)) return false;
  this->InitCache();
  return true;
}

void StackVM::Run(State* s) const {
  int64_t sp = s->sp;
  int64_t pc = s->pc;
  int64_t alloca_sp = s->sp;
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
      case EQ_HANDLE: STACK_VM_CMPOP(==, v_handle); break;
      // addressing
      case ARRAY_LOAD_UINT32: STACK_VM_LOAD(.v_int64, int64_t, uint32_t); break;
      case ARRAY_LOAD_INT32: STACK_VM_LOAD(.v_int64, int64_t, int32_t); break;
      case ARRAY_LOAD_INT64: STACK_VM_LOAD(.v_int64, int64_t, int64_t); break;
      case ARRAY_LOAD_FP64: STACK_VM_LOAD(.v_float64, double, double); break;
      case ARRAY_LOAD_HANDLE: STACK_VM_LOAD(.v_handle, void*, void*); break;
      case ARRAY_LOAD_TVMVALUE: STACK_VM_LOAD(, TVMValue, TVMValue); break;
      // store
      case ARRAY_STORE_UINT32: STACK_VM_STORE(.v_int64, uint32_t); break;
      case ARRAY_STORE_INT32: STACK_VM_STORE(.v_int64, int32_t); break;
      case ARRAY_STORE_INT64: STACK_VM_STORE(.v_int64, int64_t); break;
      case ARRAY_STORE_FP64: STACK_VM_STORE(.v_float64, double); break;
      case ARRAY_STORE_HANDLE: STACK_VM_STORE(.v_handle, void*); break;
      case ARRAY_STORE_TVMVALUE: STACK_VM_STORE(, TVMValue); break;
      // add
      case ADDR_ADD: {
        stack[sp - 1].v_handle = (char*)(stack[sp - 1].v_handle) + stack[sp].v_int64;  // NOLINT(*)
        sp = sp - 1;
        pc = pc + 1;
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
      case CALL_PACKED_LOWERED: {
        // call packed function.
        TVMValue* value_stack = static_cast<TVMValue*>(stack[sp - 1].v_handle);
        int* type_stack = static_cast<int*>(stack[sp].v_handle);
        int call_fid = code[pc + 1].v_int;
        int begin = code[pc + 2].v_int;
        int end = code[pc + 3].v_int;
        int num_args = end - begin;
        static_assert(sizeof(Code) == sizeof(int) &&
                      alignof(Code) == alignof(int), "asusmption");
        runtime::TVMRetValue rv;
        GetExtern(s, call_fid).CallPacked(
            runtime::TVMArgs(value_stack + begin, type_stack + begin, num_args), &rv);
        sp = sp - 1;
        stack[sp] = rv.value();
        pc += 4;
        break;
      }
      // intrinsics
      case TVM_STRUCT_GET: {
        using namespace ir;
        int index = code[pc + 1].v_int;
        int kind = code[pc + 2].v_int;
        TVMArray* arr = static_cast<TVMArray*>(stack[sp].v_handle);
        switch (kind) {
          case intrinsic::kArrData: {
            stack[sp].v_handle = arr[index].data; break;
          }
          case intrinsic::kArrShape: {
            stack[sp].v_handle = arr[index].shape; break;
          }
          case intrinsic::kArrStrides: {
            stack[sp].v_handle = arr[index].strides; break;
          }
          case intrinsic::kArrNDim: {
            stack[sp].v_int64 = arr[index].ndim; break;
          }
          case intrinsic::kArrTypeCode: {
            stack[sp].v_int64 = static_cast<int64_t>(
                arr[index].dtype.code); break;
          }
          case intrinsic::kArrTypeBits: {
            stack[sp].v_int64 = static_cast<int64_t>(
                arr[index].dtype.bits); break;
          }
          case intrinsic::kArrTypeLanes: {
            stack[sp].v_int64 = static_cast<int64_t>(
                arr[index].dtype.lanes); break;
          }
          case intrinsic::kArrByteOffset: {
            stack[sp].v_int64 = static_cast<int64_t>(
                arr[index].byte_offset); break;
          }
          case intrinsic::kArrDeviceId: {
            stack[sp].v_int64 = arr[index].ctx.device_id; break;
          }
          case intrinsic::kArrDeviceType: {
            stack[sp].v_int64 = static_cast<int64_t>(
                arr[index].ctx.device_type); break;
          }
          case intrinsic::kArrAddr: {
            stack[sp].v_handle = arr + index; break;
          }
          case intrinsic::kTVMValueContent: {
            stack[sp] = static_cast<TVMValue*>(stack[sp].v_handle)[index]; break;
          }
          default: LOG(FATAL) << "unhandled get " << kind;
        }
        pc = pc + 3;
        break;
      }
      case TVM_STRUCT_SET: {
        using namespace ir;
        int index = code[pc + 1].v_int;
        int kind = code[pc + 2].v_int;
        TVMArray* arr = static_cast<TVMArray*>(stack[sp - 1].v_handle);
        switch (kind) {
          case intrinsic::kArrData: {
            arr[index].data = stack[sp].v_handle; break;
          }
          case intrinsic::kArrShape: {
            arr[index].shape = static_cast<int64_t*>(stack[sp].v_handle);
            break;
          }
          case intrinsic::kArrStrides: {
            arr[index].strides = static_cast<int64_t*>(stack[sp].v_handle);
            break;
          }
          case intrinsic::kArrNDim: {
            arr[index].ndim = static_cast<int>(stack[sp].v_int64);
            break;
          }
          case intrinsic::kArrTypeCode: {
            arr[index].dtype.code = static_cast<uint8_t>(stack[sp].v_int64);
            break;
          }
          case intrinsic::kArrTypeBits: {
            arr[index].dtype.bits = static_cast<uint8_t>(stack[sp].v_int64);
            break;
          }
          case intrinsic::kArrTypeLanes: {
            arr[index].dtype.lanes = static_cast<uint16_t>(stack[sp].v_int64);
            break;
          }
          case intrinsic::kArrByteOffset: {
            arr[index].byte_offset = static_cast<uint64_t>(stack[sp].v_int64);
            break;
          }
          case intrinsic::kArrDeviceId: {
            arr[index].ctx.device_id = static_cast<int>(stack[sp].v_int64);
            break;
          }
          case intrinsic::kArrDeviceType: {
            arr[index].ctx.device_type = static_cast<DLDeviceType>(stack[sp].v_int64);
            break;
          }
          case intrinsic::kTVMValueContent: {
            static_cast<TVMValue*>(stack[sp - 1].v_handle)[index] = stack[sp]; break;
          }
          default: LOG(FATAL) << "unhandled tvm_struct_set " << kind;
        }
        sp -= 2;
        pc += 3;
        break;
      }
      // alloca
      case TVM_STACK_ALLOCA_BY_8BYTE: {
        static_assert(sizeof(TVMValue) == 8, "invariance");
        int num = code[pc + 1].v_int;
        void* addr = &stack[sp] + 1;
        sp = sp + num + 1;
        alloca_sp = sp - 1;
        stack[sp].v_handle = addr;
        pc = pc + 2;
        break;
      }
      case TVM_DEVICE_ALLOCA: {
        int device_type = static_cast<int>(stack[sp - 4].v_int64);
        int device_id = static_cast<int>(stack[sp - 3].v_int64);
        size_t nbytes = static_cast<size_t>(stack[sp - 2].v_int64);
        int dtype_code_hint = static_cast<int>(stack[sp - 1].v_int64);
        int dtype_bits_hint = static_cast<int>(stack[sp].v_int64);
        void* ptr = TVMBackendAllocWorkspace(device_type, device_id, nbytes,
                                             dtype_code_hint, dtype_bits_hint);
        stack[sp - 4].v_handle = ptr;
        sp = sp - 4;
        pc = pc + 1;
        break;
      }
      case TVM_DEVICE_FREE: {
        int device_type = static_cast<int>(stack[sp - 2].v_int64);
        int device_id = static_cast<int>(stack[sp - 1].v_int64);
        void* ptr = stack[sp].v_handle;
        int ret = TVMBackendFreeWorkspace(device_type, device_id, ptr);
        stack[sp - 2].v_int64 = ret;
        sp = sp - 2;
        pc = pc + 1;
        break;
      }
      case TVM_THROW_LAST_ERROR: {
        LOG(FATAL) << TVMGetLastError();
        break;
      }
    }
    CHECK_GE(sp, alloca_sp) << "touch allocated space";
    CHECK_LT(sp, stack_cap) << "Stack overflow";
  }
}

const PackedFunc& StackVM::GetExtern(State* s, int fid) const {
  CHECK_LT(static_cast<size_t>(fid), extern_func_cache_.size());
  // allow race write in this, since write is idempotent
  PackedFunc& f = extern_func_cache_[fid];
  if (f == nullptr) {
    CHECK(s->mod_ctx != nullptr)
        << "No local context is set in stackvm";
    const PackedFunc* pf = s->mod_ctx->GetFuncFromEnv(extern_func_name[fid]);
    CHECK(pf != nullptr);
    f = *pf;
  }
  return f;
}

}  // namespace runtime
}  // namespace tvm
