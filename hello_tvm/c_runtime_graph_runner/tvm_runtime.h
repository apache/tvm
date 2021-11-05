// 2021-08-09 11:03
#ifndef TVM_RUNTIME_H
#define TVM_RUNTIME_H

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/graph_executor.h>
#include <tvm/runtime/crt/packed_func.h>
#include <tvm/runtime/crt/page_allocator.h>

TVMGraphExecutor* tvm_runtime_create();
#endif  // TVM_RUNTIME_H
