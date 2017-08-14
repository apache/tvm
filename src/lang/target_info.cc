/*!
 *  Copyright (c) 2017 by Contributors
 * \file target_info.cc
 */
#include <tvm/target_info.h>
#include <tvm/packed_func_ext.h>

namespace tvm {

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<MemoryInfoNode>([](const MemoryInfoNode *op, IRPrinter *p) {
    p->stream << "mem-info("
              << "unit_bits=" << op->unit_bits << ", "
              << "max_num_bits=" << op->max_num_bits << ", "
              << "max_simd_bits=" << op->max_simd_bits << ", "
              << "head_address=" << op->head_address << ")";
});

TVM_REGISTER_NODE_TYPE(MemoryInfoNode);

MemoryInfo GetMemoryInfo(const std::string& scope) {
  std::string fname = "tvm.info.mem." + scope;
  const runtime::PackedFunc* f = runtime::Registry::Get(fname);
  if (f == nullptr) {
    return MemoryInfo();
  } else {
    return (*f)();
  }
}

}  // namespace tvm
