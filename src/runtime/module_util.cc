/*!
 *  Copyright (c) 2017 by Contributors
 * \file module_util.cc
 * \brief Utilities for module.
 */
#include <dmlc/memory_io.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include "./module_util.h"

namespace tvm {
namespace runtime {

void ImportModuleBlob(const char* mblob, std::vector<Module>* mlist) {
  CHECK(mblob != nullptr);
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    uint64_t c = mblob[i];
    nbytes |=  (c & 0xffUL) << (i * 8);
  }
  dmlc::MemoryFixedSizeStream fs(
      const_cast<char*>(mblob + sizeof(nbytes)), static_cast<size_t>(nbytes));
  dmlc::Stream* stream = &fs;
  uint64_t size;
  CHECK(stream->Read(&size));
  for (uint64_t i = 0; i < size; ++i) {
    std::string tkey;
    CHECK(stream->Read(&tkey));
    std::string fkey = "module.loadbinary_" + tkey;
    const PackedFunc* f = Registry::Get(fkey);
    CHECK(f != nullptr)
        << "Loader of " << tkey << "("
        << fkey << ") is not presented.";
    Module m = (*f)(static_cast<void*>(stream));
    mlist->push_back(m);
  }
}

PackedFunc WrapPackedFunc(BackendPackedCFunc faddr,
                          const std::shared_ptr<ModuleNode>& sptr_to_self) {
  return PackedFunc([faddr, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
      int ret = (*faddr)(
          const_cast<TVMValue*>(args.values),
          const_cast<int*>(args.type_codes),
          args.num_args);
      CHECK_EQ(ret, 0) << TVMGetLastError();
    });
}

}  // namespace runtime
}  // namespace tvm
