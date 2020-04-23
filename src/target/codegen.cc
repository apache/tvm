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
 * \file codegen.cc
 * \brief Common utilities to generated C style code.
 */
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>

#include <tvm/ir/module.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/function.h>

#include <tvm/runtime/container.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/c_runtime_api.h>
#include <dmlc/memory_io.h>
#include <sstream>
#include <vector>
#include <cstdint>
#include <unordered_set>
#include <cstring>

namespace tvm {
namespace codegen {

runtime::Module Build(IRModule mod, const Target& target) {
  if (BuildConfig::Current()->disable_assert) {
    mod = tir::transform::SkipAssert()(mod);
  }
  std::string build_f_name = "target.build." + target->target_name;
  // the build function.
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  CHECK(bf != nullptr)
      << "target.build." << target << " is not enabled";
  return (*bf)(mod, target->str());
}

/*! \brief Helper class to serialize module */
class ModuleSerializer {
 public:
  explicit ModuleSerializer(runtime::Module mod) : mod_(mod) {
    Init();
  }

  void SerializeModule(dmlc::Stream* stream) {
    // Only have one DSO module and it is in the root, then
    // we will not produce import_tree_.
    bool has_import_tree = true;
    if (DSOExportable(mod_.operator->()) && mod_->imports().empty()) {
      has_import_tree = false;
    }
    uint64_t sz = 0;
    if (has_import_tree) {
      // we will append one key for _import_tree
      // The layout is the same as before: binary_size, key, logic, key, logic...
      sz = mod_vec_.size() + 1;
    } else {
      // Keep the old behaviour
      sz = mod_->imports().size();
    }
    stream->Write(sz);

    for (auto m : mod_vec_) {
      std::string mod_type_key = m->type_key();
      if (!DSOExportable(m)) {
        stream->Write(mod_type_key);
        m->SaveToBinary(stream);
      } else if (has_import_tree) {
        mod_type_key = "_lib";
        stream->Write(mod_type_key);
      }
    }

    // Write _import_tree key if we have
    if (has_import_tree) {
      std::string import_key = "_import_tree";
      stream->Write(import_key);
      stream->Write(import_tree_row_ptr_);
      stream->Write(import_tree_child_indices_);
    }
  }

 private:
  void Init() {
    CreateModuleIndex();
    CreateImportTree();
  }

  // invariance: root module is always at location 0.
  // The module order is collected via DFS
  void CreateModuleIndex() {
    std::unordered_set<const runtime::ModuleNode*> visited {mod_.operator->()};
    std::vector<runtime::ModuleNode*> stack {mod_.operator->()};
    uint64_t module_index = 0;

    while (!stack.empty()) {
      runtime::ModuleNode* n = stack.back();
      stack.pop_back();
      mod2index_[n] = module_index++;
      mod_vec_.emplace_back(n);
      for (runtime::Module m : n->imports()) {
        runtime::ModuleNode* next = m.operator->();
        if (visited.count(next) == 0) {
          visited.insert(next);
          stack.push_back(next);
        }
      }
    }
  }

  void CreateImportTree() {
    for (auto m : mod_vec_) {
      for (runtime::Module im : m->imports()) {
        uint64_t mod_index = mod2index_[im.operator->()];
        import_tree_child_indices_.push_back(mod_index);
      }
      import_tree_row_ptr_.push_back(import_tree_child_indices_.size());
    }
  }

  bool DSOExportable(const runtime::ModuleNode* mod) {
    return !std::strcmp(mod->type_key(), "llvm") ||
           !std::strcmp(mod->type_key(), "c");
  }

  runtime::Module mod_;
  // construct module to index
  std::unordered_map<runtime::ModuleNode*, size_t> mod2index_;
  // index -> module
  std::vector<runtime::ModuleNode*> mod_vec_;
  std::vector<uint64_t> import_tree_row_ptr_ {0};
  std::vector<uint64_t> import_tree_child_indices_;
};

namespace {
  std::string SerializeModule(const runtime::Module& mod) {
    std::string bin;
    dmlc::MemoryStringStream ms(&bin);
    dmlc::Stream* stream = &ms;

    ModuleSerializer module_serializer(mod);
    module_serializer.SerializeModule(stream);

    return bin;
  }
}  // namespace

std::string PackImportsToC(const runtime::Module& mod, bool system_lib) {
  std::string bin = SerializeModule(mod);

  // translate to C program
  std::ostringstream os;
  os << "#ifdef _WIN32\n"
     << "#define TVM_EXPORT __declspec(dllexport)\n"
     << "#else\n"
     << "#define TVM_EXPORT\n"
     << "#endif\n";
  os << "#ifdef __cplusplus\n"
     << "extern \"C\" {\n"
     << "#endif\n";
  os << "TVM_EXPORT extern const unsigned char " << runtime::symbol::tvm_dev_mblob << "[];\n";
  uint64_t nbytes = bin.length();
  os << "const unsigned char " << runtime::symbol::tvm_dev_mblob
     << "[" << bin.length() + sizeof(nbytes) << "] = {\n  ";
  os << std::hex;
  size_t nunit = 80 / 4;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    // sperators
    if (i != 0) {
      os << ",";
    }
    os << "0x" << ((nbytes >> (i * 8)) & 0xffUL);
  }
  for (size_t i = 0; i < bin.length(); ++i) {
    // sperators
    if ((i + sizeof(nbytes)) % nunit == 0) {
      os << ",\n  ";
    } else {
      os << ",";
    }
    int c = bin[i];
    os << "0x" << (c & 0xff);
  }
  os << "\n};\n";
  if (system_lib) {
    os << "extern int TVMBackendRegisterSystemLibSymbol(const char*, void*);\n";
    os << "static int " << runtime::symbol::tvm_dev_mblob << "_reg_ = "
       << "TVMBackendRegisterSystemLibSymbol(\"" << runtime::symbol::tvm_dev_mblob << "\", (void*)"
       << runtime::symbol::tvm_dev_mblob << ");\n";
  }
  os << "#ifdef __cplusplus\n"
     << "}\n"
     << "#endif\n";
  return os.str();
}

runtime::Module PackImportsToLLVM(const runtime::Module& mod,
                                  bool system_lib,
                                  const std::string& target_triple) {
  std::string bin = SerializeModule(mod);

  uint64_t nbytes = bin.length();
  std::string header;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    header.push_back(((nbytes >> (i * 8)) & 0xffUL));
  }
  std::string blob = header + bin;
  TVMByteArray blob_byte_array;
  blob_byte_array.size = blob.length();
  blob_byte_array.data = blob.data();

  // Call codegen_blob to generate LLVM module
  std::string codegen_f_name = "codegen.codegen_blob";
  // the codegen function.
  const PackedFunc* codegen_f = runtime::Registry::Get(codegen_f_name);
  CHECK(codegen_f != nullptr)  << "codegen.codegen_blob is not presented.";
  return (*codegen_f)(blob_byte_array, system_lib, target_triple);
}

TVM_REGISTER_GLOBAL("target.Build")
.set_body_typed(Build);

// Export two auxiliary function to the runtime namespace.
TVM_REGISTER_GLOBAL("runtime.ModulePackImportsToC")
.set_body_typed(PackImportsToC);

TVM_REGISTER_GLOBAL("runtime.ModulePackImportsToLLVM")
.set_body_typed(PackImportsToLLVM);

}  // namespace codegen
}  // namespace tvm
