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
#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace codegen {

ffi::Module Build(IRModule mod, Target target) {
  if (transform::PassContext::Current()
          ->GetConfig<Bool>("tir.disable_assert", Bool(false))
          .value()) {
    mod = tir::transform::SkipAssert()(mod);
  }

  // the build function.
  std::string build_f_name = "target.build." + target->kind->name;
  const auto bf = tvm::ffi::Function::GetGlobal(build_f_name);
  ICHECK(bf.has_value()) << build_f_name << " is not enabled";
  return (*bf)(mod, target).cast<ffi::Module>();
}

/*! \brief Helper class to serialize module */
class ModuleSerializer {
 public:
  explicit ModuleSerializer(ffi::Module mod) : mod_(mod) { Init(); }

  void SerializeModuleToBytes(dmlc::Stream* stream, bool export_dso) {
    // Always _import_tree
    stream->Write(import_tree_row_ptr_);
    stream->Write(import_tree_child_indices_);
    for (const auto& group : mod_group_vec_) {
      ICHECK_NE(group.size(), 0) << "Every allocated group must have at least one module";
      // we prioritize export dso when a module is both serializable and exportable
      if (export_dso) {
        if (group[0]->GetPropertyMask() & ffi::Module::kCompilationExportable) {
          std::string mod_type_key = "_lib";
          stream->Write(mod_type_key);
        } else if (group[0]->GetPropertyMask() & ffi::Module::kBinarySerializable) {
          ICHECK_EQ(group.size(), 1U) << "Non DSO module is never merged";
          std::string mod_type_key = group[0]->kind();
          stream->Write(mod_type_key);
          std::string bytes = group[0]->SaveToBytes();
          stream->Write(bytes);
        }
      } else {
        ICHECK(group[0]->GetPropertyMask() & ffi::Module::kBinarySerializable)
            << group[0]->kind() << " is not binary serializable.";
        ICHECK_EQ(group.size(), 1U) << "Non DSO module is never merged";
        std::string mod_type_key = group[0]->kind();
        stream->Write(mod_type_key);
        std::string bytes = group[0]->SaveToBytes();
        stream->Write(bytes);
      }
    }
  }

 private:
  void Init() {
    CreateModuleIndex();
    CreateImportTree();
  }

  // invariance: root module is always at location 0.
  // The module order is collected via DFS
  // This function merges all the DSO exportable module into
  // a single one as this is also what happens in the final hierachy
  void CreateModuleIndex() {
    std::unordered_set<const ffi::ModuleObj*> visited{mod_.operator->()};
    std::vector<ffi::ModuleObj*> stack{mod_.operator->()};
    uint64_t module_index = 0;

    auto fpush_imports_to_stack = [&](ffi::ModuleObj* node) {
      for (Any m : node->imports()) {
        ffi::ModuleObj* next = m.cast<ffi::Module>().operator->();
        if (visited.count(next) == 0) {
          visited.insert(next);
          stack.push_back(next);
        }
      }
    };

    std::vector<ffi::ModuleObj*> dso_exportable_boundary;

    // Create module index that merges all dso module into a single group.
    //
    // Do a two phase visit, to ensure dso module's index
    // is always bigger than a parent of any dso module
    // and smaller than children of any dso module.
    //
    // Error will be raised in CreateImportTree
    // if merging dso module causes a cycle in the import tree

    // Phase 0: only expand non-dso-module and record the boundary.
    while (!stack.empty()) {
      ffi::ModuleObj* n = stack.back();
      stack.pop_back();
      if (n->GetPropertyMask() & ffi::Module::kCompilationExportable) {
        // do not recursively expand dso modules
        // we will expand in phase 1
        dso_exportable_boundary.emplace_back(n);
      } else {
        // expand the non-dso modules
        mod2index_[n] = module_index++;
        mod_group_vec_.emplace_back(std::vector<ffi::ModuleObj*>({n}));
        fpush_imports_to_stack(n);
      }
    }
    if (dso_exportable_boundary.size() == 0) return;

    // create the slot for dso exportable modules
    // This index is chosen so that all the DSO's parents are
    // allocated before this index, and children will be allocated after
    uint64_t dso_module_index = module_index++;
    mod_group_vec_.emplace_back(std::vector<ffi::ModuleObj*>());

    // restart visiting the stack using elements in dso exportable boundary
    stack = std::move(dso_exportable_boundary);

    // Phase 1: expand the children of dso modules.
    while (!stack.empty()) {
      ffi::ModuleObj* n = stack.back();
      stack.pop_back();

      if (n->GetPropertyMask() & ffi::Module::kCompilationExportable) {
        mod_group_vec_[dso_module_index].emplace_back(n);
        mod2index_[n] = dso_module_index;
      } else {
        mod2index_[n] = module_index++;
        mod_group_vec_.emplace_back(std::vector<ffi::ModuleObj*>({n}));
      }
      fpush_imports_to_stack(n);
    }
  }

  void CreateImportTree() {
    std::vector<int64_t> child_indices;

    for (size_t parent_index = 0; parent_index < mod_group_vec_.size(); ++parent_index) {
      child_indices.clear();
      for (const auto* m : mod_group_vec_[parent_index]) {
        for (Any im : m->imports()) {
          uint64_t mod_index = mod2index_.at(im.cast<ffi::Module>().operator->());
          // skip cycle when dso modules are merged together
          if (mod_index != parent_index) {
            child_indices.emplace_back(mod_index);
          }
        }
      }
      // sort and unique the merged indices
      std::sort(child_indices.begin(), child_indices.end());
      auto unique_end = std::unique(child_indices.begin(), child_indices.end());

      // Check cycles due to merging dso exportable modules.
      if (child_indices.size() != 0) {
        // The index is supposed to follow the topological order.
        CHECK_LT(parent_index, child_indices[0])
            << "RuntimeError: Cannot export due to multiple dso-exportables "
            << "that cannot be merged without creating a cycle in the import tree. "
            << "Related module keys: parent=" << mod_group_vec_[parent_index][0]->kind()
            << ", child=" << mod_group_vec_[child_indices[0]][0]->kind();
      }
      // insert the child indices
      import_tree_child_indices_.insert(import_tree_child_indices_.end(), child_indices.begin(),
                                        unique_end);
      import_tree_row_ptr_.push_back(import_tree_child_indices_.size());
    }
  }

  ffi::Module mod_;
  // construct module to index
  std::unordered_map<ffi::ModuleObj*, size_t> mod2index_;
  // index -> module group
  std::vector<std::vector<ffi::ModuleObj*>> mod_group_vec_;
  std::vector<uint64_t> import_tree_row_ptr_{0};
  std::vector<uint64_t> import_tree_child_indices_;
};

std::string SerializeModuleToBytes(const ffi::Module& mod, bool export_dso) {
  std::string bin;
  dmlc::MemoryStringStream ms(&bin);
  dmlc::Stream* stream = &ms;

  ModuleSerializer module_serializer(mod);
  module_serializer.SerializeModuleToBytes(stream, export_dso);
  return bin;
}

ffi::Module DeserializeModuleFromBytes(std::string blob) {
  dmlc::MemoryStringStream ms(&blob);
  dmlc::Stream* stream = &ms;

  std::vector<ffi::Module> modules;
  std::vector<uint64_t> import_tree_row_ptr;
  std::vector<uint64_t> import_tree_child_indices;

  stream->Read(&import_tree_row_ptr);
  stream->Read(&import_tree_child_indices);

  uint64_t size = import_tree_row_ptr.size() - 1;
  for (uint64_t i = 0; i < size; ++i) {
    std::string tkey;
    ICHECK(stream->Read(&tkey));
    // "_lib" serves as a placeholder in the module import tree to indicate where
    // to place the DSOModule
    ICHECK(tkey != "_lib") << "Should not contain any placeholder for DSOModule.";
    if (tkey == "_import_tree") {
      ICHECK(stream->Read(&import_tree_row_ptr));
      ICHECK(stream->Read(&import_tree_child_indices));
    } else {
      std::string bytes;
      ICHECK(stream->Read(&bytes));
      auto loader = ffi::Function::GetGlobal("ffi.Module.load_from_bytes." + tkey);
      ICHECK(loader.has_value()) << "ffi.Module.load_from_bytes." << tkey << " is not enabled";
      auto m = (*loader)(ffi::Bytes(bytes)).cast<ffi::Module>();
      modules.emplace_back(m);
    }
  }

  for (size_t i = 0; i < modules.size(); ++i) {
    for (size_t j = import_tree_row_ptr[i]; j < import_tree_row_ptr[i + 1]; ++j) {
      auto child_index = import_tree_child_indices[j];
      ICHECK(child_index < modules.size());
      modules[i]->ImportModule(modules[child_index]);
    }
  }

  ICHECK(!modules.empty()) << "modules cannot be empty when import tree is present";
  // invariance: root module is always at location 0.
  // The module order is collected via DFS
  ffi::Module root_mod = modules[0];
  return root_mod;
}

std::string PackImportsToBytes(const ffi::Module& mod) {
  std::string bin = SerializeModuleToBytes(mod, /*export_dso*/ true);

  uint64_t nbytes = bin.length();
  std::string header;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    header.push_back(((nbytes >> (i * 8)) & 0xffUL));
  }
  return header + bin;
}

std::string PackImportsToC(const ffi::Module& mod, bool system_lib,
                           const std::string& c_symbol_prefix) {
  if (c_symbol_prefix.length() != 0) {
    CHECK(system_lib)
        << "c_symbol_prefix advanced option should be used in conjuction with system-lib";
  }

  std::string mdev_blob_name = c_symbol_prefix + ffi::symbol::tvm_ffi_library_bin;
  std::string blob = PackImportsToBytes(mod);

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
  os << "TVM_EXPORT extern const unsigned char " << mdev_blob_name << "[];\n";
  os << "const unsigned char " << mdev_blob_name << "[" << blob.length() << "] = {";
  os << std::hex;
  size_t nunit = 100 / 5;  // 100 columns, 5 chars per "0xab,"
  for (size_t i = 0; i < blob.length(); ++i) {
    if (i % nunit == 0) {
      os << "\n  ";
    }
    int c = blob[i];
    os << "0x" << std::setw(2) << std::setfill('0') << (c & 0xff) << ',';
  }
  os << "\n};\n";
  if (system_lib) {
    os << "extern int TVMFFIEnvModRegisterSystemLibSymbol(const char*, void*);\n";
    os << "static int " << mdev_blob_name << "_reg_ = "
       << "TVMFFIEnvModRegisterSystemLibSymbol(\"" << mdev_blob_name << "\", (void*)"
       << mdev_blob_name << ");\n";
  }
  os << "#ifdef __cplusplus\n"
     << "}\n"
     << "#endif\n";
  return os.str();
}

ffi::Module PackImportsToLLVM(const ffi::Module& mod, bool system_lib,
                              const std::string& llvm_target_string,
                              const std::string& c_symbol_prefix) {
  if (c_symbol_prefix.length() != 0) {
    CHECK(system_lib)
        << "c_symbol_prefix advanced option should be used in conjuction with system-lib";
  }

  std::string blob = PackImportsToBytes(mod);

  // Call codegen_blob to generate LLVM module
  std::string codegen_f_name = "codegen.codegen_blob";
  // the codegen function.
  const auto codegen_f = tvm::ffi::Function::GetGlobal(codegen_f_name);
  ICHECK(codegen_f.has_value()) << "codegen.codegen_blob is not presented.";
  return (*codegen_f)(ffi::Bytes(blob), system_lib, llvm_target_string, c_symbol_prefix)
      .cast<ffi::Module>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.Build", Build);
}

// Export a few auxiliary function to the runtime namespace.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.ModuleImportsBlobName",
           []() -> std::string { return ffi::symbol::tvm_ffi_library_bin; })
      .def("runtime.ModulePackImportsToTensor",
           [](const ffi::Module& mod) {
             std::string buffer = PackImportsToBytes(mod);
             ffi::Shape::index_type size = buffer.size();
             DLDataType uchar;
             uchar.code = kDLUInt;
             uchar.bits = 8;
             uchar.lanes = 1;
             DLDevice dev;
             dev.device_type = kDLCPU;
             dev.device_id = 0;
             auto array = runtime::Tensor::Empty({size}, uchar, dev);
             array.CopyFromBytes(buffer.data(), size);
             return array;
           })
      .def("runtime.ModulePackImportsToC", PackImportsToC)
      .def("runtime.ModulePackImportsToLLVM", PackImportsToLLVM);
}

}  // namespace codegen
}  // namespace tvm
