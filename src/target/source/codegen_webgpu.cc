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
 * \file codegen_webgpu.cc
 */
#include "codegen_webgpu.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../arith/pattern_match.h"
#include "../../runtime/meta_data.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

std::string CodeGenWebGPU::Finish() {
  return decl_stream.str() + this->fwd_decl_stream.str() + stream.str();
}

void CodeGenWebGPU::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  // analyze the data;
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
  std::fill(workgroup_size_, workgroup_size_ + 3, 1);
}

CodeGenWebGPU::CodeGenWebGPU(Target target) : target_(target) {}

void CodeGenWebGPU::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // skip the first underscore, so SSA variable starts from
  name_supply_->FreshName("v_");
  // Setup the thread group info.
  ICHECK_EQ(name_supply_->FreshName("threadIdx"), "threadIdx");
  ICHECK_EQ(name_supply_->FreshName("blockIdx"), "blockIdx");

  // add to alloc buffer type.
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenWebGPU: Expect PrimFunc to have the global_symbol attribute";

  decl_stream << "//----------------------------------------\n"
              << "// function: " << global_symbol.value() << "\n"
              << "//----------------------------------------\n";

  std::vector<Var> pod_args;
  int num_buffer = 0;
  // setup buffer argumemts
  for (Var arg : f->params) {
    DataType t = arg.dtype();
    if (t.is_handle()) {
      auto* ptr = arg->type_annotation.as<PointerTypeNode>();
      ICHECK(ptr) << "All handles passed to the CodeGenWebGPU must have a type_annotation as a "
                     "PointerType, "
                  << "and must point to a PrimType";
      auto* prim = ptr->element_type.as<PrimTypeNode>();
      ICHECK(prim) << "All handles passed to the CodeGenWebGPU must have a type_annotation as a "
                      "PointerType, "
                   << "and must point to a PrimType";
      DataType value_storage_type = prim->dtype;
      if (value_storage_type == DataType::Bool()) {
        // We need a physically addressable buffer type to support boolean tensors.
        // The loaded byte is cast to bool inside the LoadNode visitor below.
        value_storage_type = boolean_storage_type_.with_lanes(value_storage_type.lanes());
      }
      std::string vid = AllocVarID(arg.get());
      this->decl_stream << "@group(0) @binding(" << num_buffer++ << ") "
                        << "var<storage, read_write> " << vid << " : array<";
      this->PrintType(value_storage_type, this->decl_stream);
      this->decl_stream << ">;\n";
    } else {
      pod_args.push_back(arg);
    }
  }

  if (pod_args.size() != 0) {
    // setup POD arguments
    // TODO(tvm-team): store as a uniform, readonly buffer.
    LOG(FATAL) << "Do not support pod arguments for now";
  }
  // add to alloc buffer type.
  // Function header.
  this->stream << "fn main(\n"
               << "  @builtin(workgroup_id) blockIdx : vec3<u32>,\n"
               << "  @builtin(local_invocation_id) threadIdx : vec3<u32>\n"
               << ") {\n";
  // the function scope.
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
  // anotate workgroup
  this->fwd_decl_stream << "@compute @workgroup_size(" << workgroup_size_[0] << ", "
                        << workgroup_size_[1] << ", " << workgroup_size_[2] << ")\n";
}

void CodeGenWebGPU::VisitStmt_(const AttrStmtNode* op) {
  // record workgroup size
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
      if (ts.rank == 1) {
        ICHECK_GE(ts.dim_index, 0) << "vthread should have been optimized out by here";
        ICHECK_LT(ts.dim_index, 3);
        auto* sizeptr = op->value.as<tir::IntImmNode>();
        ICHECK(sizeptr) << "CodeGenWebGPU: only allows constant thread group size "
                        << " get " << op->value;
        workgroup_size_[ts.dim_index] = static_cast<uint32_t>(sizeptr->value);
      }
    }
  }
  // normal operation
  CodeGenC::VisitStmt_(op);
}

void CodeGenWebGPU::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  std::ostringstream os;
  PrintType(iv->var.dtype(), os);
  os << "(" << iv->thread_tag << ")";
  std::string tidx = os.str();
  this->MarkConst(tidx);
  var_idmap_[iv->var.get()] = tidx;
}

void CodeGenWebGPU::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    LOG(FATAL) << "Cannot print handle type in WebGPU";
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }

  if (lanes != 1) {
    ICHECK(lanes >= 2 && lanes <= 4) << "CodeGenWebGPU: only allows vector with lanes in {2, 3, 4}";
    os << "vec" << lanes << "<";
  }

  if (t.is_float()) {
    ICHECK(t.bits() == 16 || t.bits() == 32) << "CodeGenWebGPU: only support f16 or f32";
    os << "f" << t.bits();
  } else if (t.is_uint()) {
    os << "u" << t.bits();
  } else if (t.is_int()) {
    os << "i" << t.bits();
  } else {
    LOG(FATAL) << "CodeGenWebGPU: Cannot convert type " << t << " to WebGPU type";
  }
  if (lanes != 1) {
    os << ">";
  }
}

void CodeGenWebGPU::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "workgroupBarrier();\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "workgroupBarrier();\n";
  } else if (sync == "global") {
    LOG(FATAL) << "global barrier not supported";
  }
}

void CodeGenWebGPU::PrintSSAAssign(const std::string& target, const std::string& src,
                                   DataType type) {
  stream << "let " << target << " : ";
  PrintType(type, stream);
  stream << " = " << src << ";\n";
}

void CodeGenWebGPU::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  PrintType(op->dtype, os);
  os << "(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenWebGPU::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(builtin::reinterpret())) {
    // generate bitcast<TYPE>(ARG)
    os << "bitcast<";
    this->PrintType(op->dtype, os);
    os << ">(";
    this->PrintExpr(op->args[0], os);
    os << ")";
  } else if (op->op.same_as(builtin::if_then_else())) {
    // conditional that skips eval if cond evals to false
    std::string result = name_supply_->FreshName("condval");
    std::string cond = PrintExpr(op->args[0]);
    this->PrintIndent();
    this->stream << "var " << result << " : ";
    PrintType(op->dtype, this->stream);
    this->stream << ";\n";
    this->PrintIndent();
    this->stream << "if (" << cond << ") {\n";
    {
      int then_scope = this->BeginScope();
      this->PrintIndent();
      this->stream << result << " = " << PrintExpr(op->args[1]) << ";\n} else {\n";
      this->EndScope(then_scope);
    }
    {
      int else_scope = this->BeginScope();
      this->PrintIndent();
      this->stream << result << " = " << PrintExpr(op->args[2]) << ";\n}\n";
      this->EndScope(else_scope);
    }
    os << result;
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenWebGPU::VisitExpr_(const CastNode* op, std::ostream& os) {  // NOLINT(*)
  PrintType(op->dtype, os);
  os << "(" << PrintExpr(op->value) << ")";
}

void CodeGenWebGPU::VisitExpr_(const SelectNode* op, std::ostream& os) {  // NOLINT(*)
  os << "select(" << PrintExpr(op->false_value) << ", " << PrintExpr(op->true_value) << ", "
     << PrintExpr(op->condition) << ")";
}

void CodeGenWebGPU::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->dtype.bits() == 32) {
    std::ostringstream temp;
    if (op->dtype.is_int()) {
      temp << op->value << "i";
    } else {
      ICHECK(op->dtype.is_uint());
      temp << op->value << "u";
    }
    this->MarkConst(temp.str());
    os << temp.str();
  } else {
    this->PrintType(op->dtype, os);
    os << "(" << op->value << ")";
  }
}

void CodeGenWebGPU::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  std::ostringstream temp;
  temp << std::scientific << op->value;
  if (op->dtype.bits() == 32) {
    temp << 'f';
  } else if (op->dtype.bits() == 16) {
    temp << 'h';
  } else {
    LOG(FATAL) << "Unsupported floating point bits " << op->dtype.bits();
  }
  MarkConst(temp.str());
  os << temp.str();
}

void CodeGenWebGPU::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {  // NOLINT(*)
  // NOTE: direct impl of load/store for correctness
  // Each printing stmt must stand on their own after all preprocessing steps
  // to ensure correctness in the case of nested-expression
  // do not try to lift common printings from each case
  ICHECK_EQ(op->indices.size(), 1) << "Load from non-flat memory not supported.";

  DataType value_dtype = op->dtype;
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;
  DataType element_dtype = op->buffer->dtype;

  int lanes = op->dtype.lanes();
  std::string buffer_vid = GetVarID(buffer_var.get());

  if (value_dtype.lanes() == element_dtype.lanes()) {
    // Direct buffer loading
    // Special handle bool loading
    if (value_dtype == DataType::Bool()) {
      this->PrintType(value_dtype, os);
      os << "(";
    } else {
      ICHECK(value_dtype == element_dtype);
    }
    ICHECK_EQ(index.dtype().lanes(), 1);
    os << buffer_vid << "[" << this->PrintExpr(index) << "]";
    // Special handle bool loading
    if (value_dtype == DataType::Bool()) {
      os << ")";
    }
  } else {
    // Vector load from scalar buffer
    ICHECK_EQ(element_dtype.lanes(), 1) << "Can only vector load scalar array";
    ICHECK(value_dtype.element_of() == element_dtype)
        << "WebGPU vector loading requires base type to match";
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, op->dtype.lanes()).Match(index)) {
      // vec3<f32>(buf[base + 0], buf[base + 1], buf[base + 2]);
      std::string base_vid = SSAGetID(PrintExpr(base.Eval()), base.Eval().dtype());
      PrintType(element_dtype.with_lanes(value_dtype.lanes()), os);
      os << "(";
      for (int i = 0; i < lanes; ++i) {
        if (i != 0) os << ", ";
        os << buffer_vid << "[" << base_vid << " + " << i << "]";
      }
      os << ")";
    } else {
      // vec3<f32>(buf[index[0]], buf[index[1]], buf[index[2]]);
      std::string index_vid = SSAGetID(PrintExpr(index), index.dtype());
      PrintType(element_dtype.with_lanes(value_dtype.lanes()), os);
      for (int i = 0; i < lanes; ++i) {
        if (i != 0) os << ", ";
        os << buffer_vid << "[" << index_vid << "[" << i << "]]";
      }
      os << ")";
    }
  }
}

void CodeGenWebGPU::VisitStmt_(const LetStmtNode* op) {
  // use ssa form.
  if (print_ssa_form_) {
    std::string value = PrintExpr(op->value);
    ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    std::string value = PrintExpr(op->value);
    this->stream << "let " << AllocVarID(op->var.get()) << " : ";
    PrintType(op->var.dtype(), this->stream);
    this->stream << " = " << value << ";\n";
  }
  PrintStmt(op->body);
}

void CodeGenWebGPU::VisitStmt_(const BufferStoreNode* op) {
  CHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";
  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;

  std::string buffer_vid = GetVarID(buffer_var.get());

  if (value_dtype.lanes() == element_dtype.lanes()) {
    // must execute print expr first
    // so we won't have recursive append to stream
    std::string index_vid = PrintExpr(index);
    std::string value_vid = PrintExpr(op->value);
    // now print the assignment line.
    this->PrintIndent();
    stream << buffer_vid << "[" << index_vid << "] = ";
    // special explicit conversion of bool
    if (value_dtype == DataType::Bool()) {
      PrintType(element_dtype, stream);
      stream << "(";
    } else {
      ICHECK(value_dtype == element_dtype);
    }
    stream << value_vid;
    // Special handle bool store
    if (value_dtype == DataType::Bool()) {
      stream << ")";
    }
    stream << ";\n";
  } else {
    // Vector store into scalar buffer
    ICHECK_EQ(element_dtype.lanes(), 1) << "Can only vector load scalar array";
    ICHECK(value_dtype.element_of() == element_dtype)
        << "WebGPU vector stire requires base type to match";
    std::string value_vid = PrintExpr(op->value);
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, value_dtype.lanes()).Match(index)) {
      // buf[base + 0] = value[0]
      // buf[base + 1] = value[1]
      std::string base_vid = SSAGetID(PrintExpr(base.Eval()), base.Eval().dtype());
      for (int i = 0; i < value_dtype.lanes(); ++i) {
        this->PrintIndent();
        stream << buffer_vid << "[" << base_vid << " + " << i << "] = " << value_vid << "[" << i
               << "];\n";
      }
    } else {
      // buf[index[0]] = value[0]
      // buf[index[1]] = value[1]
      std::string index_vid = SSAGetID(PrintExpr(index), index.dtype());
      for (int i = 0; i < value_dtype.lanes(); ++i) {
        this->PrintIndent();
        stream << buffer_vid << "[" << index_vid << "[" << i << "]] = " << value_vid << "[" << i
               << "];\n";
      }
    }
  }
}

void CodeGenWebGPU::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";
  auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));

  if (storage_scope.rank == runtime::StorageRank::kShared) {
    this->decl_stream << "var<workgroup> " << vid << " : array<";
    PrintType(op->dtype, this->decl_stream);
    this->decl_stream << ", " << constant_size << ">;\n";
  } else if (storage_scope.rank == runtime::StorageRank::kLocal) {
    this->PrintIndent();
    this->stream << "var " << vid << " : array<";
    PrintType(op->dtype, this->stream);
    this->stream << ", " << constant_size << ">;\n";
  } else {
    LOG(FATAL) << "WebGPU: Do not support storage scope: " << storage_scope.to_string();
  }
  this->PrintStmt(op->body);
}

void CodeGenWebGPU::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  stream << "for (var ";
  stream << vid << " : ";
  PrintType(op->loop_var.dtype(), stream);
  stream << " = 0; " << vid << " < " << extent << "; " << vid << "++) {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenWebGPU::VisitStmt_(const AssertStmtNode* op) {
  // skip assert
  PrintStmt(op->body);
}

void CodeGenWebGPU::VisitStmt_(const AllocateConstNode* op) {
  LOG(FATAL) << "WebGPU: do not support alloc const";
}

//-------------------------------------------------
// WebGPUSourceModule to enable export
//-------------------------------------------------
class WebGPUSourceModuleNode final : public runtime::ModuleNode {
 public:
  explicit WebGPUSourceModuleNode(std::unordered_map<std::string, std::string> smap,
                                  std::unordered_map<std::string, runtime::FunctionInfo> fmap)
      : smap_(smap), fmap_(fmap) {}

  const char* type_key() const final { return "webgpu"; }
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return runtime::ModulePropertyMask::kBinarySerializable; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "WebGPUSourceModule is not directly runnable, export and run through tvmjs";
    return PackedFunc(nullptr);
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    LOG(FATAL) << "Not implemented";
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmap_);
    stream->Write(smap_);
  }

  std::string GetSource(const std::string& format) final {
    std::ostringstream os;
    for (auto kv : smap_) {
      os << kv.second;
    }
    return os.str();
  }

 private:
  // function information table.
  std::unordered_map<std::string, std::string> smap_;
  // function information table.
  std::unordered_map<std::string, runtime::FunctionInfo> fmap_;
};

//-------------------------------------------------
// Build logic.
//-------------------------------------------------
runtime::Module BuildWebGPU(IRModule mod, Target target) {
  mod = tir::transform::PointerValueTypeRewrite()(std::move(mod));
  bool output_ssa = false;

  std::unordered_map<std::string, std::string> smap;
  for (auto kv : mod->functions) {
    CodeGenWebGPU cg(target);
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenWebGPU: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenWebGPU: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.defined())
        << "CodeGenWebGPU: Expect PrimFunc to have the global_symbol attribute";
    std::string f_name = global_symbol.value();
    cg.Init(output_ssa);
    cg.AddFunction(f);
    std::string code = cg.Finish();
    smap[f_name] = code;
  }
  auto n = make_object<WebGPUSourceModuleNode>(smap, ExtractFuncInfo(mod));
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("target.build.webgpu").set_body_typed([](IRModule mod, Target target) {
  return BuildWebGPU(mod, target);
});

}  // namespace codegen
}  // namespace tvm
