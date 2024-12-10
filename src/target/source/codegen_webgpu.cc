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
#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../arith/pattern_match.h"
#include "../../runtime/meta_data.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

// WebGPU Info
struct WebGPUWorkGroupInfo {
  int workgroup_size[3] = {1, 1, 1};
  // whether we have ref to block index z is used.
  bool has_block_index_z{false};
  // set of handles that have write access
  std::unordered_set<Var> write_access_set;
};

class WebGPUWorkgroupInfoCollector : public StmtExprVisitor {
 public:
  static WebGPUWorkGroupInfo Collect(const Stmt& stmt) {
    WebGPUWorkgroupInfoCollector collector;
    collector(stmt);
    return collector.info_;
  }

 private:
  void VisitExpr_(const VarNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    Var buffer_var = GetRef<Var>(op);
    if (buffer_var.dtype().is_handle()) {
      info_.write_access_set.insert(buffer_var);
    }
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    info_.write_access_set.insert(op->buffer->data);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
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
          info_.workgroup_size[ts.dim_index] = static_cast<uint32_t>(sizeptr->value);
        } else if (ts.rank == 0) {
          if (ts.dim_index == 2) {
            info_.has_block_index_z = true;
          }
        }
      }
    }
    // normal operation
    StmtExprVisitor::VisitStmt_(op);
  }
  WebGPUWorkGroupInfo info_;
};

std::string CodeGenWebGPU::Finish() {
  // Using f16 requires enable directive
  if (enable_fp16_) {
    header_stream << "enable f16;\n\n";
  }
  return header_stream.str() + decl_stream.str() + this->fwd_decl_stream.str() + stream.str();
}

void CodeGenWebGPU::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  // analyze the data;
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

CodeGenWebGPU::CodeGenWebGPU(Target target) : target_(target) {}

runtime::FunctionInfo CodeGenWebGPU::AddFunction(const PrimFunc& f, bool skip_readonly_decl) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  name_supply_->ReserveName("var");
  name_supply_->ReserveName("let");
  name_supply_->ReserveName("const");
  name_supply_->ReserveName("std");

  // skip the first underscore, so SSA variable starts from
  name_supply_->FreshName("v_");
  // Setup the thread group info.
  ICHECK_EQ(name_supply_->FreshName("threadIdx"), "threadIdx");
  ICHECK_EQ(name_supply_->FreshName("blockIdx"), "blockIdx");
  ICHECK_EQ(name_supply_->FreshName("gridDim"), "gridDim");

  // add to alloc buffer type.
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenWebGPU: Expect PrimFunc to have the global_symbol attribute";

  header_stream << "//----------------------------------------\n"
                << "// Function: " << global_symbol.value() << "\n"
                << "//----------------------------------------\n";
  runtime::FunctionInfo func_info;
  func_info.name = global_symbol.value();

  WebGPUWorkGroupInfo info = WebGPUWorkgroupInfoCollector::Collect(f->body);

  std::vector<Var> pod_args;
  int num_buffer = 0;

  // add param_access modes info to launch params
  std::ostringstream os_param_access;
  os_param_access << "paramWriteAccess:[";
  // setup buffer argumemts
  for (Var arg : f->params) {
    DataType t = arg.dtype();
    func_info.arg_types.push_back(t);

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
      std::string access_mode;
      if (num_buffer != 0) {
        os_param_access << ",";
      }
      if (skip_readonly_decl || info.write_access_set.count(arg)) {
        access_mode = "read_write";
        os_param_access << "1";
      } else {
        access_mode = "read";
        os_param_access << "0";
      }
      // add extra access mode info to launch params
      this->decl_stream << "@group(0) @binding(" << num_buffer++ << ") "
                        << "var<storage, " << access_mode << "> " << vid << " : array<";
      this->PrintType(value_storage_type, this->decl_stream);
      this->decl_stream << ">;\n";
    } else {
      pod_args.push_back(arg);
    }
  }

  // Store all pod arguments in a single buffer of int32
  // do bitcast to change to other data types
  // always pass gridDimX in to get around of the 65535 gridDim
  // restrictions in some platforms
  std::string type_pod_args = name_supply_->FreshName("PODArgs");
  std::string val_pod_args = name_supply_->FreshName("podArgs");
  std::string packGridDimX = name_supply_->FreshName("packGridDimX");

  this->decl_stream << "\nstruct " << type_pod_args << " {\n";

  for (size_t i = 0; i < pod_args.size(); ++i) {
    Var v = pod_args[i];
    ICHECK(!v.dtype().is_handle());
    std::string vid = AllocVarID(v.get());

    if (v.dtype() == DataType::Int(32)) {
      this->decl_stream << "  " << vid << ": i32";
    } else if (v.dtype() == DataType::UInt(32)) {
      this->decl_stream << "  " << vid << ": u32";
    } else if (v.dtype() == DataType::Float(32)) {
      this->decl_stream << "  " << vid << ": f32";
    } else {
      LOG(FATAL) << "Do not support pod argument type " << v.dtype();
    }
    this->decl_stream << ",\n";
    // value ref
    std::ostringstream vref;
    vref << val_pod_args << "." << vid;
    var_idmap_[v.get()] = vref.str();
  }
  this->decl_stream << "  " << packGridDimX << ": u32\n}\n";

  this->decl_stream << "@group(0) @binding(" << num_buffer++ << ") "
                    << "var<uniform> " << val_pod_args << " : " << type_pod_args << ";\n\n";

  // setup thread tags and param access in launch param tags;
  if (auto opt = f->GetAttr<Array<String>>(tir::attr::kKernelLaunchParams)) {
    for (const auto& thread_tag : opt.value()) {
      func_info.launch_param_tags.push_back(thread_tag);
    }
  }
  os_param_access << "]";
  func_info.launch_param_tags.push_back(os_param_access.str());

  ICHECK(!info.has_block_index_z)
      << "blockIdx.z is not supported in WebGPU to accomodate large blockIdx.x";
  // anotate workgroup
  this->stream << "@compute @workgroup_size(" << info.workgroup_size[0] << ", "
               << info.workgroup_size[1] << ", " << info.workgroup_size[2] << ")\n";

  // add to alloc buffer type.
  // Function header.
  this->stream << "fn " << func_info.name << "(\n"
               << "  @builtin(workgroup_id) blockIdx : vec3<u32>,\n"
               << "  @builtin(num_workgroups) gridDim : vec3<u32>,\n"
               << "  @builtin(local_invocation_id) threadIdx : vec3<u32>\n"
               << ") {\n";
  // skip out of bound grids
  this->stream << "  if (blockIdx.z * gridDim.x + blockIdx.x > "  // NOLINT(*)
               << val_pod_args << "." << packGridDimX << ") { return; }\n";
  // the function scope.
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
  return func_info;
}

void CodeGenWebGPU::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  std::ostringstream os;
  PrintType(iv->var.dtype(), os);
  if (iv->thread_tag == "blockIdx.x") {
    // WebGPU have restriction to limit the maximum size of blockId.x to be 65535
    // We allow runtime to spread the load out to blockIdx.z so it can be a large number.
    os << "(blockIdx.z * gridDim.x + blockIdx.x)";
    std::string tidx = os.str();
    std::string aggregated_bidx = SSAGetID(os.str(), iv->var.dtype());
    var_idmap_[iv->var.get()] = aggregated_bidx;
  } else {
    os << "(" << iv->thread_tag << ")";
    std::string tidx = os.str();
    this->MarkConst(tidx);
    var_idmap_[iv->var.get()] = tidx;
  }
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
    // Currently WebGPU doesn't support `i8` and an `int8x4` is represented as a `u32`.
    if (t.is_int() && t.bits() == 8 && lanes == 4) {
      os << "u32";
      return;
    }
    os << "vec" << lanes << "<";
  }

  if (t.is_float()) {
    ICHECK(t.bits() == 16 || t.bits() == 32) << "CodeGenWebGPU: only support f16 or f32";
    if (t.bits() == 16) {
      // Using f16 requires enable directive
      enable_fp16_ = true;
    }
    os << "f" << t.bits();
  } else if (t.is_uint()) {
    ICHECK(t.bits() != 64) << "CodeGenWebGPU: do not support u64";
    os << "u" << t.bits();
  } else if (t.is_int()) {
    ICHECK(t.bits() != 64) << "CodeGenWebGPU: do not support i64";
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
  int lanes = op->dtype.lanes();
  PrintType(op->dtype, os);
  os << "(";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << ')';
}

PrimExpr CodeGenWebGPU::EnforceU32(PrimExpr value) {
  return cast(DataType::UInt(32, value.dtype().lanes()), value);
}

void CodeGenWebGPU::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(builtin::reinterpret())) {
    // generate bitcast<TYPE>(ARG)
    os << "bitcast<";
    this->PrintType(op->dtype, os);
    os << ">(";
    this->PrintExpr(op->args[0], os);
    os << ")";
  } else if (op->op.same_as(builtin::shift_right())) {
    os << '(';
    this->PrintExpr(op->args[0], os);
    os << ">>";
    // WebGPU requires shift bits to be u32.
    this->PrintExpr(EnforceU32(op->args[1]), os);
    os << ')';
  } else if (op->op.same_as(builtin::shift_left())) {
    os << '(';
    this->PrintExpr(op->args[0], os);
    os << "<<";
    // WebGPU requires shift bits to be u32.
    this->PrintExpr(EnforceU32(op->args[1]), os);
    os << ')';
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
      std::string true_val = PrintExpr(op->args[1]);
      this->PrintIndent();
      this->stream << result << " = " << true_val << ";\n} else {\n";
      this->EndScope(then_scope);
    }
    {
      int else_scope = this->BeginScope();
      std::string false_val = PrintExpr(op->args[2]);
      this->PrintIndent();
      this->stream << result << " = " << false_val << ";\n}\n";
      this->EndScope(else_scope);
    }
    os << result;
  } else if (op->op.same_as(builtin::dp4a())) {
    // generate `dot4I8Packed(vec1, vec2) + acc` for the builtin `dp4a`
    os << "dot4I8Packed(";
    this->PrintExpr(op->args[0], os);
    os << ", ";
    this->PrintExpr(op->args[1], os);
    os << ") + ";
    this->PrintExpr(op->args[2], os);
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

void CodeGenWebGPU::VisitExpr_(const LetNode* op, std::ostream& os) {  // NOLINT(*)
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
  os << PrintExpr(op->body);
  // Pop the defined var from var_idmap when exiting its scope.
  // We do this because it is hard to completely avoid a same LetNode appearing
  // at different places.
  bool removed = var_idmap_.erase(op->var.get());
  ICHECK(removed);
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
    // Using f16 requires enable directive
    enable_fp16_ = true;
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
  ICHECK(!op->predicate.defined()) << "Predicated buffer load is not supported.";

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
      os << "(";
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
  ICHECK(!op->predicate.defined()) << "Predicated buffer store is not supported.";

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
    // TODO(Charlie): These code would cause non-uniformity as it introduces variables in module
    // scope rather than function scope; but it was included for some unknown reasons; kept for now.
    // this->decl_stream << "var<private> " << vid << " : array<";
    // PrintType(op->dtype, this->decl_stream);
    // this->decl_stream << ", " << constant_size << ">;\n";
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
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  PrintIndent();
  stream << "for (var " << vid << " : ";
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

void CodeGenWebGPU::VisitStmt_(const WhileNode* op) {
  PrintIndent();
  stream << "while (true) {\n";
  int while_scope = BeginScope();
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  stream << "if (!(" << cond << ")) { break; }\n";
  PrintStmt(op->body);
  this->EndScope(while_scope);
  PrintIndent();
  stream << "}\n";
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

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    LOG(FATAL) << "WebGPUSourceModule is not directly runnable, export and run through tvmjs";
    return PackedFunc(nullptr);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmap_);
    stream->Write(smap_);
  }

  String GetSource(const String& format) final {
    if (format == "func_info") {
      std::ostringstream stream;
      dmlc::JSONWriter(&stream).Write(fmap_);
      return stream.str();
    } else {
      std::ostringstream os;
      for (auto kv : smap_) {
        os << kv.second;
      }
      return os.str();
    }
  }

 private:
  // function shader code table.
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
  bool skip_readonly_decl = false;
  std::unordered_map<std::string, std::string> smap;
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  // narrow all i64 to i32
  mod = tir::transform::ForceNarrowIndexToInt32()(std::move(mod));

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
    fmap[f_name] = cg.AddFunction(f, skip_readonly_decl);
    std::string code = cg.Finish();
    smap[f_name] = code;
  }

  auto n = make_object<WebGPUSourceModuleNode>(smap, fmap);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("target.build.webgpu").set_body_typed([](IRModule mod, Target target) {
  return BuildWebGPU(mod, target);
});

}  // namespace codegen
}  // namespace tvm
