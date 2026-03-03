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
 * \file codegen_tpc.cc
 * \brief TPC-C code generator for Habana Gaudi accelerators.
 *
 * Generates TPC-C kernel source code from TVM TIR.
 * Currently targets Gaudi2 with float32 support.
 */

#include "codegen_tpc.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/tir/stmt_functor.h>

#include <string>
#include <vector>

#include "../../tir/transforms/ir_utils.h"

namespace tvm {
namespace codegen {

using namespace tir;

CodeGenTPC::CodeGenTPC() {
  // TPC-C doesn't use restrict keyword
  restrict_keyword_ = "";
}

void CodeGenTPC::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  // Reset TPC-specific state
  tensor_buffers_.clear();
  index_space_emitted_ = false;
}

std::string CodeGenTPC::Finish() {
  // TPC-C kernel preamble: no special includes needed.
  // TPC-C has built-in types (float64, int5, tensor) and intrinsics.
  return CodeGenC::Finish();
}

// ---------------------------------------------------------------------------
// Function signature: void main(tensor input0, tensor input1, tensor output)
// ---------------------------------------------------------------------------

void CodeGenTPC::PrintFuncPrefix(std::ostream& os) {
  // TPC-C kernels don't need a prefix like OpenCL's "__kernel"
}

void CodeGenTPC::PrintExtraAttrs(const PrimFunc& f, std::ostream& os) {
  // No extra attributes for TPC kernels (unlike CUDA's __launch_bounds__)
}

void CodeGenTPC::PrintFunctionSignature(const ffi::String& function_name, const PrimFunc& func,
                                        std::ostream& os) {
  PrintFuncPrefix(os);

  // TPC kernel entry point is always "void main"
  os << "void main(";

  for (size_t i = 0; i < func->params.size(); ++i) {
    tir::Var v = func->params[i];
    LOG(INFO) << v;
    // auto type = GetType(v);
    // LOG(INFO) << type.is_scalar();
    // LOG(INFO) << "type: " << v->type_annotation.as<PointerTypeNode>()->element_type.as<PrimTypeNode>();
    if (i > 0) {
      os << ", ";
    }

    if (v.dtype().is_handle()) {
      // Handle-type parameters are TPC tensor descriptors
      os << "tensor " << AllocVarID(v.get());
      // Track this buffer as a TPC tensor for intrinsic-based access
      tensor_buffers_.insert(v.get());
    } else {
      // Scalar parameters (int, float, etc.) stay as-is
      PrintType(GetType(v), os);
      os << " " << AllocVarID(v.get());
    }
  }
  os << ")";

  // Register handle data types for buffer access resolution
  for (const auto& param : func->params) {
    if (auto* ptr = param->type_annotation.as<PointerTypeNode>()) {
      if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
        LOG(INFO) << prim->dtype;
        RegisterHandleType(param.get(), prim->dtype);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// PreFunctionBody: inject TPC index space initialization
// ---------------------------------------------------------------------------

void CodeGenTPC::PreFunctionBody(const PrimFunc& f) {
  // Emit TPC index space boilerplate at the start of every kernel
  PrintIndent();
  stream << "const int5 index_space_start = get_index_space_offset();\n";
  PrintIndent();
  stream << "const int5 index_space_end = get_index_space_size() + index_space_start;\n";
  // Shared 5D coordinate vector used by all tensor loads/stores
  PrintIndent();
  stream << "int5 " << coords_var_ << " = {0, 0, 0, 0, 0};\n";
  stream << "\n";
  index_space_emitted_ = true;
}

// ---------------------------------------------------------------------------
// Type printing: TPC-C has unique SIMD vector types
// ---------------------------------------------------------------------------

void CodeGenTPC::PrintType(DataType t, std::ostream& os) {
  LOG(INFO) <<"here";
  int lanes = t.lanes();

  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "TPC: do not support vector of handles";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  // TPC-C SIMD vector types for float32:
  //   64 lanes -> float64 (64 x float32, the native SIMD width)
  // For int32:
  //   64 lanes -> int64 (64 x int32)
  if (t.is_float()) {
    switch (t.bits()) {
      case 32:
        if (lanes == 1) {
          os << "float";
        } else if (lanes == 64) {
          // TPC native SIMD vector: 64 x float32
          os << "float64";
        } else {
          LOG(FATAL) << "TPC: unsupported float32 vector width " << lanes
                     << " (only scalar and 64-lane supported)";
        }
        return;
      default:
        LOG(FATAL) << "TPC: unsupported float bit width " << t.bits()
                   << " (only float32 supported currently)";
    }
  }

  if (t.is_int()) {
    switch (t.bits()) {
      case 32:
        if (lanes == 1) {
          os << "int";
        } else if (lanes == 64) {
          os << "int64";
        } else {
          LOG(FATAL) << "TPC: unsupported int32 vector width " << lanes;
        }
        return;
      case 16:
        if (lanes == 1) {
          os << "short";
        } else {
          LOG(FATAL) << "TPC: unsupported int16 vector width " << lanes;
        }
        return;
      case 8:
        if (lanes == 1) {
          os << "char";
        } else {
          LOG(FATAL) << "TPC: unsupported int8 vector width " << lanes;
        }
        return;
      case 1:
        os << "bool";
        return;
      default:
        LOG(FATAL) << "TPC: unsupported int bit width " << t.bits();
    }
  }

  if (t.is_uint()) {
    switch (t.bits()) {
      case 32:
        if (lanes == 1) {
          os << "unsigned int";
        } else if (lanes == 64) {
          os << "uint64";
        } else {
          LOG(FATAL) << "TPC: unsupported uint32 vector width " << lanes;
        }
        return;
      case 16:
        os << "unsigned short";
        return;
      case 8:
        os << "unsigned char";
        return;
      case 1:
        os << "bool";
        return;
      default:
        LOG(FATAL) << "TPC: unsupported uint bit width " << t.bits();
    }
  }

  LOG(FATAL) << "TPC: unknown type " << t;
}

// ---------------------------------------------------------------------------
// AttrStmt: intercept thread_extent to generate TPC index space range loops
// ---------------------------------------------------------------------------

void CodeGenTPC::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    std::string tag = iv->thread_tag;

    // Map thread tag to TPC index space dimension
    int dim = -1;
    if (tag == "threadIdx.x" || tag == "tpc.index_space.0") {
      dim = 0;
    } else if (tag == "threadIdx.y" || tag == "tpc.index_space.1") {
      dim = 1;
    } else if (tag == "threadIdx.z" || tag == "tpc.index_space.2") {
      dim = 2;
    } else if (tag == "blockIdx.x" || tag == "tpc.index_space.3") {
      dim = 3;
    } else if (tag == "blockIdx.y" || tag == "tpc.index_space.4") {
      dim = 4;
    }

    if (dim >= 0) {
      // dim 0 (depth) is the SIMD dimension: 1 index space unit = 64 f32 elements.
      // Other dims: 1 unit = 1 element (step=1).
      // The step can be overridden via a "tpc.step" annotation on the IterVar if needed.
      int step = (dim == 0) ? 64 : 1;

      // Allocate C variable name for this loop variable
      std::string vid;
      if (!var_idmap_.count(iv->var.get())) {
        vid = AllocVarID(iv->var.get());
      } else {
        vid = GetVarID(iv->var.get());
      }

      static const char* dim_names[] = {"depth", "width", "height", "batch", "fifthDim"};
      const char* dname = dim_names[dim];

      // Emit: const int dimStart = index_space_start[dim] * step;
      PrintIndent();
      stream << "const int " << dname << "Start = index_space_start[" << dim << "]";
      if (step != 1) stream << " * " << step;
      stream << ";\n";

      // Emit: const int dimEnd = index_space_end[dim] * step;
      PrintIndent();
      stream << "const int " << dname << "End = index_space_end[" << dim << "]";
      if (step != 1) stream << " * " << step;
      stream << ";\n";

      // Emit the range for loop
      PrintIndent();
      stream << "for (int " << vid << " = " << dname << "Start; "
             << vid << " < " << dname << "End; "
             << vid << " += " << step << ") {\n";

      int for_scope = BeginScope();
      // Update the shared coordinate vector for this dimension
      PrintIndent();
      stream << coords_var_ << "[" << dim << "] = " << vid << ";\n";
      PrintStmt(op->body);
      EndScope(for_scope);

      PrintIndent();
      stream << "}\n";
      return;
    }
  }

  // For all other AttrStmts, fall back to base class
  CodeGenC::VisitStmt_(op);
}

// ---------------------------------------------------------------------------
// Buffer access: TPC tensor intrinsics
// ---------------------------------------------------------------------------

void CodeGenTPC::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {
  ICHECK_EQ(op->indices.size(), 1) << "TPC: load from non-flat memory not supported";

  DataType value_dtype = op->dtype;
  Var buffer_var = op->buffer->data;

  // Check if this buffer is a TPC tensor (function parameter with handle type)
  if (tensor_buffers_.count(buffer_var.get()) && value_dtype.lanes() == 64 &&
      value_dtype.is_float() && value_dtype.bits() == 32) {
    // Vector load from TPC tensor: v_f32_ld_tnsr_b(coords, tensor_name)
    // Use the shared coords variable that is updated by the enclosing index space loops.
    std::string vid = GetVarID(buffer_var.get());
    os << "v_f32_ld_tnsr_b(" << coords_var_ << ", " << vid << ")";
    return;
  }

  // Fall back to base class for scalar access or local buffers
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTPC::VisitStmt_(const BufferStoreNode* op) {
  ICHECK_EQ(op->indices.size(), 1) << "TPC: store to non-flat memory not supported";

  DataType value_dtype = op->value.dtype();
  Var buffer_var = op->buffer->data;

  // Check if this buffer is a TPC tensor for vector store
  if (tensor_buffers_.count(buffer_var.get()) && value_dtype.lanes() == 64 &&
      value_dtype.is_float() && value_dtype.bits() == 32) {
    // Vector store to TPC tensor: v_f32_st_tnsr(coords, tensor_name, value)
    // Use the shared coords variable updated by the enclosing index space loops.
    std::string value = PrintExpr(op->value);
    std::string vid = GetVarID(buffer_var.get());

    PrintIndent();
    stream << "v_f32_st_tnsr(" << coords_var_ << ", " << vid << ", " << value << ");\n";
    return;
  }

  // Fall back to base class for scalar access or local buffers
  CodeGenC::VisitStmt_(op);
}

// ---------------------------------------------------------------------------
// Storage scope: TPC has no shared memory concept
// ---------------------------------------------------------------------------

void CodeGenTPC::PrintStorageScope(const std::string& scope, std::ostream& os) {
  // TPC has no storage scope qualifiers like CUDA's __shared__ or OpenCL's __local
  // All local variables are in register file or local memory automatically
}

// ---------------------------------------------------------------------------
// Build function registration
// ---------------------------------------------------------------------------

void TPCCodegen(ffi::PackedArgs args, ffi::Any* rv) {
  codegen::CodeGenTPC cg;
  cg.Init(false);

  IRModule mod = args[0].cast<IRModule>();

  ffi::Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenTPC: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    functions.Set(gvar, prim_func);
  }

  // Two-pass: declare all functions first, then define them
  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  *rv = cg.Finish();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tpcCodegen", TPCCodegen);
}

}  // namespace codegen
}  // namespace tvm
