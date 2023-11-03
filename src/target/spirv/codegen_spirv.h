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
 * \file ir_builder.h
 * \brief Utility for building SPIRV code block
 */
#ifndef TVM_TARGET_SPIRV_CODEGEN_SPIRV_H_
#define TVM_TARGET_SPIRV_CODEGEN_SPIRV_H_

#include <tvm/arith/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../runtime/spirv/spirv_shader.h"
#include "../../runtime/thread_storage_scope.h"
#include "ir_builder.h"
#include "spirv_support.h"

namespace tvm {
namespace codegen {

using namespace tir;

/*!
 * \brief Code generator into SPIRV
 */
class CodeGenSPIRV : public ExprFunctor<spirv::Value(const PrimExpr&)>,
                     public StmtFunctor<void(const Stmt&)> {
 public:
  /*!
   * \brief Initialize the codegen based on a specific target.
   *
   * \param target The target for which code should be generated.  The
   * device_type for this target must be kDLVulkan.
   */
  CodeGenSPIRV(Target target);

  /*!
   * \brief Compile and add function f to the current module.
   * \param f The function to be added.
   * \param name The name of the target function.
   * \return The final spirv module.
   */
  virtual runtime::SPIRVShader BuildFunction(const PrimFunc& f, const std::string& name);
  /*!
   * \brief Create Value for expression e
   * \param e The expression to be created value for.
   * \return created value.
   */
  spirv::Value MakeValue(const PrimExpr& e) { return VisitExpr(e); }
  // override codegen
  spirv::Value VisitExpr_(const VarNode* op) override;
  spirv::Value VisitExpr_(const CastNode* op) override;
  spirv::Value VisitExpr_(const IntImmNode* op) override;
  spirv::Value VisitExpr_(const FloatImmNode* op) override;
  spirv::Value VisitExpr_(const StringImmNode* op) override;
  spirv::Value VisitExpr_(const AddNode* op) override;
  spirv::Value VisitExpr_(const SubNode* op) override;
  spirv::Value VisitExpr_(const MulNode* op) override;
  spirv::Value VisitExpr_(const DivNode* op) override;
  spirv::Value VisitExpr_(const ModNode* op) override;
  spirv::Value VisitExpr_(const MinNode* op) override;
  spirv::Value VisitExpr_(const MaxNode* op) override;
  spirv::Value VisitExpr_(const LTNode* op) override;
  spirv::Value VisitExpr_(const LENode* op) override;
  spirv::Value VisitExpr_(const GTNode* op) override;
  spirv::Value VisitExpr_(const GENode* op) override;
  spirv::Value VisitExpr_(const EQNode* op) override;
  spirv::Value VisitExpr_(const NENode* op) override;
  spirv::Value VisitExpr_(const AndNode* op) override;
  spirv::Value VisitExpr_(const OrNode* op) override;
  spirv::Value VisitExpr_(const NotNode* op) override;
  spirv::Value VisitExpr_(const SelectNode* op) override;
  spirv::Value VisitExpr_(const LetNode* op) override;
  spirv::Value VisitExpr_(const CallNode* op) override;
  spirv::Value VisitExpr_(const RampNode* op) override;
  spirv::Value VisitExpr_(const BroadcastNode* op) override;
  spirv::Value VisitExpr_(const BufferLoadNode* op) override;
  spirv::Value VisitExpr_(const ShuffleNode* op) override;
  // stmt
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const WhileNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const DeclBufferNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;

 protected:
  /*! \brief Storage information for a buffer */
  struct StorageInfo {
    /*! \brief The name of the tir::Var for the buffer
     *
     * Used for error messages.
     */
    std::string name_hint;

    /*! \brief Whether it is volatile */
    bool is_volatile{false};

    /*! \brief Whether the element type of the buffer is known.
     *
     * This value is determined based on the type_annotation of the
     * buffer variable (AllocateNode) or of the parameter (shader
     * arguments).
     */
    bool element_type_known{false};

    /*! \brief The known element type of the buffer.
     *
     * This value is determined based on the type_annotation of the
     * buffer variable (AllocateNode) or of the parameter (shader
     * arguments).
     */
    DataType element_type{DataType()};

    /* \brief Check that the access type matches the known type
     *
     * Asserts that the type given is the same as the type previously
     * stored in this array.
     *
     * @param type The data type being stored/loaded in the buffer
     *
     * @param index_lanes The number of lanes of the index.  The
     * number of lanes in the value being stored/loaded should be the
     * product of the number of lanes of the buffer element type and
     * the number of lanes of the index.
     */
    void CheckContentType(DataType type, int index_lanes = 1) const {
      ICHECK(element_type_known) << "Cannot check element type of buffer " << name_hint
                                 << " no previous element type defined";
      DataType expected_type = element_type.with_lanes(index_lanes * element_type.lanes());
      ICHECK_EQ(type, expected_type) << "Attempted to access buffer " << name_hint
                                     << " as element type " << type << " using an index of size "
                                     << index_lanes << " when the element type is " << element_type;
    }

    // Update content type if it hasn't been updated.
    void SetContentType(DataType type, std::string name_hint) {
      ICHECK(!element_type_known) << "Cannot set element type of buffer " << name_hint
                                  << " a second time.";
      this->element_type = type;
      this->name_hint = name_hint;
      element_type_known = true;
    }
  };

  struct FragmentInfo {
    std::string shape;
    std::string scope;
    spirv::SType stype;
    spv::StorageClass sclass;
  };

  // Reset the state so it works for a new function.
  void InitFuncState();
  // Get the thread index
  spirv::Value GetThreadIndex(const IterVar& iv, const PrimExpr& extent);

  spirv::Value CreateStorageSync(const CallNode* op);
  void Scalarize(const PrimExpr& e, std::function<void(int i, spirv::Value v)> f);

  spirv::SType GetFragmentSType(const VarNode* buffer, const DataType& dtype);
  DataType GetElementDataType(const VarNode* buffer);

  // SPIRV-related capabilities of the target
  SPIRVSupport spirv_support_;

  // The builder
  std::unique_ptr<spirv::IRBuilder> builder_;

  // Work group size of three
  uint32_t workgroup_size_[3];

  // Likely branch
  uint32_t weight_likely_branch_{128};

  /* The data type used for the backing array for booleans.
   *
   * Currently matched to the data type used in Buffer::vstore and
   * Buffer::vload.  In the future, this should be the smallest
   * integer type supported by the device, as not all Vulkan
   * implementations support int8.
   */
  DataType boolean_storage_type_{DataType::Int(8)};

  // the storage scope of allocation
  std::unordered_map<const VarNode*, StorageInfo> storage_info_;

  // The definition of local variable.
  std::unordered_map<const VarNode*, spirv::Value> var_map_;

  // The analyzer.
  std::unique_ptr<arith::Analyzer> analyzer_;

  // deep comparison of PrimExpr
  ExprDeepEqual deep_equal_;

  // binding of let variables. Enables duplicate var defs that map to same value
  std::unordered_map<Var, const LetNode*, ObjectPtrHash, ObjectPtrEqual> let_binding_;

  // Running total of the number of bytes of shared memory used.
  // Checked against the max_shared_memory_per_group
  size_t shared_memory_bytes_used_{0};

  std::unordered_map<const VarNode*, FragmentInfo> fragment_info_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SPIRV_CODEGEN_SPIRV_H_
