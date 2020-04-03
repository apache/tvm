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
 * \file tvm/arith/linear_system.h
 * \brief Linear system data structures and solvers
 */
#ifndef TVM_ARITH_LINEAR_SYSTEM_H_
#define TVM_ARITH_LINEAR_SYSTEM_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/expr.h>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace arith {

using tir::Var;
using tir::VarNode;
using tir::IterVar;

/*!
 * \brief Represent a linear system including variables, their ranges and
 *        the relations between them (either equations or inequalities).
 * \sa LinearSystem
 */
class LinearSystemNode : public Object {
 public:
  // e.g., \alpha, \beta
  Array<Var> variables;
  // e.g., 1 <= \alpha <= N, etc.
  Map<Var, Range> ranges;
  // linear equalities or inequalities
  // e.g., A \alpha = \beta or A \alpha <= \beta
  Array<PrimExpr> relations;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("variables", &variables);
    v->Visit("ranges", &ranges);
    v->Visit("relations", &relations);
  }

  static constexpr const char* _type_key = "arith.LinearSystem";
  TVM_DECLARE_FINAL_OBJECT_INFO(LinearSystemNode, Object);
};

/*!
 * \brief Managed reference to LinearSystemNode.
 * \sa LinearSystemNode
 */
class LinearSystem : public ObjectRef {
 public:
  /*!
   * \brief Constructor by fields
   * \param variables The variables in the system.
   * \param ranges    The ranges of the variables.
   * \param relations The linear relations between the variables
   *                  (either equations or inequalities)
   */
  TVM_DLL LinearSystem(Array<Var> variables,
                       Map<Var, Range> ranges,
                       Array<PrimExpr> relations);

  TVM_DEFINE_OBJECT_REF_METHODS(LinearSystem, ObjectRef, LinearSystemNode);
};

/*!
 * \brief We can have different set of variables to represent the same linear system.
 *        For example, the following two systems are equivalent,
 *        {a + b = 0 | a >= 0, b >= 0} and
 *        {m - n = 0 | m >= 0, n <= 0}
 *        This data structure represents the transformation
 *        between two equivalent linear systems.
 *        In the above example,
 *        src        : {a + b = 0 | a >= 0, b >= 0}
 *        dst        : {m - n = 0 | m >= 0, n <= 0}
 *        src_to_dst : {a -> m, b -> -n}
 *        dst_to_src : {m -> a, n -> -b}
 * \sa LinearSystemTransform
 */
class LinearSystemTransformNode : public Object {
 public:
  LinearSystem src;
  LinearSystem dst;
  Map<Var, PrimExpr> src_to_dst;
  Map<Var, PrimExpr> dst_to_src;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("src", &src);
    v->Visit("dst", &dst);
    v->Visit("src_to_dst", &src_to_dst);
    v->Visit("dst_to_src", &dst_to_src);
  }

  static constexpr const char* _type_key = "arith.LinearSystemTransform";
  TVM_DECLARE_FINAL_OBJECT_INFO(LinearSystemTransformNode, Object);
};

/*!
 * \brief Managed reference to LinearSystemTransformNode.
 * \sa LinearSystemTransformNode
 */
class LinearSystemTransform : public ObjectRef {
 public:
  /*!
   * \brief Constructor by fields
   * \param src        source linear system, e.g., {a + b = 0 | a >= 0, b >= 0}
   * \param dst        linear system equivalent to the source, e.g., {m - n = 0 | m >= 0, n <= 0}
   * \param src_to_dst mapping from variables in the \p src to the variables in the \p dst,
   *                   e.g., {a -> m, b -> -n}
   * \param dst_to_src mapping from variables in the \p dst to the variables in the \p src,
   *                   e.g., {m -> a, n -> -b}
   */
  TVM_DLL LinearSystemTransform(LinearSystem src,
                                LinearSystem dst,
                                Map<Var, PrimExpr> src_to_dst,
                                Map<Var, PrimExpr> dst_to_src);

  TVM_DEFINE_OBJECT_REF_METHODS(LinearSystemTransform, ObjectRef, LinearSystemTransformNode);
};

/*!
 * \brief Obtain Smith Normal Form of linear equation A x = y.
 *        Smith Normal Form of matrix A_{mxn} is S_{mxn} = U_{mxm} A_{mxn} V_{nxn},
 *        in which S_{mxn} is diag(s1, s2, ..., sr, 0, ..., 0) and r is the rank of A.
 *        NOTE: Although in standard Smith Normal Form the diagonal elements satisfy
 *              s_i | s_{i+1} (| means divides), the implement here does not guarantee it.
 *        U_{mxm} and V_{nxn} are invertible matrices.
 *        This function modifies \p S to be S_{mxn}, \p V to be V_{nxn},
 *        \p y to be U_{mxm} y_{mx1} and \p x to be V^{-1} x.
 * \param S  the original A_{mxn}, it will be modified to S_{mxn}
 * \param V  an identity matrix, it will be modified to V_{nxn}
 * \param x  the x in A x = y. it will be modified to V^{-1}_{nxn} x_{nx1}
 * \param y  the y in A x = y. it will be modified to U_{mxm} y_{mx1}
 */
void SmithNormalFormDiag(std::vector<std::vector<int64_t>> *S,
                         std::vector<std::vector<int64_t>> *V,
                         std::vector<PrimExpr>* x,
                         std::vector<PrimExpr> *y);

/*!
 * \brief Solve linear equations.
 * \param system_to_solve the variables to solve, their ranges, and a list of equations.
 * \return  A new linear system, with less variables (if \p system_to_solve is NOT of full rank),
 *          or no variable (if \p system_to_solve is of full rank),
 *          or an empty linear system (if \p system_to_solve is unsolvable).
 *          It also provides the ranges of the variables in the new system,
 *          as well as inequalities inferred from the \p system_to_solve.
 *          You can get the mapping from the original variables to the solution via ret->src_to_dst.
 */
LinearSystemTransform SolveEquations(const LinearSystem& system_to_solve);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_LINEAR_SYSTEM_H_
