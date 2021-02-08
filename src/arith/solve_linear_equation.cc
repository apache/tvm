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
 * \file tvm/arith/solve_linear_equation.cc
 * \brief Solve linear equations.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "int_operator.h"

namespace tvm {
namespace arith {

using namespace tvm::runtime;

void SmithNormalFormDiag(std::vector<std::vector<int64_t>>* S, std::vector<std::vector<int64_t>>* V,
                         std::vector<PrimExpr>* x, std::vector<PrimExpr>* y) {
  if (S->empty() || V->empty()) return;
  size_t m = S->size();
  size_t n = (*S)[0].size();  // n is # of variables
  ICHECK_EQ(V->size(), n);
  ICHECK_EQ((*V)[0].size(), n);

  for (size_t index = 0; index < std::min(m, n); ++index) {
    // Here A is partially diagonalized, that is A[i, j] is zero for all i, j
    // such that (i < index) or (j < index), unless (i == j).
    // That is, now we are diagonalizing the submatrix with i >= index and j >= index

    // Find a row with a nonzero element in the index-th column
    // (We also prefer rows where this element has minimal abs value)
    size_t best_i = index;
    for (size_t i = best_i; i < m; ++i) {
      int64_t s_old = (*S)[best_i][index];
      int64_t s_new = (*S)[i][index];
      if (s_new != 0) {
        if (s_old == 0 || std::abs(s_new) < std::abs(s_old)) {
          best_i = i;
        }
      }
    }
    // Move the row we found to the index-th position
    std::swap((*S)[index], (*S)[best_i]);
    std::swap((*y)[index], (*y)[best_i]);

    // If the index-th diagonal element is still zero, try to find a column with nonzero index-th
    // element and move it to the index-th position
    if ((*S)[index][index] == 0) {
      for (size_t j = index + 1; j < n; ++j) {
        if ((*S)[index][j] != 0) {
          for (size_t i = index; i < m; ++i) {
            std::swap((*S)[i][index], (*S)[i][j]);
          }
          // swapping columns corresponds to swapping the corresponding x
          std::swap((*x)[index], (*x)[j]);
          for (size_t i = 0; i < n; ++i) {
            std::swap((*V)[i][index], (*V)[i][j]);
          }
          break;
        }
      }
    }

    // If the index-th diagonal element is still zero, then both the index-th row and the index-th
    // column are completely zero, and we don't need to do anything; just go to the next index
    if ((*S)[index][index] == 0) {
      continue;
    }

    // Now the index-th diagonal element is non-zero and we can zero all the index-th column
    // below it by subtracting rows from each other
    for (auto i = index + 1; i < m; ++i) {
      if ((*S)[i][index] != 0) {
        int64_t g, a, b;
        // g = a*matrix[index][index] + b*matrix[i][index]
        if ((*S)[i][index] % (*S)[index][index] != 0) {
          g = ExtendedEuclidean((*S)[index][index], (*S)[i][index], &a, &b);
        } else {
          // Explicitly avoid changing the index-th row. This is important to avoid infinite loop.
          g = (*S)[index][index];
          a = 1;
          b = 0;
        }

        // Let m = S[index][index], n = S[i][index], then the following is true:
        //
        // [ a   n/g ][ m/g  n/g ] = [ 1  0 ]
        // [ b  -m/g ][ b    -a  ] = [ 0  1 ]
        //
        // Note that the two matrices are integer (since g = gcd(m, n)).
        // We will essentially multiply our matrix on the left by a dilated and transposed version
        // of the first of these two matrices. The second matrix is not needed here, however we will
        // use it while zeroing the index-th row.

        int64_t m_g = (*S)[index][index] / g;
        int64_t n_g = (*S)[i][index] / g;

        // Note that j is the index of the column, not the row
        for (size_t j = index; j < (*S)[i].size(); ++j) {
          // Multiply index-th row by a and add the i-th row multiplied by b
          // This will make the index-th diagonal element equal to the gcd
          int64_t new_index_j = a * (*S)[index][j] + b * (*S)[i][j];
          // This transformation performs zeroing of matrix[i][index]
          int64_t new_i_j = n_g * (*S)[index][j] - m_g * (*S)[i][j];
          (*S)[index][j] = new_index_j;
          (*S)[i][j] = new_i_j;
        }
        // We have to do the same with rhs
        PrimExpr ea = tir::make_const((*y)[index].dtype(), a);
        PrimExpr eb = tir::make_const((*y)[i].dtype(), b);
        PrimExpr e_m_g = tir::make_const((*y)[i].dtype(), m_g);
        PrimExpr e_n_g = tir::make_const((*y)[index].dtype(), n_g);
        PrimExpr new_index_rhs = ea * (*y)[index] + eb * (*y)[i];
        PrimExpr new_i_rhs = e_n_g * (*y)[index] - e_m_g * (*y)[i];
        (*y)[index] = new_index_rhs;
        (*y)[i] = new_i_rhs;
      }
    }

    bool changed = false;

    // Now we have to zero the elements of the index-th row by manipulating columns.
    // This is more difficult because column manipulation corresponds to variable manipulation,
    // but the algorithm is essentially the same as before.
    for (size_t j = index + 1; j < n; ++j) {
      if ((*S)[index][j] != 0) {
        int64_t g, a, b;
        // g = a*matrix[index][index] + b*matrix[index][j]
        if ((*S)[index][j] % (*S)[index][index] != 0) {
          g = ExtendedEuclidean((*S)[index][index], (*S)[index][j], &a, &b);
          // During this phase we may disrupt the zeroness of the index-th column, so we will
          // have to take some action if this might have happened.
          changed = true;
        } else {
          // Explicitly avoid changing the index-th column. This is important to avoid infinite
          // loop. Note that here we don't have to set `changed` to true since we don't change the
          // index-th column.
          g = (*S)[index][index];
          a = 1;
          b = 0;
        }

        // Let m = S[index][index], n = S[index][j], then the following is true:
        //
        // [ a   n/g ][ m/g  n/g ] = [ 1  0 ]
        // [ b  -m/g ][ b    -a  ] = [ 0  1 ]
        //
        // Now we are going to multiply our matrix on the right (to manipulate columns instead of
        // rows), we will also transform the old_to_new matrix the same way, and we will use the
        // second matrix to transform new_to_old.

        int64_t m_g = (*S)[index][index] / g;
        int64_t n_g = (*S)[index][j] / g;

        for (size_t i = index; i < m; ++i) {
          int64_t new_i_index = a * (*S)[i][index] + b * (*S)[i][j];
          int64_t new_i_j = n_g * (*S)[i][index] - m_g * (*S)[i][j];
          (*S)[i][index] = new_i_index;
          (*S)[i][j] = new_i_j;
        }
        // We do exactly the same transformations with V
        for (size_t i = 0; i < n; ++i) {
          int64_t new_i_index = a * (*V)[i][index] + b * (*V)[i][j];
          int64_t new_i_j = n_g * (*V)[i][index] - m_g * (*V)[i][j];
          (*V)[i][index] = new_i_index;
          (*V)[i][j] = new_i_j;
        }
        // And apply reverse transformations to new_to_old.
        PrimExpr ea = tir::make_const((*x)[j].dtype(), a);
        PrimExpr eb = tir::make_const((*x)[index].dtype(), b);
        PrimExpr e_m_g = tir::make_const((*x)[index].dtype(), m_g);
        PrimExpr e_n_g = tir::make_const((*x)[j].dtype(), n_g);
        PrimExpr new_index = e_m_g * (*x)[index] + e_n_g * (*x)[j];
        PrimExpr new_j = eb * (*x)[index] - ea * (*x)[j];
        (*x)[index] = new_index;
        (*x)[j] = new_j;
      }
    }

    if (changed) {
      // We might have changed the first column, so we have to zero it once more
      // (or at least check if it's zero), so just perform this iteration once more.
      index -= 1;
    }
  }
}

Map<Var, Range> InferRange(const Map<Var, PrimExpr>& vars_to_infer, const Array<Var>& ori_vars,
                           const Map<Var, Range>& ori_ranges) {
  // The resulting ranges
  Map<Var, Range> new_ranges;

  std::unordered_set<const VarNode*> ori_vset;
  for (const Var& v : ori_vars) {
    ori_vset.insert(v.get());
  }

  std::unordered_map<const VarNode*, IntSet> var_intsets;
  for (const auto& p : ori_ranges) {
    if (!ori_vset.count(p.first.get())) {
      // First of all, fill the new ranges with outer variable ranges
      new_ranges.Set(p.first, p.second);
    }
    // Convert original ranges to IntSets
    var_intsets[p.first.get()] = IntSet::FromRange(p.second);
  }

  // Infer ranges for the new variables and add them to the resulting ranges
  for (const auto& p : vars_to_infer) {
    const auto& var = p.first;
    const auto& expr = p.second;
    Range range = EvalSet(expr, var_intsets).CoverRange(Range());
    if (range.defined()) {
      new_ranges.Set(var, range);
    }
  }
  return new_ranges;
}

// pretty print matrix equation
void DebugPrint(const std::vector<std::vector<int64_t>>& S,
                const std::vector<std::vector<int64_t>>& V, const std::vector<PrimExpr>& V_inv_x,
                const std::vector<PrimExpr>& rhs) {
  std::cout << "S:\n";
  for (size_t i = 0; i < S.size(); ++i) {
    for (auto e : S[i]) {
      std::cout << e << "\t";
    }
    std::cout << "\t->\t" << rhs[i];
    std::cout << "\n";
  }
  std::cout << "V:\n";
  for (const auto& r : V) {
    for (auto e : r) {
      std::cout << e << "\t";
    }
    std::cout << "\n";
  }
  std::cout << "V_inv x:\n" << Array<PrimExpr>(V_inv_x);
  std::cout << "\n" << std::endl;
}

IntConstraintsTransform SolveLinearEquations(const IntConstraints& system_to_solve) {
  // m: # of equations
  // n: # of variables
  // we first construct A_{mxn} x_{nx1} = y_{mx1}
  // then get Smith normal form of matrix A,
  // S_{mxn} = U_{mxm} A_{mxn} V_{nxn}
  // => U^{-1} S V^{-1} x = y
  // S V^{-1} x = U y
  std::vector<PrimExpr> Uy;             // mx1
  std::vector<std::vector<int64_t>> S;  // mxn
  std::vector<std::vector<int64_t>> V;  // nxn
  std::vector<PrimExpr> V_inv_x;        // V^{-1} x, nx1
  // Conditions we don't know what to do with
  std::vector<PrimExpr> rest;

  Analyzer analyzer_problem;
  analyzer_problem.Bind(system_to_solve->ranges);

  size_t num_vars = system_to_solve->variables.size();

  // initialize V_{nxn} with identity matrix,
  // initialize V^{-1} x as x
  for (size_t i = 0; i < num_vars; ++i) {
    V.emplace_back(num_vars);
    V.back()[i] = 1;
    V_inv_x.push_back(system_to_solve->variables[i]);
  }

  // Transform formulas into rows of the matrix
  // S_{mxn} V^{-1}_{nxn} x_{nx1} = U y, in which n is # of variables
  // here we initialize S_{mxn} to be A, U to be identity matrix.
  for (const PrimExpr& equation : system_to_solve->relations) {
    if (const tir::EQNode* eq = equation.as<tir::EQNode>()) {
      // a-b = sum_{i=0}^{n-1} variables[i] * coeff[i] + coeff[n]
      Array<PrimExpr> coeffs = arith::DetectLinearEquation(analyzer_problem.Simplify(eq->a - eq->b),
                                                           system_to_solve->variables);
      if (!coeffs.empty()) {
        std::vector<int64_t> row;
        for (size_t j = 0; j < coeffs.size() - 1; ++j) {
          PrimExpr c = coeffs[j];
          if (const IntImmNode* ic = c.as<IntImmNode>()) {
            row.push_back(ic->value);
          } else {
            // elements in matrix S V must be integers
            // ignore equations that we cannot deal with.
            LOG(WARNING) << "Cannot deal with non-integer coefficients, ignore equation "
                         << equation;
            row.clear();
            break;
          }
        }

        if (!row.empty()) {
          // S V^{-1} (a-b) = Uy
          // V is identity for now
          S.push_back(row);
          Uy.push_back(-coeffs[coeffs.size() - 1]);
          continue;
        }
      }
    }

    // otherwise
    rest.push_back(equation);
  }

  // After diagonalizing, we have
  // S_{mxn} is the Smith normal form (diagonal matrix)
  // V_{nxn} is invertible
  // V_inv_x is V^{-1} \times x
  // Uy is U \times y
  SmithNormalFormDiag(&S, &V, &V_inv_x, &Uy);

  Array<Var> new_vars;
  Array<PrimExpr> new_relations;
  Map<Var, PrimExpr> new_to_old_map;
  Map<Var, PrimExpr> old_to_new_map;

  // Simplify right hand sides
  for (PrimExpr r : Uy) {
    r = analyzer_problem.Simplify(r);
  }

  // Create the relations of the existence of a solution
  for (size_t j = 0; j < S.size(); ++j) {
    PrimExpr new_relation;
    if (j >= num_vars || S[j][j] == 0) {
      // The row of matrix is zero. A solution exists only if the Ub[j] is also zero
      new_relation = (Uy[j] == 0);
    } else {
      // The diagonal element is non-zero. A solution exists only if the diagonal element
      // is a divisor of the Ub[j]
      new_relation = (floormod(Uy[j], std::abs(S[j][j])) == 0);
    }
    new_relation = analyzer_problem.Simplify(new_relation);
    if (tir::is_const_int(new_relation, 0)) {
      // unable to solve the system.
      return IntConstraintsTransform(system_to_solve,
                                     IntConstraints(
                                         /*variables=*/{},
                                         /*ranges=*/{},
                                         /*relations=*/{tir::make_zero(DataType::Bool())}),
                                     {}, {});
    } else if (!tir::is_const_int(new_relation, 1)) {
      new_relations.push_back(new_relation);
    }
  }

  Array<PrimExpr> solution_for_V_inv_x;
  // Now create new variables or directly solve the equations
  // suppose the rank of A is r, aka r = # of non-zeros in S
  // the solution of S_{mxn} V^{-1}_{nxn} x_{nx1} = U b
  // is
  // x = (pseudo-inverse of A) b + K_{(n)x(n-r)} z_{n-r}
  //   = V_{nxn} S^{-1}_{nxm} (Ub)_{mxn} + K_{(n)x(n-r)} z_{n-r}
  // in which K is the right n-r columns of V, z is variable vector
  // thus,
  // V^{-1} x = S^{-1}_{nxm} (Ub)_{mxn} +
  //            [[0, ... 0]_{n-r}, ... [0, ..., 0], diag(1, ..., 1)_{(n-r)x(n-r)}] z_{n-r}
  for (size_t j = 0; j < num_vars; ++j) {
    if (j >= S.size() || S[j][j] == 0) {
      // The j-th variable can take any integer value, create a tvm variable for it
      PrimExpr to_old = analyzer_problem.Simplify(V_inv_x[j]);
      std::string name_hint = "n" + std::to_string(new_vars.size());
      if (const VarNode* v_old = to_old.as<VarNode>()) {
        name_hint += "_" + v_old->name_hint;
      }
      Var v = Var(name_hint, V_inv_x[j].dtype());
      solution_for_V_inv_x.push_back(v);
      new_vars.push_back(v);
      new_to_old_map.Set(v, to_old);
    } else {
      // The j-th variable is just a single value, don't create a tvm variable
      // S^{-1}_{nxm} Uy_{mxn}
      if (S[j][j] >= 0) {
        PrimExpr a = tir::make_const(Uy[j].dtype(), S[j][j]);
        solution_for_V_inv_x.push_back(analyzer_problem.Simplify(floordiv(Uy[j], a)));
      } else {
        // This is required because some simplifiers
        // have problems with dividing by negative numbers
        PrimExpr a = tir::make_const(Uy[j].dtype(), -S[j][j]);
        solution_for_V_inv_x.push_back(analyzer_problem.Simplify(floordiv(-Uy[j], a)));
      }
    }
  }

  // V V^{-1} x = x
  for (size_t i = 0; i < num_vars; ++i) {
    PrimExpr e = tir::make_zero(system_to_solve->variables[i].dtype());
    for (size_t j = 0; j < num_vars; ++j) {
      e = e + tir::make_const(e.dtype(), V[i][j]) * solution_for_V_inv_x[j];
    }
    e = analyzer_problem.Simplify(e);
    old_to_new_map.Set(system_to_solve->variables[i], e);
  }

  // The resulting ranges
  Map<Var, Range> new_ranges =
      InferRange(new_to_old_map, system_to_solve->variables, system_to_solve->ranges);
  Analyzer analyzer_solution;
  analyzer_solution.Bind(new_ranges);

  // We have to transform ranges of the old variables into relations over new variables because
  // new ranges are not enough usually.
  for (const auto& old_var : system_to_solve->variables) {
    if (system_to_solve->ranges.find(old_var) != system_to_solve->ranges.end()) {
      const Range& old_range = system_to_solve->ranges.at(old_var);
      PrimExpr express_by_new_vars = old_to_new_map.at(old_var);
      PrimExpr lower_cond = analyzer_solution.Simplify(old_range->min <= express_by_new_vars);
      PrimExpr upper_cond =
          analyzer_solution.Simplify(express_by_new_vars < old_range->min + old_range->extent);
      if (!tir::is_const_int(lower_cond, 1)) {
        new_relations.push_back(lower_cond);
      }
      if (!tir::is_const_int(upper_cond, 1)) {
        new_relations.push_back(upper_cond);
      }
    }
  }

  // Add the rest conditions
  for (const PrimExpr& cond : rest) {
    new_relations.push_back(Substitute(cond, old_to_new_map));
  }

  IntConstraints solution(new_vars, new_ranges, new_relations);
  IntConstraintsTransform transform(system_to_solve, solution, old_to_new_map, new_to_old_map);

  return transform;
}

TVM_REGISTER_GLOBAL("arith.SolveLinearEquations").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 1) {
    *ret = SolveLinearEquations(args[0]);
  } else if (args.size() == 3) {
    IntConstraints problem(args[0], args[1], args[2]);
    *ret = SolveLinearEquations(problem);
  } else {
    LOG(FATAL) << "arith.SolveLinearEquations expects 1 or 3 arguments, gets " << args.size();
  }
});

}  // namespace arith
}  // namespace tvm
