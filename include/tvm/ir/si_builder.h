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
 * \file tvm/ir/si_builder.h
 * \brief build a source info during rewriting expressions.
 */
#ifndef TVM_IR_SI_BUILDER_H_
#define TVM_IR_SI_BUILDER_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/tir/stmt.h>

#include <memory>
#include <unordered_set>

namespace tvm {

/*!
 * \brief Source Information Builder, SIBuilder provides helper APIs for filling spans,
 *        particularly useful for one-to-many, many-to-one and many-to-many IR transformations.
 */
class SIBuilder {
 public:
  /*!
   * \brief Create SIBuilder from a given span
   */
  explicit SIBuilder(const Span& span = Span());

  /*!
   * \brief Create SIBuilder from a given span sequence
   */
  explicit SIBuilder(const Array<Span>& spans = Array<Span>());
  explicit SIBuilder(const std::initializer_list<Span>& init);

  /*!
   * \brief Create SIBuilder via a subgraph,
   *        Will construct span based on the exprs in the subgraph. Including the inputs exprs.
   *
   * \param entry Entry expr for subgraph
   * \param inputs End exprs for subgraph
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of<BaseExpr, T>::value>>
  explicit SIBuilder(const T& entry, const tvm::Array<T>& inputs = {});
  explicit SIBuilder(const tir::Stmt& entry, const tvm::Array<PrimExpr>& inputs = {});
  explicit SIBuilder(const tir::Stmt& entry, const tvm::Array<tir::Stmt>& inputs = {});

  ~SIBuilder();

  SIBuilder(const SIBuilder&) = delete;
  SIBuilder& operator=(const SIBuilder&) = delete;

  /*!
   * \brief build a span of source information, which is based on the given span or subgraph.
   *
   * \return the built span
   */
  Span Build() const;

  /*!
   * \brief Recursively fill all span of exprs in subgraph from entry until inputs.
   *
   * \param entry Entry expr for subgraph.
   * \param inputs End exprs for subgraph, will not be filled with new span.
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of<BaseExpr, T>::value>>
  void RecursivelyFillSpan(
      const T& entry, const std::unordered_set<T, ObjectPtrHash, ObjectPtrEqual>& inputs) const;

  void RecursivelyFillSpan(
      const tir::Stmt& entry,
      const std::unordered_set<PrimExpr, ObjectPtrHash, ObjectPtrEqual>& inputs) const;
  void RecursivelyFillSpan(
      const tir::Stmt& entry,
      const std::unordered_set<tir::Stmt, ObjectPtrHash, ObjectPtrEqual>& inputs) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  std::unique_ptr<Impl> CreateImpl(const Span& span);
};

}  // namespace tvm

#endif  // TVM_IR_SI_BUILDER_H_
