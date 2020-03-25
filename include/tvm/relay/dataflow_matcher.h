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
 * \file tvm/relay/dataflow_matcher.h
 * \brief A pattern matcher for matching dataflow properties.
 */
#ifndef TVM_RELAY_DATAFLOW_MATCHER_H_
#define TVM_RELAY_DATAFLOW_MATCHER_H_

#include <tvm/relay/dataflow_pattern.h>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace relay {

/*!
 * \brief A dynamical functor that dispatches on in the first DFPattern argument.
 *
 * \tparam FType function signiture
 *  This type is only defined for FType with function signature R(const DFPattern&,
 * Args...)
 */
template <typename FType>
class DFPatternFunctor;

// functions to be overriden.
#define DFPATTERN_FUNCTOR_DEFAULT \
  { return VisitDFPatternDefault_(op, std::forward<Args>(args)...); }

#define RELAY_DFPATTERN_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {          \
    return self->VisitDFPattern_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class DFPatternFunctor<R(const DFPattern& n, Args...)> {
 private:
  using TSelf = DFPatternFunctor<R(const DFPattern& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~DFPatternFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const DFPattern& n, Args... args) {
    return VisitDFPattern(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitDFPattern(const DFPattern& n, Args... args) {
    CHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitDFPattern_(const AltPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const AttrPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const CallPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const ExprPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TupleGetItemPatternNode* op,
                            Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TuplePatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TypePatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const VarPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const WildcardPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPatternDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(AltPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(AttrPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(CallPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(ExprPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(TupleGetItemPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(TuplePatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(TypePatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(VarPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(WildcardPatternNode);
    return vtable;
  }
};

class DFPatternMutator : public DFPatternFunctor<DFPattern(const DFPattern&)> {
 public:
  virtual DFPattern Mutate(const DFPattern& pattern);
  DFPattern VisitDFPattern(const DFPattern& pattern) override;
  DFPattern VisitDFPattern_(const AltPatternNode* op) override;
  DFPattern VisitDFPattern_(const AttrPatternNode* op) override;
  DFPattern VisitDFPattern_(const CallPatternNode* op) override;
  DFPattern VisitDFPattern_(const ExprPatternNode* op) override;
  DFPattern VisitDFPattern_(const TupleGetItemPatternNode* op) override;
  DFPattern VisitDFPattern_(const TuplePatternNode* op) override;
  DFPattern VisitDFPattern_(const TypePatternNode* op) override;
  DFPattern VisitDFPattern_(const VarPatternNode* op) override;
  DFPattern VisitDFPattern_(const WildcardPatternNode* op) override;

 protected:
  std::unordered_map<DFPattern, DFPattern, ObjectHash, ObjectEqual> memo_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_DATAFLOW_MATCHER_H_
