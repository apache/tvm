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
 * \file tvm/relax/nested_msg.h
 * \brief Helper container to store nested message for robust tuple-aware analysis.
 *
 * Please see NestedMsg for description of usage.
 *
 * \sa NestedMsg
 */
#ifndef TVM_RELAX_NESTED_MSG_H_
#define TVM_RELAX_NESTED_MSG_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>

#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/*!
 * \brief Container that stores possibly nested message with leaf message type T.
 *
 * NestedMsg is a helper structure to store intermediate
 * message state in pass analysis so we can robustly handle message
 * passing with the presence of nested tuple types.
 *
 * Under the hood, NestedMsg[T] = Union[T, NullOpt, Array[NestedMsg[T]]].
 * Each nested message corresponds to the same nesting structure as
 * the nested tuple types when we encounter them in analysis.
 *
 * Relax support nested tuple structures in the IR. Nested tuple structure
 * is important to support advanced groupings in cases such as gradient calculation
 * and other scenarios.
 *
 * The possible presence of nested tuple does mean that we need to
 * to robustly handle analysis that contains nested tuple structures
 * in a dataflow graph.
 *
 * \code
 *
 * v1 = relu(v0)
 * v2 = exp(v0)
 * t = ((v0, v1), (v2,), v0)
 * t1 = t[0]
 * v3 = concat(t1)
 * v4 = t[2]
 * v5 = add(v4, v3)
 *
 * \endcode
 *
 * Consider the above code sequence that contains a mixture of tuple
 * nesting and normal operations. A common message-passing-based analysis
 * will track messages attached to each intermediate variable.
 *
 * Because the intermediate value can contain nested-tuples, we need to have
 * abilities to nest messages according to tuple structure and propagate them
 * along the way. In python, this simply corresponds to using a tuple to hold
 * nested messages. This class provides a helper wrapper in C++ to present such
 * possibly nested message for a given leaf message.
 *
 * This design pattern is necessary to handle tuple values regardless of
 * the normal form design of the IR to enable different messages for each
 * tuple component without enforcing all tuple elements to have the same message.
 *
 * Please consider the following patterns in our pass:
 *
 * On a forward propagation message passing analysis:
 * - Create map [leafnode=>NestedMsg<T>], scan forward
 * - input_msg = [MapToNestedMsg<T>(x, lookup_map) for x in call->args]
 * - output_msg = ForwardProp[call->op](input_msg, call)
 * - map[binding->var] = output_msg
 * - Use MapToNestedMsg to remap the remaining body.
 *
 * On a backward propagation message passing analysis:
 * - Create map [leafnode=>NestedMsg<T>], scan backward
 * - output_msg = lookup map(binding->var)
 * - handle case when output_msg is null
 * - input_msg = BackProp[call->op](out_msg, call)
 * - for arg, msg in zip(call->args, input_msg),
 *     DecomposeNestedMessage(arg, msg, lambda node, m: update_map(node, m))
 * - update_map(node, m) => CombineNestedMessage(map[node], m)
 *
 * Here leafnode is a node that you would like to propagate messages to
 * such as constant, var and should not include tuple.
 *
 * We also recommend writing unit-test cases that involve nested tuple composition
 * and decomposition.
 *
 * \sa MapToNestedMsg, DecomposeNestedMsg, CombineNestedMsg, ForEachLeaf, Equal
 *
 * \note If you want to write robust message passing-based analysis for
 *       programs that can contain nested tuples, you likely need to
 *       use this class or logic of a similar kind.
 */
template <typename T>
class NestedMsg : public ObjectRef {
 public:
  // default constructors.
  NestedMsg() = default;
  NestedMsg(const NestedMsg<T>&) = default;
  NestedMsg(NestedMsg<T>&&) = default;
  NestedMsg<T>& operator=(const NestedMsg<T>&) = default;
  NestedMsg<T>& operator=(NestedMsg<T>&&) = default;
  /*!
   * \brief Construct from an ObjectPtr
   *        whose type already satisfies the constraint
   * \param ptr
   */
  explicit NestedMsg(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  /*! \brief Nullopt handling */
  NestedMsg(runtime::NullOptType) {}  // NOLINT(*)
  // nullptr handling.
  // disallow implicit conversion as 0 can be implicitly converted to nullptr_t
  explicit NestedMsg(std::nullptr_t) {}
  NestedMsg<T>& operator=(std::nullptr_t) {
    data_ = nullptr;
    return *this;
  }
  // normal value handling.
  NestedMsg(T other)  // NOLINT(*)
      : ObjectRef(std::move(other)) {}
  NestedMsg<T>& operator=(T other) {
    ObjectRef::operator=(std::move(other));
    return *this;
  }
  // Array<NestedMsg<T>> handling
  NestedMsg(Array<NestedMsg<T>, void> other)  // NOLINT(*)
      : ObjectRef(std::move(other)) {}
  NestedMsg<T>& operator=(Array<NestedMsg<T>, void> other) {
    ObjectRef::operator=(std::move(other));
    return *this;
  }

  // initializer list handling
  NestedMsg(std::initializer_list<NestedMsg<T>> other)  // NOLINT(*)
      : NestedMsg(Array<NestedMsg<T>, void>(other)) {}
  NestedMsg<T>& operator=(std::initializer_list<NestedMsg<T>> other) {
    return operator=(Array<NestedMsg<T>, void>(other));
  }

  // delete the int constructor
  // since NestedMsg<Integer>(0) is ambiguous
  // 0 can be implicitly casted to nullptr_t
  explicit NestedMsg(int val) = delete;
  NestedMsg<T>& operator=(int val) = delete;
  // operator overloadings
  bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }

  /*! \return Whether the nested message is not-null leaf value */
  bool IsLeaf() const { return data_ != nullptr && data_->IsInstance<LeafContainerType>(); }

  /*! \return Whether the nested message is null */
  bool IsNull() const { return data_ == nullptr; }

  /*! \return Whether the nested message is nested */
  bool IsNested() const { return data_ != nullptr && data_->IsInstance<ArrayNode>(); }

  /*!
   * \return The underlying leaf value.
   * \note This function checks if the msg is leaf.
   */
  T LeafValue() const {
    ICHECK(IsLeaf());
    return T(data_);
  }

  /*!
   * \return a corresponding nested array.
   * \note This checks if the underlying data type is array.
   */
  Array<NestedMsg<T>, void> NestedArray() const {
    ICHECK(IsNested());
    return Array<NestedMsg<T>, void>(data_);
  }

  using ContainerType = Object;
  using LeafContainerType = typename T::ContainerType;

  static_assert(std::is_base_of<ObjectRef, T>::value, "NestedMsg is only defined for ObjectRef.");

  static constexpr bool _type_is_nullable = true;
};

/*!
 * \brief Apply fvisit for each leaf elements in the nested message.
 * \param fvisit The visit callback.
 * \param msg The input nested message.
 * \tparam T the content type of nested msg
 * \tparam FType the visitor type with signature void fvisit(T)
 */
template <typename T, typename FType>
void ForEachLeaf(const NestedMsg<T>& msg, FType fvisit) {
  if (msg == nullptr) return;
  if (msg.IsLeaf()) {
    fvisit(msg.LeafValue());
  } else {
    for (NestedMsg<T> x : msg.NestedArray()) {
      ForEachLeaf(x, fvisit);
    }
  }
}

/*!
 * \brief Recursively compare two nested messages.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param fequal The equal functor with signature bool fequal(T, T)
 * \tparam T the content type of nested msg
 * \tparam FType the equal comparator type
 */
template <typename T, typename FType>
bool Equal(const NestedMsg<T>& lhs, const NestedMsg<T>& rhs, FType fequal) {
  if (lhs.IsNull()) return rhs.IsNull();
  if (rhs.IsNull()) return lhs.IsNull();
  if (lhs.IsLeaf()) {
    return rhs.IsLeaf() && fequal(lhs.LeafValue(), rhs.LeafValue());
  } else {
    if (!rhs.IsNested()) return false;
    Array<NestedMsg<T>> arr_lhs = lhs.NestedArray();
    Array<NestedMsg<T>> arr_rhs = rhs.NestedArray();
    if (arr_lhs.size() != arr_rhs.size()) return false;
    for (size_t i = 0; i < arr_lhs.size(); ++i) {
      if (!Equal(arr_lhs[i], arr_rhs[i], fequal)) return false;
    }
    return true;
  }
}

/*!
 * \brief Map expr with possible nested-tuple to nested message.
 *
 * This function will unpack recursive tuples and run fmapleaf for each leaf,
 * then recursively combines the results together into a NestedMsg.
 *
 * The nesting structure will corresponds to the tuple structure.
 *
 * \param expr The input expression.
 * \param fmapleaf The mapping function for each leaf with signature `NestedMsg<T> fmap(Expr)`
 * \tparam T the content type of nested msg
 * \tparam FType The mapping function type
 */
template <typename T, typename FType>
NestedMsg<T> MapToNestedMsg(Expr expr, FType fmapleaf) {
  if (auto* tuple = expr.as<TupleNode>()) {
    Array<NestedMsg<T>> res;
    res.reserve(tuple->fields.size());
    for (Expr x : tuple->fields) {
      res.push_back(MapToNestedMsg<T, FType>(x, fmapleaf));
    }
    return res;
  } else {
    return fmapleaf(expr);
  }
}

/*!
 * \brief Map structinfo with possible nested-sinfo to nested message.
 *
 * This function will unpack recursive sinfo and run fmapleaf for each leaf,
 * then recursively combines the results together into a NestedMsg.
 *
 * The nesting structure will corresponds to the tuple structure.
 *
 * \param sinfo The input struct info.
 * \param fmapleaf The mapping function for each leaf with signature `NestedMsg<T> fmap(StructInfo)`
 * \tparam T the content type of nested msg
 * \tparam FType The mapping function type
 */
template <typename T, typename FType>
NestedMsg<T> MapToNestedMsg(StructInfo sinfo, FType fmapleaf) {
  if (auto* tuple = sinfo.as<TupleStructInfoNode>()) {
    Array<NestedMsg<T>> res;
    res.reserve(tuple->fields.size());
    for (StructInfo x : tuple->fields) {
      res.push_back(MapToNestedMsg<T, FType>(x, fmapleaf));
    }
    return res;
  } else {
    return fmapleaf(sinfo);
  }
}

/*!
 * \brief Map expr with possible nested-tuple to nested message.
 *
 * This function will unpack recursive expr by its struct info and
 * run fmapleaf for each leaf, then recursively combines the results
 * together into a NestedMsg.
 *
 * The nesting structure will corresponds to the struct info of expr.
 *
 * \param expr The input expression which should have struct info.
 * \param fmapleaf The mapping function for each leaf with signature `NestedMsg<T> fmapleaf(Expr)`
 * \tparam T the content type of nested msg
 * \tparam FType The mapping function type
 */
template <typename T, typename FType>
NestedMsg<T> MapToNestedMsgBySInfo(Expr expr, FType fmapleaf) {
  auto sinfo = GetStructInfo(expr);
  if (auto* tuple = sinfo.as<TupleStructInfoNode>()) {
    Array<NestedMsg<T>> res;
    res.reserve(tuple->fields.size());
    for (size_t i = 0; i < tuple->fields.size(); ++i) {
      Expr field;
      if (const auto* expr_tuple = expr.as<TupleNode>()) {
        field = expr_tuple->fields[i];
      } else {
        field = TupleGetItem(expr, i);
      }
      res.push_back(MapToNestedMsgBySInfo<T, FType>(field, fmapleaf));
    }
    return res;
  } else {
    return fmapleaf(expr);
  }
}

/*!
 * \brief Map nested message back to TargetType.
 *
 * This function will decompose the nested message and
 * run fmapleaf for each leaf message and get the leaf value,
 * then recursively combines the results by fcombine.
 *
 * \param msg The input nested message.
 * \param fmapleaf The mapping function for each leaf with signature
 * `TargetType fmapleaf(Optional<T>)`.
 * \param fcombine The function for combining all childs of a node into TargetType with signature
 * `TargetType fmapleaf(Array<TargetType>)`.
 * \tparam TargetType the target type to map nested msg to.
 * \tparam T the content type of nested msg.
 * \tparam FMapLeaf The leaf mapping function type.
 * \tparam FCombine The combining function type.
 */
template <typename TargetType, typename T, typename FMapLeaf, typename FCombine>
TargetType NestedMsgTo(NestedMsg<T> msg, FMapLeaf fmapleaf, FCombine fcombine) {
  if (msg.IsNull()) {
    return fmapleaf(NullOpt);
  } else if (msg.IsLeaf()) {
    return fmapleaf(msg.LeafValue());
  } else {
    ICHECK(msg.IsNested());
    Array<NestedMsg<T>> arr = msg.NestedArray();
    Array<TargetType> subexpr;
    subexpr.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
      subexpr.push_back(NestedMsgTo<TargetType>(arr[i], fmapleaf, fcombine));
    }
    return fcombine(subexpr);
  }
}

/*!
 * \brief Map nested message back to the expr.
 *
 * This function will decompose the nested message and
 * run fmapleaf for each leaf message and get the leaf expr,
 * then recursively combines the results as tuple expr.
 *
 * \param msg The input nested message.
 * \param fmapleaf The mapping function for each leaf with signature `Expr fmapleaf(Optional<T>)`.
 * \tparam T the content type of nested msg.
 * \tparam FType The mapping function type.
 */
template <typename T, typename FType>
Expr NestedMsgToExpr(NestedMsg<T> msg, FType fmapleaf) {
  return NestedMsgTo<Expr>(msg, fmapleaf, [](Array<Expr> arr) {
    Optional<Expr> simplified_tuple;
    bool simplified_flag = false;
    if (arr.size() >= 1) {
      simplified_flag = true;
      for (size_t i = 0; i < arr.size() && simplified_flag; ++i) {
        auto* node = arr[i].as<TupleGetItemNode>();
        if (node == nullptr || node->index != static_cast<int>(i)) {
          simplified_flag = false;
        } else {
          if (simplified_tuple.defined()) {
            simplified_flag &= (simplified_tuple == node->tuple);
          } else {
            simplified_tuple = node->tuple;
            ICHECK(simplified_tuple.defined());
          }
        }
      }
    }
    return simplified_flag ? simplified_tuple.value() : Tuple(arr);
  });
}

/*!
 * \brief Recursively combine two nested message into one.
 *
 * This function requires the two messages to be compatible with each other.
 * The combination rule is as follows:
 * - combine(null, msg) => msg
 * - combine(leaf1, leaf2) => fcombine(leaf1, leaf2)
 * - combine(array1, array2) => [combine(x, y) for x, y in zip(array1, array2)]
 * - This function will throw an error if array have different size
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param fcombine with signature T fcombine(T lhs, T rhs)
 * \tparam T the content type of nested msg
 * \tparam FType combine function type.
 */
template <typename T, typename FType>
NestedMsg<T> CombineNestedMsg(NestedMsg<T> lhs, NestedMsg<T> rhs, FType fcombine) {
  if (lhs.IsNull()) return rhs;
  if (rhs.IsNull()) return lhs;

  if (lhs.IsLeaf()) {
    ICHECK(rhs.IsLeaf()) << "Cannot combine leaf with nested";
    return NestedMsg<T>(fcombine(lhs.LeafValue(), rhs.LeafValue()));
  } else {
    ICHECK(lhs.IsNested());
    ICHECK(rhs.IsNested()) << "Cannot combine leaf with nested";
    Array<NestedMsg<T>> arr_lhs = lhs.NestedArray();
    Array<NestedMsg<T>> arr_rhs = rhs.NestedArray();
    ICHECK_EQ(arr_lhs.size(), arr_rhs.size())
        << "Cannot combine two nested array with different sizes";
    Array<NestedMsg<T>> res;
    res.reserve(arr_lhs.size());
    for (size_t i = 0; i < arr_lhs.size(); ++i) {
      res.push_back(CombineNestedMsg<T, FType>(arr_lhs[i], arr_rhs[i], fcombine));
    }
    return NestedMsg<T>(res);
  }
}

/*!
 * \brief Recursively map a nested message to another one, with leaf mapped by the input fmapleaf.
 * \param msg The nested message to be mapped.
 * \param fmapleaf The leaf map function, with signature NestedMsg<T> fmapleaf(T msg)
 * \tparam T The content type of nested message.
 * \tparam FType The leaf map function type.
 * \return The new nested message.
 */
template <typename T, typename FType>
NestedMsg<T> MapNestedMsg(NestedMsg<T> msg, FType fmapleaf) {
  if (msg.IsNull()) {
    return msg;
  } else if (msg.IsLeaf()) {
    return fmapleaf(msg.LeafValue());
  } else {
    ICHECK(msg.IsNested());
    Array<NestedMsg<T>> arr = msg.NestedArray();
    Array<NestedMsg<T>> res;
    res.reserve(arr.size());
    for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
      res.push_back(MapNestedMsg(arr[i], fmapleaf));
    }
    return NestedMsg<T>(res);
  }
}

/*!
 * \brief Recursively decompose the tuple structure in expr and msg along with it.
 *
 * This function will call fvisitleaf for each leaf expression in expr.
 * This function will throw an error if the nesting structure in msg does not
 * match the tuple nesting structure in expr.
 *
 * \param expr The input expression to be decomposed.
 * \param msg The input nested message.
 * \param fvisitleaf with signature fvisitleaf(Expr expr, NestedMsg<T> msg)
 * \tparam T the content type of nested msg
 * \tparam FType The visit function type.
 */
template <typename T, typename FType>
void DecomposeNestedMsg(Expr expr, NestedMsg<T> msg, FType fvisitleaf) {
  if (auto* tuple = expr.as<TupleNode>()) {
    ICHECK(msg.IsNested()) << "Expected nested to match tuple";
    Array<NestedMsg<T>> arr = msg.NestedArray();
    ICHECK_EQ(arr.size(), tuple->fields.size()) << "Expected nested array size to match tuple size";
    for (size_t i = 0; i < arr.size(); ++i) {
      DecomposeNestedMsg(tuple->fields[i], arr[i], fvisitleaf);
    }
  } else {
    fvisitleaf(expr, msg);
  }
}

/*!
 * \brief Recursively transform the tuple structure in expr and msgs along with it.
 *
 * This function will call ftransleaf for each leaf expression in expr.
 * This function will throw an error if the nesting structure in msg does not
 * match the tuple nesting structure in expr.
 *
 * \param expr The input expression to be transform. 
 * \param msgs The input messages to guide the transformation.
 * \param ftransleaf with signature ftransleaf(Expr, Array<NestedMsg<T>>)->Expr
 * \tparam T the content type of nested msg
 * \tparam N the number of messages
 * \tparam FType The visit function type.
 */
template <typename T, std::size_t N, typename FType>
Expr TransformTupleLeaf(Expr expr, std::array<NestedMsg<T>, N> msgs, FType ftransleaf) {
  StructInfo sinfo = GetStructInfo(expr);
  if (const auto* tuple = sinfo.as<TupleStructInfoNode>()) {
    std::array<Array<NestedMsg<T>>, N> msg_arrays;
    for (size_t i = 0; i < N; ++i) {
      ICHECK(msgs[i].IsNested()) << "Expected nested to match tuple";
      msg_arrays[i] = msgs[i].NestedArray();
    }
    bool same = true;
    Array<Expr> fields;
    fields.reserve(tuple->fields.size());
    for (size_t i = 0; i < tuple->fields.size(); ++i) {
      Expr field;
      if (const auto* expr_tuple = expr.as<TupleNode>()) {
        field = expr_tuple->fields[i];
      } else {
        field = TupleGetItem(expr, i);
      }
      std::array<NestedMsg<T>, N> sub_msgs;
      for (size_t j = 0; j < N; ++j) {
        sub_msgs[j] = msg_arrays[j][i];
      }
      fields.push_back(TransformTupleLeaf(field, std::move(sub_msgs), ftransleaf));
      same &= (fields.back().same_as(field));
    }
    return same ? expr : Tuple(fields);
  } else {
    for (const auto& msg : msgs) {
      ICHECK(msg.IsLeaf()) << "Expected leaf to match non-tuple";
    }
    return ftransleaf(expr, msgs);
  }
}

/*!
 * \brief Recursively transform the tuple structure in sinfo and msgs along with it.
 *
 * This function will call ftransleaf for each leaf sinfo in sinfo.
 * This function will throw an error if the nesting structure in msg does not
 * match the tuple nesting structure in sinfo.
 *
 * \param sinfo The input sinfo to be transform. 
 * \param msgs The input messages to guide the transformation.
 * \param ftransleaf with signature ftransleaf(StructInfo, Array<NestedMsg<T>>)->StructInfo
 * \tparam T the content type of nested msg
 * \tparam N the number of messages
 * \tparam FType The visit function type.
 */
template <typename T, std::size_t N, typename FType>
StructInfo TransformTupleLeaf(StructInfo sinfo, std::array<NestedMsg<T>, N> msgs,
                              FType ftransleaf) {
  if (const auto* tuple = sinfo.as<TupleStructInfoNode>()) {
    std::array<Array<NestedMsg<T>>, N> msg_arrays;
    for (size_t i = 0; i < N; ++i) {
      ICHECK(msgs[i].IsNested()) << "Expected nested to match tuple";
      msg_arrays[i] = msgs[i].NestedArray();
    }
    bool same = true;
    Array<StructInfo> fields;
    fields.reserve(tuple->fields.size());
    for (size_t i = 0; i < tuple->fields.size(); ++i) {
      StructInfo field = tuple->fields[i];
      std::array<NestedMsg<T>, N> sub_msgs;
      for (size_t j = 0; j < N; ++j) {
        sub_msgs[j] = msg_arrays[j][i];
      }
      fields.push_back(TransformTupleLeaf(field, std::move(sub_msgs), ftransleaf));
      same &= (fields.back().same_as(field));
    }
    return same ? sinfo : TupleStructInfo(fields);
  } else {
    for (const auto& msg : msgs) {
      ICHECK(msg.IsLeaf()) << "Expected leaf to match non-tuple";
    }
    return ftransleaf(sinfo, msgs);
  }
}

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_NESTED_MSG_H_
