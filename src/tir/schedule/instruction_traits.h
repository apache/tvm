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
#ifndef TVM_TIR_SCHEDULE_INSTRUCTION_TRAITS_H_
#define TVM_TIR_SCHEDULE_INSTRUCTION_TRAITS_H_

#include <tvm/tir/schedule/instruction.h>
#include <tvm/tir/schedule/schedule.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief Register an InstructionKind using a trait class
 * \param InstructionKindTraits A traits class of an InstructionKind
 *
 * Example:
 *
 * \code
 *
 * struct SomeInstructionKindTraits {
 *   static constexpr const char* kName = "name-of-the-instruction";
 *   static constexpr bool kIsPure = false;
 *
 *   // Convertible to `InstructionKindNode::FInstructionApply`
 *   static Array<ObjectRef> ApplyToSchedule(
 *      const tir::Schedule& sch,
 *      const Array<ObjectRef>& inputs,
 *      const Array<ObjectRef>& attrs,
 *      const Optional<ObjectRef>& decision);
 *
 *   // Convertible to `InstructionKindNode::FInstructionAsPython`
 *   static String AsPython(
 *      const Array<String>& inputs,
 *      const Array<ObjectRef>& attrs,
 *      const Optional<ObjectRef>& decision,
 *      const Array<String>& outputs);
 *
 *   // Convertible to `InstructionKindNode::FInstructionAttrsAsJSON`
 *   static ObjectRef AttrsAsJSON(
 *      const Array<ObjectRef>& attrs);
 *
 *   // Convertible to `InstructionKindNode::FInstructionAttrsFromJSON`
 *   static Array<ObjectRef> AttrsFromJSON(
 *      const ObjectRef& attrs_record);
 * };
 *
 * TVM_REGISTER_INST_KIND_TRAITS(SomeInstructionKindTraits);
 *
 * \endcode
 */
#define TVM_REGISTER_INST_KIND_TRAITS(InstructionKindTraits)         \
  TVM_REGISTER_INST_KIND(InstructionKindTraits::kName)               \
      .set_is_pure(InstructionKindTraits::kIsPure)                   \
      .set_apply_to_schedule(InstructionKindTraits::ApplyToSchedule) \
      .set_attrs_as_json(InstructionKindTraits::AttrsAsJSON)         \
      .set_attrs_from_json(InstructionKindTraits::AttrsFromJSON)     \
      .set_as_python(InstructionKindTraits::AsPython)

/*!
 * \brief A helper to conveniently register an InstructionKind. When inherited in curiously
 * recursive template pattern, the derived class `TTraits` only needs to define two functions on the
 * unpacked inputs, and the helper handles unpacking and downcasting. See the example for more
 * details.
 *
 * \tparam TTraits The derived class
 *
 * Example:
 *
 * \code
 *
 * struct SamplePerfectTileTraits : public UnpackedInstTraits<SamplePerfectTileTraits> {
 *   // The name of this kind of instruction
 *   static constexpr const char* kName = "SamplePerfectTile";
 *   // A boolean indicating if the instruction is pure, i.e. change nothing in the schedule state
 *   static constexpr bool kIsPure = true;
 *   // The number of inputs in this kind of instruction
 *   static constexpr size_t kNumInputs = 1;
 *   // The number of attributes in this kind of instruction
 *   static constexpr size_t kNumAttrs = 2;
 *   // The number of decisions in this kind of instruction (only 0 or 1 is allowed)
 *   static constexpr size_t kNumDecisions = 1;
 *
 *   // Calling convention:
 *   // - All the arguments must be ObjectRef
 *   // - The 1st argument is Schedule
 *   // - The next `kNumInputs` arguments are input random variables
 *   // - The next `kNumAttrs` arguments are attributes
 *   // - The next argument is decision, if `kNumDecisions == 1`
 *   static Array<Var> UnpackedApplyToSchedule(
 *      Schedule sch,
 *      LoopRV loop_rv,
 *      Integer n,
 *      Integer max_innermost_factor,
 *      Optional<Array<Integer>> decision) {
 *     return sch->SamplePerfectTile(loop_rv, n->value, max_innermost_factor->value, decision);
 *   }
 *
 *   // Calling convention:
 *   // - All the arguments must be ObjectRef
 *   // - The 1st argument is an array containing names of output random variables
 *   // - The next `kNumInputs` arguments are names of input random variables
 *   // - The next `kNumAttrs` arguments are attributes
 *   // - The next argument is decision, if `kNumDecisions == 1`
 *   static String UnpackedAsPython(
 *      Array<String> outputs,
 *      String loop_rv,
 *      Integer n,
 *      Integer max_innermost_factor,
 *      Optional<Array<Integer>> decision) {
 *     PythonAPICall py("sample_perfect_tile");
 *     py.Input("loop", loop_rv);
 *     py.Input("n", n->value);
 *     py.Input("max_innermost_factor", max_innermost_factor->value);
 *     py.Decision(decision);
 *     py.OutputList(outputs);
 *     return py.Str();
 *   }
 *
 *   template <typename>
 *   friend struct UnpackedInstTraits;
 * };
 *
 * TVM_REGISTER_INST_KIND(SamplePerfectTileTraits);
 * \endcode
 */
template <class TTraits>
struct UnpackedInstTraits {
  /*!
   * \brief Unpack the arguments in the calling convention, and feed them into
   * `TTraits::UnpackedApplyToSchedule`
   * \sa InstructionKindNode::f_apply_to_schedule
   */
  static Array<ObjectRef> ApplyToSchedule(const Schedule& sch, const Array<ObjectRef>& inputs,
                                          const Array<ObjectRef>& attrs,
                                          const Optional<ObjectRef>& decision);

  /*!
   * \brief Unpack the arguments in the calling convention, and feed them into
   * `TTraits::UnpackedAsPython`
   * \sa InstructionKindNode::f_as_python
   */
  static String AsPython(const Array<ObjectRef>& inputs, const Array<ObjectRef>& attrs,
                         const Optional<ObjectRef>& decision, const Array<String>& outputs);

  /*! \brief No customized serializer by default */
  static constexpr std::nullptr_t AttrsAsJSON = nullptr;

  /*! \brief No customized deserializer by default */
  static constexpr std::nullptr_t AttrsFromJSON = nullptr;

 protected:
  template <size_t index_offset>
  static TVM_ALWAYS_INLINE void _SetInputs(const runtime::TVMArgsSetter& setter,
                                           const Array<ObjectRef>& inputs);
  template <size_t index_offset>
  static TVM_ALWAYS_INLINE void _SetAttrs(const runtime::TVMArgsSetter& setter,
                                          const Array<ObjectRef>& attrs);
  template <size_t index_offset>
  static TVM_ALWAYS_INLINE void _SetDecision(const runtime::TVMArgsSetter& setter,
                                             const Optional<ObjectRef>& decision);
  static TVM_ALWAYS_INLINE Array<ObjectRef> _ConvertOutputs(const TVMRetValue& rv);
};

/*!
 * \brief A helper class that constructs schedule API call in python syntax,
 * which helps convert an Inst to a python statement.
 * \sa InstructionKindNode::f_as_python
 */
class PythonAPICall {
 public:
  /*!
   * \brief Constructor
   * \param method_name The name of the schedule API to be called
   */
  explicit PythonAPICall(String method_name) : method_name_(method_name), output_(NullOpt) {}
  /*! \brief Add an integer input */
  inline void Input(String arg_name, int arg);
  /*! \brief Add an integer input */
  inline void Input(String arg_name, int64_t arg);
  /*! \brief Add a bool input */
  inline void Input(String arg_name, bool arg);
  /*! \brief Add a double input */
  inline void Input(String arg_name, double arg);
  /*! \brief Add an input random variable */
  inline void Input(String arg_name, String arg);
  /*! \brief Add an input, dispatched to different implementations according to the object's type */
  inline void Input(String arg_name, ObjectRef arg);
  /*! \brief Add the decision */
  inline void Decision(ObjectRef decision);
  /*!
   * \brief Add a single output random variable
   * \param unit_array An array containing only one element
   */
  inline void SingleOutput(Array<String> unit_array);
  /*! \brief Add a list of output random variables */
  inline void OutputList(Array<String> outputs);
  /*! \returns The schedule API call in python syntax */
  inline String Str() const;

 private:
  /*! \brief Converts a TVM object to python string and print to the output stream */
  inline void AsPythonString(const ObjectRef& obj, std::ostream& os);

 private:
  /*! \brief The name of the API to call */
  String method_name_;
  /*! \brief The output of the instruction */
  Optional<String> output_;
  /*! \brief The names of input arguments */
  std::vector<String> arg_names_;
  /*! \brief The values of input arguments */
  std::vector<String> args_;
};

/********** implementation details **********/

// forward declaration
namespace details {

template <typename... Args>
struct _ArgsPacker;

template <>
struct _ArgsPacker<> {
  static constexpr bool checked = true;
};

template <typename TObjectRef, typename... Args>
struct _ArgsPacker<TObjectRef, Args...> {
  static constexpr bool checked =
      std::is_base_of<ObjectRef, TObjectRef>::value && _ArgsPacker<Args...>::checked;
};

template <typename T>
struct _MethodType {};

template <typename TReturn, typename... Args>
struct _MethodType<TReturn(Args...)> {
  using return_type = TReturn;
  using argument_type = _ArgsPacker<Args...>;
};

template <typename T>
struct _NumArgs {};

template <typename TReturn, typename... Args>
struct _NumArgs<TReturn(Args...)> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename>
struct _IsTVMArray : std::false_type {};

template <typename T>
struct _IsTVMArray<runtime::Array<T>> : std::true_type {};

template <typename T>
struct _IsSingleObject
    : std::integral_constant<bool, std::is_base_of<ObjectRef, T>::value && !_IsTVMArray<T>::value> {
};

template <class T>
using ReturnType = typename _MethodType<std::remove_cv_t<T>>::return_type;

template <class T>
static constexpr bool ArgumentAreAllObjects =
    _MethodType<std::remove_cv_t<T>>::argument_type::checked;

template <class T>
static constexpr size_t NumArgs = _NumArgs<std::remove_cv_t<T>>::value;

template <class T>
static constexpr int IsTVMArray = _IsTVMArray<std::remove_cv_t<T>>::value;

template <class T>
static constexpr int IsSingleObject = _IsSingleObject<std::remove_cv_t<T>>::value;

};  // namespace details

template <class TTraits>
Array<ObjectRef> UnpackedInstTraits<TTraits>::ApplyToSchedule(const Schedule& sch,
                                                              const Array<ObjectRef>& inputs,
                                                              const Array<ObjectRef>& attrs,
                                                              const Optional<ObjectRef>& decision) {
  using method_type = decltype(TTraits::UnpackedApplyToSchedule);
  using return_type = details::ReturnType<method_type>;
  static_assert(details::ArgumentAreAllObjects<method_type>,
                "All arguments to `UnpackedApplyToSchedule` must be subclasses of ObjectRef");
  constexpr size_t kNumArgs = details::NumArgs<method_type>;
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumArgs == 1 + kNumInputs + kNumAttrs + kNumDecisions,
                "length of argument list mismatch");
  TVMValue tvm_values[kNumArgs];
  int tvm_type_codes[kNumArgs];
  runtime::TVMArgsSetter setter(tvm_values, tvm_type_codes);
  setter(0, sch);
  TTraits::template _SetInputs<1>(setter, inputs);
  TTraits::template _SetAttrs<1 + kNumInputs>(setter, attrs);
  TTraits::template _SetDecision<1 + kNumInputs + kNumAttrs>(setter, decision);
  PackedFunc pf([](const TVMArgs& args, TVMRetValue* rv) -> void {
    using runtime::detail::unpack_call;
    constexpr size_t kNumArgs = details::NumArgs<method_type>;
    ICHECK_EQ(args.size(), kNumArgs);
    unpack_call<return_type, kNumArgs>(nullptr, TTraits::UnpackedApplyToSchedule, args, rv);
  });
  TVMRetValue rv;
  pf.CallPacked(TVMArgs(tvm_values, tvm_type_codes, kNumArgs), &rv);
  return TTraits::_ConvertOutputs(rv);
}

template <class TTraits>
String UnpackedInstTraits<TTraits>::AsPython(const Array<ObjectRef>& inputs,
                                             const Array<ObjectRef>& attrs,
                                             const Optional<ObjectRef>& decision,
                                             const Array<String>& outputs) {
  using method_type = decltype(TTraits::UnpackedAsPython);
  using return_type = details::ReturnType<method_type>;
  static_assert(details::ArgumentAreAllObjects<method_type>,
                "All arguments to `UnpackedAsPython` must be subclasses of ObjectRef");
  constexpr size_t kNumArgs = details::NumArgs<method_type>;
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumArgs == 1 + kNumInputs + kNumAttrs + kNumDecisions,
                "length of argument list mismatch");
  TVMValue tvm_values[kNumArgs];
  int tvm_type_codes[kNumArgs];
  runtime::TVMArgsSetter setter(tvm_values, tvm_type_codes);
  setter(0, outputs);
  TTraits::template _SetInputs<1>(setter, inputs);
  TTraits::template _SetAttrs<1 + kNumInputs>(setter, attrs);
  TTraits::template _SetDecision<1 + kNumInputs + kNumAttrs>(setter, decision);
  PackedFunc pf([](const TVMArgs& args, TVMRetValue* rv) -> void {
    using runtime::detail::unpack_call;
    constexpr size_t kNumArgs = details::NumArgs<method_type>;
    ICHECK_EQ(args.size(), kNumArgs);
    unpack_call<return_type, kNumArgs>(nullptr, TTraits::UnpackedAsPython, args, rv);
  });
  TVMRetValue rv;
  pf.CallPacked(TVMArgs(tvm_values, tvm_type_codes, kNumArgs), &rv);
  String result = rv;
  return result;
}

template <class TTraits>
template <size_t index_offset>
TVM_ALWAYS_INLINE void UnpackedInstTraits<TTraits>::_SetInputs(const runtime::TVMArgsSetter& setter,
                                                               const Array<ObjectRef>& inputs) {
  constexpr size_t kNumInputs = TTraits::kNumInputs;
  ICHECK_EQ(kNumInputs, inputs.size())
      << "ValueError: Incorrect kNumInputs for instruction: " << TTraits::kName;
  const ObjectRef* ptr = inputs.template as<ArrayNode>()->begin();
  for (size_t i = 0; i < kNumInputs; ++i) {
    setter(i + index_offset, *(ptr + i));
  }
}

template <class TTraits>
template <size_t index_offset>
TVM_ALWAYS_INLINE void UnpackedInstTraits<TTraits>::_SetAttrs(const runtime::TVMArgsSetter& setter,
                                                              const Array<ObjectRef>& attrs) {
  constexpr size_t kNumAttrs = TTraits::kNumAttrs;
  ICHECK_EQ(kNumAttrs, attrs.size())
      << "ValueError: Incorrect kNumAttrs for instruction: " << TTraits::kName;
  const ObjectRef* ptr = attrs.as<ArrayNode>()->begin();
  for (size_t i = 0; i < kNumAttrs; ++i) {
    setter(i + index_offset, *(ptr + i));
  }
}

template <class TTraits>
template <size_t index_offset>
TVM_ALWAYS_INLINE void UnpackedInstTraits<TTraits>::_SetDecision(
    const runtime::TVMArgsSetter& setter, const Optional<ObjectRef>& decision) {
  constexpr size_t kNumDecisions = TTraits::kNumDecisions;
  static_assert(kNumDecisions <= 1, "an instruction is supposed to have at most 1 decision");
  if (kNumDecisions == 1) {
    setter(index_offset, decision);
  } else {
    ICHECK(!decision.defined());
  }
}

template <class TTraits>
TVM_ALWAYS_INLINE Array<ObjectRef> UnpackedInstTraits<TTraits>::_ConvertOutputs(
    const TVMRetValue& rv) {
  using method_type = decltype(TTraits::UnpackedApplyToSchedule);
  using return_type = details::ReturnType<method_type>;
  constexpr int is_array = details::IsTVMArray<return_type>;
  constexpr int is_single_obj = details::IsSingleObject<return_type>;
  constexpr int is_void = std::is_void<return_type>::value;
  static_assert(is_array || is_single_obj || is_void, "return type not supported");
  static_assert(is_array + is_single_obj + is_void == 1, "internal template error");
  if (is_void) {
    return {};
  } else if (is_single_obj) {
    ObjectRef obj = rv;
    return {obj};
  } else if (is_array) {
    ObjectRef obj = rv;
    const ArrayNode* array = obj.as<ArrayNode>();
    return GetRef<Array<ObjectRef>>(array);
  }
}

/********** PythonAPICall **********/

inline void PythonAPICall::AsPythonString(const ObjectRef& obj, std::ostream& os) {
  if (!obj.defined()) {
    os << "None";
  } else if (const auto* str = obj.as<runtime::StringObj>()) {
    os << str->data;
  } else if (const auto* int_imm = obj.as<IntImmNode>()) {
    os << int_imm->value;
  } else if (const auto* float_imm = obj.as<FloatImmNode>()) {
    os.precision(17);
    os << float_imm->value;
  } else if (const auto* runtime_int = obj.as<runtime::Int::ContainerType>()) {
    os << runtime_int->value;
  } else if (const auto* runtime_float = obj.as<runtime::Float::ContainerType>()) {
    os.precision(17);
    os << runtime_float->value;
  } else if (const auto* array = obj.as<ArrayNode>()) {
    os << '[';
    bool is_first = true;
    for (const ObjectRef& e : *array) {
      if (is_first) {
        is_first = false;
      } else {
        os << ", ";
      }
      AsPythonString(e, os);
    }
    os << ']';
  } else if (const auto* dict = obj.as<MapNode>()) {
    os << '{';
    bool is_first = true;
    std::vector<std::pair<std::string, std::string>> dict_items;
    for (auto it = dict->begin(); it != dict->end(); ++it) {
      std::ostringstream ks;
      AsPythonString(it->first, ks);
      std::ostringstream vs;
      AsPythonString(it->second, vs);
      dict_items.emplace_back(ks.str(), vs.str());
    }
    std::sort(dict_items.begin(), dict_items.end(),
              [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
    for (const auto& kv : dict_items) {
      if (is_first) {
        is_first = false;
      } else {
        os << ", ";
      }
      os << '\"' << kv.first << "\": " << kv.second;
    }
    os << '}';
  } else {
    LOG(FATAL) << "ValueError: Cannot translate type '" << obj->GetTypeKey()
               << "' to python. Its value is: " << obj;
    throw;
  }
}

void PythonAPICall::Input(String arg_name, int arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.push_back(std::to_string(arg));
}

void PythonAPICall::Input(String arg_name, int64_t arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.push_back(std::to_string(arg));
}

void PythonAPICall::Input(String arg_name, bool arg) {
  static const char* true_str = "True";
  static const char* false_str = "False";
  arg_names_.emplace_back(std::move(arg_name));
  if (arg) {
    args_.push_back(true_str);
  } else {
    args_.push_back(false_str);
  }
}

void PythonAPICall::Input(String arg_name, double arg) {
  arg_names_.emplace_back(std::move(arg_name));
  std::ostringstream os;
  os.precision(17);
  os << arg;
  args_.push_back(os.str());
}

void PythonAPICall::Input(String arg_name, String arg) {
  arg_names_.emplace_back(std::move(arg_name));
  args_.emplace_back(std::move(arg));
}

void PythonAPICall::Input(String arg_name, ObjectRef arg) {
  arg_names_.emplace_back(std::move(arg_name));
  std::ostringstream os;
  AsPythonString(arg, os);
  args_.push_back(os.str());
}

void PythonAPICall::Decision(ObjectRef decision) {
  if (decision.defined()) {
    this->Input("decision", decision);
  }
}

void PythonAPICall::SingleOutput(Array<String> unit_array) {
  ICHECK_EQ(unit_array.size(), 1);
  this->output_ = unit_array[0];
}

void PythonAPICall::OutputList(Array<String> outputs) {
  if (outputs.empty()) {
    return;
  }
  if (outputs.size() == 1) {
    this->output_ = outputs[0] + ",";
    return;
  }
  std::ostringstream os;
  os << outputs[0];
  for (int i = 1, n = outputs.size(); i < n; ++i) {
    os << ", " << outputs[i];
  }
  this->output_ = os.str();
}

String PythonAPICall::Str() const {
  std::ostringstream os;
  if (output_.defined()) {
    os << output_.value() << " = ";
  }
  os << "sch." << method_name_ << '(';
  int n = args_.size();
  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    if (arg_names_[i].empty()) {
      os << args_[i];
    } else {
      os << arg_names_[i] << '=' << args_[i];
    }
  }
  os << ')';
  return os.str();
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_INSTRUCTION_TRAITS_H_
