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
#include "./utils.h"

namespace tvm {
namespace tir {

/**************** Constructors  ****************/

Trace::Trace() { data_ = make_object<TraceNode>(); }

Trace::Trace(Array<Instruction> insts, Map<Instruction, Any> decisions) {
  ObjectPtr<TraceNode> n = make_object<TraceNode>();
  n->insts = std::move(insts);
  n->decisions = std::move(decisions);
  data_ = std::move(n);
}

/**************** Utilities  ****************/

int GetNumValidInstructions(const Array<Instruction>& insts, bool remove_postproc) {
  if (!remove_postproc) {
    return insts.size();
  }
  int n_insts = 0;
  for (const Instruction& inst : insts) {
    if (!inst->kind->IsPostproc()) {
      ++n_insts;
    } else {
      break;
    }
  }
  return n_insts;
}

/**************** TranslateInputRVs  ****************/

Array<Any> TranslateInputRVs(const Array<Any>& inputs,
                             const std::unordered_map<const Object*, const Object*>& rv_map) {
  Array<Any> result;
  result.reserve(inputs.size());
  auto f_subst_with_rv_map = [&rv_map](const Var& var) -> Optional<PrimExpr> {
    auto it = rv_map.find(var.get());
    if (it == rv_map.end()) {
      return std::nullopt;
    }
    const Object* dst = it->second;
    ICHECK(dst->IsInstance<VarNode>())
        << "TypeError: Expect 'tir.Var', but gets: " << dst->GetTypeKey();
    return GetRef<Var>(static_cast<const VarNode*>(dst));
  };

  for (const Any& input : inputs) {
    if (input.type_index() < ffi::TypeIndex::kTVMFFIStaticObjectBegin) {
      // directly put back POD type
      result.push_back(input);
    } else if (auto expr = input.as<ffi::String>()) {
      result.push_back(expr.value());
    } else if (input.as<BlockRVNode>() ||  // RV: block
               input.as<LoopRVNode>() ||   // RV: loop
               input.as<VarNode>()) {      // RV: var
      auto it = rv_map.find(input.as<Object>());
      ICHECK(it != rv_map.end()) << "IndexError: Random variable doesn't exist: " << input;
      result.push_back(GetRef<ObjectRef>(it->second));
    } else if (auto expr = input.as<PrimExpr>()) {  // RV: Expr
      result.push_back(Substitute(expr.value(), f_subst_with_rv_map));
    } else if (auto index_map = input.as<IndexMap>()) {
      result.push_back(Substitute(index_map.value(), f_subst_with_rv_map));
    } else if (auto arr = input.as<Array<Any>>()) {
      // Recursively convert elements of the array into a new list of ObjectRefs.
      result.push_back(TranslateInputRVs(arr.value(), rv_map));
    } else {
      ICHECK(false) << "TypeError: Cannot recognize the type of an input random variable: "
                    << input.GetTypeKey();
      throw;
    }
  }
  return result;
}

// translate rv to string
Array<Any> TranslateInputRVs(
    const Array<Any>& inputs,
    const std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual>& rv_names) {
  Array<Any> results;
  results.reserve(inputs.size());
  for (const Any& input : inputs) {
    if (input == nullptr) {
      // Case 0. nullptr => None
      results.push_back(String("None"));
      continue;
    }
    if (input.type_index() < ffi::TypeIndex::kTVMFFIStaticObjectBegin) {
      // directly put back POD type
      results.push_back(input);
    } else if (input.as<BlockRVNode>() ||  // RV: block
               input.as<LoopRVNode>() ||   // RV: loop
               input.as<VarNode>()) {      // RV: var
      auto it = rv_names.find(input.cast<ObjectRef>());
      if (it != rv_names.end()) {
        // Case 1. BlockRV, LoopRV, VarRV
        results.push_back(it->second);
      } else {
        LOG(FATAL) << "IndexError: Random variable is not defined " << input;
        throw;
      }
    } else if (const auto* str_obj = input.as<ffi::StringObj>()) {
      // Case 2. string => "content"
      results.push_back(String('"' + std::string(str_obj->data) + '"'));
    } else if (input.as<IntImmNode>() || input.as<FloatImmNode>()) {
      // Case 3. integer or floating-point number
      results.push_back(input);
    } else if (input.as<ffi::ArrayObj>()) {
      // Case 4: array
      results.push_back(TranslateInputRVs(Downcast<Array<Any>>(Any(input)), rv_names));
    } else if (input.as<ffi::MapObj>()) {
      // Case 5: dict
      results.push_back(input);
    } else if (input.as<IndexMapNode>()) {
      // // Case 6: IndexMap
      IndexMap index_map = Downcast<IndexMap>(input);
      index_map = index_map.RenameVariables([&rv_names](const Var& var) -> Optional<String> {
        if (auto it = rv_names.find(var); it != rv_names.end()) {
          return it->second;
        }
        return std::nullopt;
      });
      results.push_back(index_map);
    } else {
      LOG(FATAL) << "TypeError: Stringifying is not supported for type: " << input.GetTypeKey();
      throw;
    }
  }
  return results;
}

Array<Any> TranslateInputRVs(const Array<Any>& inputs,
                             const std::unordered_map<std::string, ObjectRef>& named_rvs) {
  Array<Any> results;
  results.reserve(inputs.size());
  for (const Any& input : inputs) {
    if (input.type_index() < ffi::TypeIndex::kTVMFFIStaticObjectBegin) {
      // directly put back POD type
      results.push_back(input);
      continue;
    }
    // Case 3. integer or floating-point immediate
    if (input.as<IntImmNode>() || input.as<FloatImmNode>()) {
      results.push_back(input);
      continue;
    }
    // Case 4. array
    if (input.as<ffi::ArrayObj>()) {
      results.push_back(TranslateInputRVs(Downcast<Array<Any>>(input), named_rvs));
      continue;
    }
    // Case 5. dict
    if (input.as<ffi::MapObj>()) {
      results.push_back(input);
      continue;
    }
    const auto* str = input.as<ffi::StringObj>();
    CHECK(str) << "TypeError: Expect String, but gets: " << input.GetTypeKey();
    CHECK_GT(str->size, 0) << "ValueError: Empty string is not allowed in input names";
    const char* name = str->data;
    int64_t size = str->size;
    if (name[0] == '{' && name[size - 1] == '}') {
      Any obj = LoadJSON(name);
      // Case 6. IndexMap
      if (obj.as<IndexMapNode>()) {
        IndexMap index_map = Downcast<IndexMap>(obj);
        index_map = Substitute(index_map, [&named_rvs](const Var& var) -> Optional<PrimExpr> {
          auto it = named_rvs.find(var->name_hint);
          if (it != named_rvs.end()) {
            return Downcast<Var>(it->second);
          }
          return std::nullopt;
        });
        results.push_back(index_map);
        continue;
      } else {
        LOG(FATAL) << "TypeError: Unexpected object: " << obj.GetTypeKey();
        throw;
      }
    }
    // Case 2. string
    if (size >= 2 && name[0] == '"' && name[size - 1] == '"') {
      results.push_back(String(std::string(name + 1, size - 2)));
      continue;
    }
    // Case 0 & 1. None, BlockRV, LoopRV, VarRV
    auto it = named_rvs.find(name);
    CHECK(it != named_rvs.end()) << "ValueError: The random variable is not defined: " << name;
    results.push_back(it->second);
  }
  return results;
}

/**************** TranslateAddOutputRVs  ****************/

void TranslateAddOutputRVs(const Array<Any>& old_outputs, const Array<Any>& new_outputs,
                           std::unordered_map<const Object*, const Object*>* rv_map) {
  ICHECK_EQ(old_outputs.size(), new_outputs.size());
  int n = old_outputs.size();
  for (int i = 0; i < n; ++i) {
    const Object* old_rv = old_outputs[i].as<Object>();
    const Object* new_rv = new_outputs[i].as<Object>();
    ICHECK(old_rv != nullptr && new_rv != nullptr);
    (*rv_map)[old_rv] = new_rv;
  }
}

Array<String> TranslateAddOutputRVs(
    const Array<Any>& outputs,
    std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual>* rv_names) {
  Array<String> results;
  results.reserve(outputs.size());
  for (const Any& output : outputs) {
    int i = rv_names->size();
    ICHECK(!rv_names->count(output.cast<ObjectRef>()))
        << "ValueError: The random variable has been produced once: "
        << rv_names->at(output.cast<ObjectRef>());
    String result{ffi::ObjectPtr<ffi::StringObj>{nullptr}};
    if (output == nullptr) {
      result = "_";
    } else if (output.as<BlockRVNode>()) {
      result = "b" + std::to_string(i);
    } else if (output.as<LoopRVNode>()) {
      result = "l" + std::to_string(i);
    } else if (output.as<VarNode>()) {
      result = "v" + std::to_string(i);
    } else {
      LOG(FATAL) << "TypeError: Cannot recognize the type of the random variable: "
                 << output.GetTypeKey();
      throw;
    }
    results.push_back(result);
    rv_names->emplace(output.cast<ObjectRef>(), std::move(result));
  }
  return results;
}

void TranslateAddOutputRVs(const Array<String>& old_outputs, const Array<Any>& new_outputs,
                           std::unordered_map<std::string, ObjectRef>* named_rvs) {
  ICHECK_EQ(old_outputs.size(), new_outputs.size());
  int n = old_outputs.size();
  for (int i = 0; i < n; ++i) {
    named_rvs->emplace(Downcast<String>(old_outputs[i]), new_outputs[i].cast<ObjectRef>());
  }
}

/**************** Add/Remove/Get ****************/

Any TraceNode::GetDecision(const Instruction& inst) const {
  return this->decisions.Get(inst).value_or(Any{nullptr});
}

void TraceNode::Append(Instruction inst) { insts.push_back(std::move(inst)); }

void TraceNode::Append(Instruction inst, Any decision) {
  decisions.Set(inst, std::move(decision));
  insts.push_back(std::move(inst));
}

Optional<Instruction> TraceNode::Pop() {
  if (insts.empty()) {
    return std::nullopt;
  }
  Instruction inst = insts.back();
  insts.pop_back();
  if (decisions.count(inst)) {
    decisions.erase(inst);
  }
  return inst;
}

/**************** Interfacing with InstructionKind ****************/

void TraceNode::ApplyToSchedule(
    Schedule sch, bool remove_postproc,
    ffi::TypedFunction<Any(const Instruction& inst, const Array<Any>& inputs,  //
                           const Array<Any>& attrs,                            //
                           const Any& decision)>
        decision_provider) const {
  std::unordered_map<const Object*, const Object*> rv_map;
  for (const Instruction& inst : this->insts) {
    if (remove_postproc && inst->kind->IsPostproc()) {
      break;
    }
    Array<Any> inputs = TranslateInputRVs(inst->inputs, rv_map);
    Array<Any> attrs = inst->attrs;
    Any decision = this->GetDecision(inst);
    if (decision_provider != nullptr) {
      decision = decision_provider(inst, inputs, attrs, decision);
    }
    Array<Any> outputs = inst->kind->f_apply_to_schedule(sch, inputs, attrs, decision);
    TranslateAddOutputRVs(inst->outputs, outputs, &rv_map);
  }
}

ObjectRef TraceNode::AsJSON(bool remove_postproc) const {
  std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual> rv_names;
  Array<ObjectRef> json_insts;
  Array<ObjectRef> json_decisions;
  json_insts.reserve(this->insts.size());
  json_decisions.reserve(this->insts.size());

  int i = 0;
  for (const Instruction& inst : this->insts) {
    const InstructionKind& kind = inst->kind;
    if (remove_postproc && kind->IsPostproc()) {
      break;
    }
    json_insts.push_back(Array<ObjectRef>{
        /* 0: inst name */ kind->name,
        /* 1: inputs    */ TranslateInputRVs(inst->inputs, rv_names),
        /* 2: attrs     */ kind->f_attrs_as_json != nullptr ? kind->f_attrs_as_json(inst->attrs)
                                                            : ObjectRef(inst->attrs),
        /* 3: outputs   */ TranslateAddOutputRVs(inst->outputs, &rv_names),
    });
    if (auto decision = this->GetDecision(inst).cast<Optional<ObjectRef>>()) {
      json_decisions.push_back(Array<ObjectRef>{
          /* 0: index    */ Integer(i),
          /* 1: decision */ decision.value(),
      });
    }
    ++i;
  }
  return Array<ObjectRef>{
      /* 0: trace    */ std::move(json_insts),
      /* 1: decision */ std::move(json_decisions),
  };
}

Array<String> TraceNode::AsPython(bool remove_postproc) const {
  std::unordered_map<ObjectRef, String, ObjectPtrHash, ObjectPtrEqual> rv_names;
  Array<String> py_trace;
  py_trace.reserve(this->insts.size());
  for (const Instruction& inst : this->insts) {
    if (remove_postproc && inst->kind->IsPostproc()) {
      break;
    }
    Array<Any> attrs;
    attrs.reserve(inst->attrs.size());
    for (const Any& obj : inst->attrs) {
      if (const auto* str = obj.as<ffi::StringObj>()) {
        attrs.push_back(String('"' + std::string(str->data) + '"'));
      } else {
        attrs.push_back(obj);
      }
    }
    py_trace.push_back(
        inst->kind->f_as_python(/*inputs=*/TranslateInputRVs(inst->inputs, rv_names),
                                /*attrs=*/attrs,
                                /*decision=*/this->GetDecision(inst),
                                /*outputs=*/TranslateAddOutputRVs(inst->outputs, &rv_names)));
  }
  return py_trace;
}

void Trace::ApplyJSONToSchedule(ObjectRef json, Schedule sch) {
  Array<Any> json_insts{nullptr};
  Array<Any> json_decisions{nullptr};
  // Parse `json` into `json_insts` and `json_decisions`
  try {
    const ffi::ArrayObj* arr = json.as<ffi::ArrayObj>();
    ICHECK(arr && arr->size() == 2);
    const auto* arr0 = arr->at(0).as<ffi::ArrayObj>();
    const auto* arr1 = arr->at(1).as<ffi::ArrayObj>();
    ICHECK(arr0 && arr1);
    json_insts = GetRef<Array<Any>>(arr0);
    json_decisions = GetRef<Array<Any>>(arr1);
  } catch (const tvm::Error& e) {
    LOG(FATAL) << "ValueError: The json entry of a trace should contain two arrays, an array of "
                  "instructions and an array of decisions, but gets: "
               << json;
    throw;
  }
  // Parse `json_decisions`
  std::vector<Any> decisions(json_insts.size(), Any{nullptr});
  for (const Any& decision_entry : json_decisions) {
    int index = -1;
    Any decision{nullptr};
    try {
      const ffi::ArrayObj* arr = decision_entry.as<ffi::ArrayObj>();
      ICHECK(arr && arr->size() == 2);
      auto arr0 = arr->at(0).as<IntImm>();
      ICHECK(arr0);
      index = arr0.value()->value;
      decision = arr->at(1);
    } catch (const tvm::Error& e) {
      LOG(FATAL) << "ValueError: Each entry of a json decision should be a tuple [index, "
                    "decision], but gets: "
                 << decision_entry;
      throw;
    }
    decisions[index] = std::move(decision);
  }
  // Parse `json_insts`
  std::unordered_map<std::string, ObjectRef> named_rvs{{"None", ObjectRef{nullptr}}};
  int i = 0;
  for (const Any& inst_entry : json_insts) {
    InstructionKind kind{nullptr};
    Array<Any> inputs{nullptr};
    Array<Any> attrs{nullptr};
    Array<String> outputs{ObjectPtr<Object>{nullptr}};
    // Parse the entry
    try {
      const auto* arr = inst_entry.as<ffi::ArrayObj>();
      ICHECK(arr && arr->size() == 4);
      const auto* arr0 = arr->at(0).as<ffi::StringObj>();
      kind = InstructionKind::Get(arr0->data);
      inputs = arr->at(1).cast<Array<Any>>();
      attrs = arr->at(2).cast<Array<Any>>();
      outputs = arr->at(3).cast<Array<String>>();
    } catch (const tvm::Error& e) {
      LOG(FATAL) << "ValueError: Each entry of a json instruction should be a tuple [inst_name, "
                    "inputs, attrs, outputs], but gets: "
                 << inst_entry << "\nThe error is: " << e.what();
      throw;
    }
    // Parse inputs
    inputs = TranslateInputRVs(inputs, named_rvs);
    // Parse attrs
    if (kind->f_attrs_from_json != nullptr) {
      attrs = kind->f_attrs_from_json(attrs);
    }
    // Apply to the schedule
    Array<Any> new_outputs = kind->f_apply_to_schedule(sch, inputs, attrs, decisions[i]);
    // Parse outputs
    TranslateAddOutputRVs(outputs, new_outputs, &named_rvs);
    ++i;
  }
}

/**************** Creation ****************/

Trace TraceNode::WithDecision(Instruction inst, Any decision, bool remove_postproc) const {
  int n_insts = GetNumValidInstructions(this->insts, remove_postproc);
  Array<Instruction> new_insts =
      Array<Instruction>{this->insts.begin(), this->insts.begin() + n_insts};
  Map<Instruction, Any> new_decisions{this->decisions.begin(), this->decisions.end()};
  new_decisions.Set(std::move(inst), std::move(decision));
  return Trace(new_insts, new_decisions);
}

Trace TraceNode::Simplified(bool remove_postproc) const {
  int n_insts = GetNumValidInstructions(this->insts, remove_postproc);
  std::unordered_set<const Object*> used_rvs;
  std::vector<Instruction> new_insts;
  std::unordered_map<Instruction, Any, ObjectPtrHash, ObjectPtrEqual> new_decisions;
  new_insts.reserve(n_insts);
  new_decisions.reserve(this->decisions.size());
  for (int inst_idx = n_insts - 1; inst_idx >= 0; --inst_idx) {
    const Instruction& inst = this->insts[inst_idx];
    // Check if all the variables the instruction defined are dead
    // If so, and the instruction is pure, we can safely remove this instruction
    bool all_defs_dead = inst->kind->is_pure;
    if (all_defs_dead) {
      for (const Any& obj : inst->outputs) {
        if (auto* obj_ptr = obj.as<Object>()) {
          if (used_rvs.count(obj_ptr)) {
            all_defs_dead = false;
            break;
          }
        }
      }
    }
    // Remove this instruction
    if (all_defs_dead) {
      continue;
    }
    // Otherwise this instruction is not dead
    new_insts.push_back(inst);
    Any decision = this->GetDecision(inst);
    if (decision != nullptr) {
      new_decisions.emplace(inst, std::move(decision));
    }
    // Add its inputs as "used" ones
    for (const Any& obj : inst->inputs) {
      if (obj == nullptr) {
        continue;
      } else if (obj.as<BlockRVNode>() || obj.as<LoopRVNode>() || obj.as<VarNode>()) {
        used_rvs.insert(obj.as<Object>());
        continue;
      } else if (auto prim_expr = obj.as<PrimExpr>()) {
        PostOrderVisit(*prim_expr, [&used_rvs](const ObjectRef& obj) -> void {
          if (obj.as<VarNode>()) {
            used_rvs.insert(obj.get());
          }
        });
      }
    }
  }
  return Trace(Array<Instruction>(new_insts.rbegin(), new_insts.rend()),
               Map<Instruction, Any>(new_decisions));
}

/**************** Repr ****************/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TraceNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* self = obj.as<TraceNode>();
      ICHECK_NOTNULL(self);
      p->stream << "# from tvm import tir\n";
      p->stream << "def apply_trace(sch: tir.Schedule) -> None:\n";
      Array<String> repr = self->AsPython(/*remove_postproc=*/false);
      bool is_first = true;
      for (const String& line : repr) {
        if (is_first) {
          is_first = false;
        } else {
          p->stream << '\n';
        }
        p->stream << "  " << line;
      }
      if (is_first) {
        p->stream << "  pass";
      }
      p->stream << std::flush;
    });

/**************** Instruction Registration ****************/

struct EnterPostprocTraits : public UnpackedInstTraits<EnterPostprocTraits> {
  static constexpr const char* kName = "EnterPostproc";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch) { return sch->EnterPostproc(); }

  static String UnpackedAsPython(Array<String> outputs) {
    PythonAPICall py("enter_postproc");
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(EnterPostprocTraits);

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(TraceNode);
TVM_REGISTER_GLOBAL("tir.schedule.Trace")
    .set_body_typed([](Optional<Array<Instruction>> insts,
                       Optional<Map<Instruction, Any>> decisions) {
      return Trace(insts.value_or(Array<Instruction>()), decisions.value_or({}));
    });
TVM_REGISTER_GLOBAL("tir.schedule.TraceGetDecision").set_body_method(&TraceNode::GetDecision);
TVM_REGISTER_GLOBAL("tir.schedule.TraceAppend")
    .set_body_typed([](Trace self, Instruction inst, Optional<ObjectRef> decision) {
      if (decision.defined()) {
        return self->Append(inst, decision.value());
      } else {
        return self->Append(inst);
      }
    });
TVM_REGISTER_GLOBAL("tir.schedule.TracePop").set_body_method(&TraceNode::Pop);
TVM_REGISTER_GLOBAL("tir.schedule.TraceApplyToSchedule")
    .set_body_method(&TraceNode::ApplyToSchedule);
TVM_REGISTER_GLOBAL("tir.schedule.TraceAsJSON").set_body_method(&TraceNode::AsJSON);
TVM_REGISTER_GLOBAL("tir.schedule.TraceAsPython").set_body_method(&TraceNode::AsPython);
TVM_REGISTER_GLOBAL("tir.schedule.TraceWithDecision").set_body_method(&TraceNode::WithDecision);
TVM_REGISTER_GLOBAL("tir.schedule.TraceSimplified").set_body_method(&TraceNode::Simplified);
TVM_REGISTER_GLOBAL("tir.schedule.TraceApplyJSONToSchedule")
    .set_body_typed(Trace::ApplyJSONToSchedule);

}  // namespace tir
}  // namespace tvm
