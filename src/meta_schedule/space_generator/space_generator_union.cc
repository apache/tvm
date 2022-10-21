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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The union of design space generators. */
class SpaceGeneratorUnionNode : public SpaceGeneratorNode {
 public:
  /*! \brief The array of design space generators unioned, could be recursive. */
  Array<SpaceGenerator> space_generators;

  void VisitAttrs(tvm::AttrVisitor* v) {
    SpaceGeneratorNode::VisitAttrs(v);
    v->Visit("space_generators", &space_generators);
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    SpaceGeneratorNode::InitializeWithTuneContext(context);
    for (const SpaceGenerator& space_generator : space_generators) {
      space_generator->InitializeWithTuneContext(context);
    }
  }

  Array<tir::Schedule> GenerateDesignSpace(const IRModule& mod) final {
    Array<tir::Schedule> design_spaces;
    for (const SpaceGenerator& space_generator : space_generators) {
      // Generate partial design spaces from each design space generator.
      Array<tir::Schedule> partial = space_generator->GenerateDesignSpace(mod);
      // Merge the partial design spaces.
      design_spaces.insert(design_spaces.end(), partial.begin(), partial.end());
    }
    return design_spaces;
  }

  SpaceGenerator Clone() const final {
    ObjectPtr<SpaceGeneratorUnionNode> n = make_object<SpaceGeneratorUnionNode>(*this);
    n->space_generators = Array<SpaceGenerator>();
    for (const SpaceGenerator& space_generator : this->space_generators) {
      n->space_generators.push_back(space_generator->Clone());
    }
    CloneRules(this, n.get());
    return SpaceGenerator(n);
  }

  static constexpr const char* _type_key = "meta_schedule.SpaceGeneratorUnion";
  TVM_DECLARE_FINAL_OBJECT_INFO(SpaceGeneratorUnionNode, SpaceGeneratorNode);
};

/*!
 * \brief Create a design space generator as union of given design space generators.
 * \param space_generators Array of the design space generators to be unioned.
 * \return The design space generator created.
 */
SpaceGenerator SpaceGenerator::SpaceGeneratorUnion(Array<SpaceGenerator> space_generators,
                                                   Optional<Array<ScheduleRule>> sch_rules,
                                                   Optional<Array<Postproc>> postprocs,
                                                   Optional<Map<Mutator, FloatImm>> mutator_probs) {
  ObjectPtr<SpaceGeneratorUnionNode> n = make_object<SpaceGeneratorUnionNode>();
  n->sch_rules = std::move(sch_rules);
  n->postprocs = std::move(postprocs);
  n->mutator_probs = std::move(mutator_probs);
  n->space_generators = std::move(space_generators);
  return SpaceGenerator(n);
}

TVM_REGISTER_NODE_TYPE(SpaceGeneratorUnionNode);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorSpaceGeneratorUnion")
    .set_body_typed(SpaceGenerator::SpaceGeneratorUnion);

}  // namespace meta_schedule
}  // namespace tvm
