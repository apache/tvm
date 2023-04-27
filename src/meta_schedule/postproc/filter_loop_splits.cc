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

#include "tvm/tir/schedule/instruction.h"
#include "tvm/tir/op.h"

namespace tvm {
namespace meta_schedule {

/*! \brief Filters splitting loops according to filter conditions */
class FilterLoopSplitsNode : public PostprocNode {
 public:
  using FFilter = Postproc::FFilter;
  /*! \brief The packed function to the `filter` function.*/ 
  FFilter f_filter;
  /*! \brief The TODO.*/ 
  size_t max_continuous_error = 150;
  FFilter GetFilter() {  return f_filter; };
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch, const tir::Schedule& orig) final;

  Postproc Clone() const {
    ObjectPtr<FilterLoopSplitsNode> n = make_object<FilterLoopSplitsNode>(*this);
    return Postproc(n);
  }

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.FilterLoopSplits";
  TVM_DECLARE_FINAL_OBJECT_INFO(FilterLoopSplitsNode, PostprocNode);
};

bool FilterLoopSplitsNode::Apply(const tir::Schedule& sch, const tir::Schedule& orig) {
  // std::ostringstream stream;
  // auto sch_copy = sch->Copy();
  // tir::Trace trace = sch_copy->trace().value();
  // stream << "FilterLoopSplits::Apply begin insts.size() "  << trace->insts.size() << std::endl << std::flush;
  // stream << "FilterLoopSplits::Apply begin orig->trace() "  << orig->trace() << std::endl << std::flush;
  // stream << "FilterLoopSplits::Apply begin sch->trace() "  << sch->trace() << std::endl << std::flush;


  // static auto kind_get_child_blocks = tir::InstructionKind::Get("GetChildBlocks");
  // static auto kind_get_block = tir::InstructionKind::Get("GetBlock");
  
  // static auto kind_get_loops = tir::InstructionKind::Get("GetLoops");
  // static auto kind_sample_perfect_tile = tir::InstructionKind::Get("SamplePerfectTile");
  // static auto kind_split = tir::InstructionKind::Get("Split");

  // // Array<LoopRV> loops_orig = orig->GetLoops(block_rv_orig);

  // // for (int i = 0, n = loops_orig.size(); i < n; ++i) {
  // //   data.push_back({loops_orig[i], {}});
  // // }

  // std::vector<std::pair<tir::LoopRV, Array<Integer>>> data;
  // for(const tir::Instruction& inst : trace->insts){
  //   if (inst->kind->IsPostproc()) {
  //     break;
  //   }
  //   if (inst->kind.same_as(kind_split)) {
  //     stream << "inst " << inst << std::endl; 
  //     tir::LoopRV flooprv = Downcast<tir::LoopRV>(inst->inputs[0]);
  //     // std::cout << "inst loop " << this->GetSRef(inst->inputs[0]) << std::endl; 
  //     data.push_back({flooprv, {}});
  //     // auto find = std::find(inst->inputs.begin(), inst->inputs.end(), loop_rv); // TODO
  //     for(size_t i = 1; i < inst->inputs.size(); ++i) {
  //         CHECK(inst->inputs[i].defined()) << "ValueError: ICE";
  //         tir::ExprRV exprrv = Downcast<tir::ExprRV>(inst->inputs[i]);
  //         // std::cout << "loop: " << loop_rv << " inst->inputs " << inst->inputs << std::endl << std::flush; 
  //         auto newf = sch->Get(exprrv);
  //         stream << "loop: " << flooprv <<  " sref "  << orig->Get(flooprv) << " factor " << *tir::as_const_int(newf) << std::endl; 
  //         data.back().second.push_back(*tir::as_const_int(newf));
  //     }
  //   }
  // }

  // Map<tir::LoopRV, Array<Integer>> loop_factors;
  // for(auto& it: data) {
  //   Array<Integer> factors;
  //   for (auto i: it.second) {
  //     factors.push_back(i);
  //   }
  //   loop_factors.Set(it.first, factors);
  // }
  // std::cout << stream.str() << std::flush;
  
  // bool s = f_filter(orig, loop_factors);



  // // stream.clear();
  // // std::unordered_map<const Object*, const Object*> rv_map;
  // // size_t loop_it = 0;
  // // for(const tir::Instruction& inst : trace->insts){
  // //   stream << "inst " << inst << std::endl;  
  // //   if (inst->kind->IsPostproc()) {
  // //     break;
  // //   }
  // //   try {
  // //     Array<ObjectRef> inputs = tir::TranslateInputRVs(inst->inputs, rv_map);
  // //     Array<ObjectRef> attrs = inst->attrs;
  // //     stream << "Translate inputs " << inputs << std::endl;  
  // //     Optional<ObjectRef> decision = trace->GetDecision(inst);
  // //     // if (decision.defined()) {
  // //     //   auto d = decision.value();
  // //     //   auto darr = Downcast<Array<Integer>>(d);
  // //     //   stream << "darr " << darr << std::endl;  
  // //     //   stream << "darr " << darr[0] << std::endl;  
  // //     //   stream << "darr " << darr[1] << std::endl;  
  // //     //   stream << "darr " << darr[2] << std::endl;  
  // //     //   stream << "darr " << darr[3] << std::endl;  
  // //     // }
      
  // //     stream << "decision " << decision << std::endl;
  // //     Array<ObjectRef> outputs = inst->kind->f_apply_to_schedule(sch_copy, inputs, attrs, decision);
  // //     stream << "Translate outputs " << outputs << std::endl;  
  // //     tir::TranslateAddOutputRVs(inst->outputs, Array<ObjectRef>(outputs.begin(), outputs.begin() + inst->outputs.size()), &rv_map); // TODO HACK
  // //     // tir::TranslateAddOutputRVs(inst->outputs, outputs, &rv_map);
  // //   } catch(std::exception &e) {
  // //     stream << "FilterLoopSplits::Apply EXCEPTION end" << std::endl;
  // //     stream << e.what() << std::endl;
  // //     std::cout << stream.str() << std::flush;
  // //     throw;
  // //   }
  // //   if (inst->kind.same_as(kind_sample_perfect_tile)) {
  // //     stream << "SamplePerfectTile inst " << inst << std::endl;  
  // //     Optional<ObjectRef> decision = trace->GetDecision(inst);
  // //     if (decision.defined()) {
  // //       // if (inst->inputs[0].defined()){
  // //         auto loop_expr = Downcast<tir::LoopRV>(inst->inputs[0]);
  // //         stream << "loop: " << loop_expr << std::endl;  
  // //       // }
  // //       auto darr = Downcast<Array<Integer>>(decision.value());
  // //       stream << "decisions " << darr << std::endl; 
  // //       for(size_t i = 0; i < darr.size(); ++i) {
  // //         stream << "decision: " << darr[i] << std::endl;  
  // //       }
  // //       // TODO
  // //       // if (loop_it == 0 && !(*tir::as_const_int(darr[0]) == 1)) { std::cout << "filtered!!!!!!!!!!!!!!!" << std::endl; return false; } 
  // //       // if (loop_it == 1 && !(*tir::as_const_int(darr[0]) >= 2)) { std::cout << "filtered!!!!!!!!!!!!!!!" << std::endl; return false; } 
  // //       // if (loop_it == 2 && !(*tir::as_const_int(darr[0]) >= 2)) { std::cout << "filtered!!!!!!!!!!!!!!!" << std::endl; return false; } 
  // //       // if (loop_it == 3 && !(*tir::as_const_int(darr[0]) >= 2)) { std::cout << "filtered!!!!!!!!!!!!!!!" << std::endl; return false; } 
  // //       // if (loop_it == 4 && !(*tir::as_const_int(darr[0]) == 1)) { std::cout << "filtered!!!!!!!!!!!!!!!" << std::endl; return false; } 
  // //       // if (loop_it == 5 && !(*tir::as_const_int(darr[0]) == 1)) { std::cout << "filtered!!!!!!!!!!!!!!!" << std::endl; return false; } 
  // //     }
  // //     loop_it++;
  // //   }
  // //   if (inst->kind.same_as(kind_split)) {
  // //     stream << "Split inst " << inst << std::endl;  
  // //     stream << "Split attrs " << inst->attrs << std::endl;  
  // //     stream << "Split inputs " << inst->inputs << std::endl;  
  // //     stream << "Split inputs size " << inst->inputs.size() << std::endl; 
  // //     stream << "Split outputs size " << inst->outputs.size() << std::endl; 
  // //     // if (inst->inputs.size() == 4) {
  // //     if (inst->inputs[1].defined()) {
  // //       auto pexpr = Downcast<PrimExpr>(inst->inputs[1]);
  // //       stream << "Split inst->inputs[1] " << pexpr->Script() << std::endl;
  // //       const int64_t* as_int = tir::as_const_int(pexpr);
  // //       if (as_int) {
  // //         stream << "Split inst->inputs[1] as_int " << *as_int << std::endl;
  // //       } else {
  // //         stream << "Split inst->inputs[1] as_int FAILED" << std::endl;
  // //       }
  // //       // if (inst->inputs[2].defined())
  // //       //   Downcast<PrimExpr>(inst->inputs[2]).get()
  // //       // if (inst->inputs[3].defined())
  // //       //   Downcast<PrimExpr>(inst->inputs[3]).get()
  // //       // if (inst->inputs[4].defined())
  // //       //   Downcast<PrimExpr>(inst->inputs[4]).get()
  // //     }
  // //     // }
  // //   }
  // // }
  
  // std::cout << stream.str() << std::flush;
  // stream << "FilterLoopSplits::Apply filter " << s << std::endl;
  // stream << "FilterLoopSplits::Apply end" << std::endl;
  return true; // max_continuous_error , reset, filter,
}

Postproc Postproc::FilterLoopSplits(FilterLoopSplitsNode::FFilter f_filter) {
  std::cout << "Postproc::FilterLoopSplits" << std::endl << std::flush;
  ObjectPtr<FilterLoopSplitsNode> n = make_object<FilterLoopSplitsNode>();
  n->f_filter = std::move(f_filter);
  // n->max_continuous_error = std::move(max_continuous_error);
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(FilterLoopSplitsNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocFilterLoopSplits")
    .set_body_typed(Postproc::FilterLoopSplits);

}  // namespace meta_schedule
}  // namespace tvm
