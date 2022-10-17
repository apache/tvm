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
#ifndef TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_
#define TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_

#include <tvm/meta_schedule/schedule_rule.h>
#include <tvm/tir/schedule/schedule.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "../../support/array.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Configuration of data reuse type:
 * 0) kNoReuse: no reuse is allowed, then no cache_read/write is performed.
 * 1) kMayReuse: reuse is allowed, but no reuse is explored.
 * 2) kMustReuse: reuse is allowed and no reuse is not explored.
 */
enum class ReuseType : int32_t {
  kNoReuse = 0,
  kMayReuse = 1,
  kMustReuse = 2,
};

/*!
 * \brief Converts a string to ReuseType.
 * \param str The string to be converted.
 * \return The converted ReuseType.
 */
inline ReuseType Str2ReuseType(const String& str) {
  if (str == "no") {
    return ReuseType::kNoReuse;
  } else if (str == "may") {
    return ReuseType::kMayReuse;
  } else if (str == "must") {
    return ReuseType::kMustReuse;
  } else {
    LOG(FATAL) << "ValueError: Unknown ReuseType: " << str;
    throw;
  }
}

/*! \brief Configuration of data reuse patterns */
struct ReuseConfig {
  /*! \brief Type of data reuse: no-reuse, may-reuse or must-reuse */
  ReuseType req;
  /*! \brief Which levels are caching stage inserted at */
  std::vector<int> levels;
  /*! \brief The storage scope */
  String scope;

  /*! \brief Default constructor: no data reuse */
  ReuseConfig() : req(ReuseType::kNoReuse) {}

  /*! \brief Construct from a configuration dictionary */
  explicit ReuseConfig(const Map<String, ObjectRef>& config)
      : req(Str2ReuseType(Downcast<String>(config.at("req")))),
        levels(support::AsVector<Integer, int>(Downcast<Array<Integer>>(config.at("levels")))),
        scope(Downcast<String>(config.at("scope"))) {
    ICHECK_EQ(config.size(), 3);
  }
};

// Forware declaration
class State;

/*! \brief The state of auto scheduling for the multi-level tiling rule */
class StateNode : public Object {
 public:
  /*! \brief The schedule to date */
  tir::Schedule sch;
  /*! \brief The block to be tiled */
  tir::BlockRV block_rv;
  /*! \brief The loop tiles */
  Array<Array<tir::LoopRV>> tiles;
  /*! \brief The mapping from buffer index to read cache block. */
  std::unordered_map<int, tir::BlockRV> read_reuse;
  /*! \brief The mapping from buffer index to write cache block. */
  std::unordered_map<int, tir::BlockRV> write_reuse;

  /*!
   * \brief Create a copy of the state. The underlying schedule is copied. Schedule rules that
   * produce multiple states should use this method to create new states.
   */
  virtual State Copy() const;

  static constexpr const char* _type_key = "meta_schedule.State";
  TVM_DECLARE_BASE_OBJECT_INFO(StateNode, Object);
};

/*! \brief Managed reference to StateNode */
class State : public ObjectRef {
 public:
  /*! \brief Default constructor */
  explicit State(tir::Schedule sch, tir::BlockRV block_rv, Array<Array<tir::LoopRV>> tiles = {});
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(State, ObjectRef, StateNode);
};

/*!
 * \brief Helper to apply a sub-rule to a list of auto scheduling states
 * \tparam FLambda The type of the sub-rule functor
 * \param states The list of states to be applied
 * \return The list of states after applying the sub-rule
 */
template <class FLambda>
std::vector<State> SubRule(std::vector<State> states, FLambda sub_rule) {
  std::vector<State> results;
  for (auto&& state : states) {
    std::vector<State> next = sub_rule(std::move(state));
    results.insert(results.end(),                          //
                   std::make_move_iterator(next.begin()),  //
                   std::make_move_iterator(next.end()));
  }
  return results;
}

/*!
 * \brief The mega rule: multi-level tiling with data reuse
 */
class MultiLevelTilingNode : public ScheduleRuleNode {
 public:
  virtual ~MultiLevelTilingNode() = default;

  // SubRule 1. add write cache
  std::vector<State> AddWriteReuse(State state) const;
  // SubRule 2. tile the loop nest
  std::vector<State> TileLoopNest(State state) const;
  // SubRule 3. add read cache
  std::vector<State> AddReadReuse(State state) const;

  // Do nothing; Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final;

  // Entry of the mega rule; Inherited from ScheduleRuleNode
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) override;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const override;

 protected:
  virtual std::vector<State> ApplySubRules(std::vector<State> states);

  virtual Array<tir::LoopRV> SplitLoop(const tir::Schedule& sch, tir::BlockRV block,
                                       tir::LoopRV loop, int n_tiles) const;

  // Annotate a block to use cooperative fetching
  void AnnotateCooperativeFetching(tir::Schedule* sch, const tir::BlockRV& block) const;

 public:
  /*!
   * \brief The tiling structure. Recommended:
   * - 'SSRSRS' on CPU
   * - 'SSSRRSRS' on GPU
   */
  String structure;
  /*! \brief For each level of tiles, which thread axis it is bound to */
  Array<String> tile_binds;
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;
  /*! \brief The length of vector lane in vectorized cooperative fetching */
  std::vector<int> vector_load_lens;
  /*! \brief Data reuse configuration for reading */
  ReuseConfig reuse_read_;
  /*! \brief Data reuse configuration for writing */
  ReuseConfig reuse_write_;
  /*! \brief The indices of spatial tiles in `structure` */
  std::vector<int> s_indices_;
  /*! \brief The indices of reduction tiles in `structure` */
  std::vector<int> r_indices_;
  /*! \brief The size of the thread warp */
  int thread_warp_size_;
  /*! \brief The maximum number of threads to be used size of a thread warp */
  int max_threads_per_block_;
  /*! \brief The logging function */
  PackedFunc logger;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("structure", &structure);
    v->Visit("tile_binds", &tile_binds);
    v->Visit("max_innermost_factor", &max_innermost_factor);
    // `vector_load_lens` is not visited
    // `reuse_read_` is not visited
    // `reuse_write_` is not visited
    // `s_indices_` is not visited
    // `r_indices_` is not visited
    // `thread_warp_size_` is not visited
    // `max_threads_per_block` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTiling";
  TVM_DECLARE_BASE_OBJECT_INFO(MultiLevelTilingNode, ScheduleRuleNode);
};

template <typename NodeType>
ObjectPtr<NodeType> MultiLevelTilingInitCommon(String structure, Optional<Array<String>> tile_binds,
                                               Optional<Integer> max_innermost_factor,
                                               Optional<Array<Integer>> vector_load_lens,
                                               Optional<Map<String, ObjectRef>> reuse_read,
                                               Optional<Map<String, ObjectRef>> reuse_write) {
  ObjectPtr<NodeType> n = make_object<NodeType>();
  n->structure = structure;
  n->tile_binds = tile_binds.value_or({});
  n->max_innermost_factor = max_innermost_factor.value_or(Integer(-1))->value;
  n->vector_load_lens = vector_load_lens.defined()
                            ? support::AsVector<Integer, int>(vector_load_lens.value())
                            : std::vector<int>();
  n->reuse_read_ = reuse_read.defined() ? ReuseConfig(reuse_read.value()) : ReuseConfig();
  n->reuse_write_ = reuse_write.defined() ? ReuseConfig(reuse_write.value()) : ReuseConfig();
  for (int i = 0, len = structure.size(); i < len; ++i) {
    char c = structure.data()[i];
    if (c == 'S') {
      n->s_indices_.push_back(i);
    } else if (c == 'R') {
      n->r_indices_.push_back(i);
    } else {
      LOG(FATAL) << "ValueError: Invalid tiling structure: " << structure;
    }
  }
  n->thread_warp_size_ = -1;
  n->max_threads_per_block_ = -1;
  return n;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SCHEDULE_RULE_MULTI_LEVEL_TILING_H_
