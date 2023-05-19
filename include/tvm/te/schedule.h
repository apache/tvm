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
 * \file tvm/te/schedule.h
 * \brief Define a schedule.
 */
// Acknowledgement: Many schedule primitives originate from Halide and Loopy.
#ifndef TVM_TE_SCHEDULE_H_
#define TVM_TE_SCHEDULE_H_

#include <tvm/support/with.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/index_map.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace te {
// Node container for Stage
class StageNode;
// Node container for Schedule
class ScheduleNode;
// Node container for IterVarRelation
class IterVarRelationNode;
// Attribute of itervar.
class IterVarAttrNode;

/*! \brief the attachment type */
enum AttachType : int {
  kGroupRoot = 1,
  kInline = 2,
  kInlinedAlready = 3,
  kScope = 4,
  kScanUpdate = 5
};

/*! \brief Stage, contains scheduling for a stage of computation. */
class Stage : public ObjectRef {
 public:
  Stage() {}
  explicit Stage(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief create a new schedule for op.
   * \param op The operator in the schedule
   * \param sch The schedule which current stage belongs to
   */
  explicit Stage(Operation op, const ScheduleNode* sch);
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const StageNode* operator->() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline StageNode* operator->();
  /*!
   * \brief set the memory scope of the stage
   * \param scope The memory scope.
   */
  TVM_DLL Stage& set_scope(std::string scope);  // NOLINT(*)
  /*!
   * \brief specify the schedule to be computed at the parent schedule's scope.
   * \param parent The parent schedule.
   * \param scope The iteration point to carry the schedule.
   * \return reference to self.
   */
  TVM_DLL Stage& compute_at(Stage parent, IterVar scope);  // NOLINT(*)
  /*!
   * \brief Compute the function inline.
   * \return reference to self.
   */
  TVM_DLL Stage& compute_inline();  // NOLINT(*)
  /*!
   * \brief Compute the function at group root.
   * \return reference to self.
   */
  TVM_DLL Stage& compute_root();  // NOLINT(*)
  /*!
   * \brief Bind the IterVar to thread index.
   *
   * \param ivar The IterVar to be bound.
   * \param thread_ivar The thread axis to be bound.
   * \return reference to self.
   */
  TVM_DLL Stage& bind(IterVar ivar, IterVar thread_ivar);
  /*!
   * \brief Set the predicate to determine whether a store to the array should be performed.
   *  Use this when there are multiple threads performing the same store and we only
   *  need one of them to do the store.
   *
   * \note This is a dangerous scheduling primitive that can change behavior of program.
   *    Only do when we are certain that thare are duplicated stores.
   * \param predicate The condition to be checked.
   * \return reference to self.
   */
  TVM_DLL Stage& set_store_predicate(PrimExpr predicate);
  /*!
   * \brief Specify environment threads that launched around the group's scope.
   *  This can only be used in group stage.
   * \param threads The threads to be launched around the scope.
   * \note Each thread can only appear in one env_threads.
   *    This is a beta feature.
   * \return reference to self.
   */
  TVM_DLL Stage& env_threads(Array<IterVar> threads);
  /*!
   * \brief Split the parent by factor, generate
   * \param parent The parent iteration domain.
   * \param factor The split factor of the loop.
   * \param p_outer The result outer domain
   * \param p_inner The result inner domain.
   * \return reference to self.
   */
  TVM_DLL Stage& split(IterVar parent, PrimExpr factor, IterVar* p_outer,
                       IterVar* p_inner);  // NOLINT(*)
  /*!
   * \brief Split the iteration with given number of parts.
   *
   * \param parent The parent domain.
   * \param nparts The number of parts in the outer domain.
   * \param p_outer The result outer domain.
   * \param p_inner The result inner domain.
   * \return reference to self.
   */
  TVM_DLL Stage& split_by_nparts(IterVar parent, PrimExpr nparts, IterVar* p_outer,
                                 IterVar* p_inner);  // NOLINT(*)
  /*!
   * \brief Fuse the inner outer domain to the target
   * \param outer The outer domain to be fused.
   * \param inner The inner domain to be fused
   * \param p_target The result target domain.
   * \return reference to self.
   */
  TVM_DLL Stage& fuse(IterVar outer, IterVar inner, IterVar* p_target);  // NOLINT(*)
  /*!
   * \brief Fuse all the axes together into a single axis.
   *
   * \param axes All the axes to be fused.
   * \param p_target The result target domain.
   *
   * \note axes can be an empty array,
   *       in that case, a singleton IterVar is created and
   *       inserted to the outermost loop.
   *       The fuse of empty array is used to support zero-dimension tensors.
   *
   * \return reference to self.
   */
  TVM_DLL Stage& fuse(const Array<IterVar>& axes, IterVar* p_target);  // NOLINT(*)
  /*!
   * \brief Reorder the iteration
   * \param order The order of iteration variable.
   * \return reference to self.
   */
  TVM_DLL Stage& reorder(const Array<IterVar>& order);  // NOLINT(*)
  /*!
   * \brief Perform tiling on two dimensions
   *  The final loop order from outmost to inner most are
   *  [x_outer, y_outer, x_inner, y_inner]
   *
   * \param x_parent The original x dimension
   * \param y_parent The original y dimension
   * \param x_factor The stride factor on x axis
   * \param y_factor The stride factor on y axis
   * \param p_x_outer Outer axis of x dimension
   * \param p_y_outer Outer axis of y dimension
   * \param p_x_inner Inner axis of x dimension
   * \param p_y_inner Inner axis of y dimension
   * \return reference to self.
   */
  TVM_DLL Stage& tile(IterVar x_parent, IterVar y_parent,  // NOLINT(*)
                      PrimExpr x_factor, PrimExpr y_factor, IterVar* p_x_outer, IterVar* p_y_outer,
                      IterVar* p_x_inner, IterVar* p_y_inner);
  /*!
   * \brief Vectorize iteration.
   * \param var The axis to be vectorized.
   * \return reference to self.
   */
  TVM_DLL Stage& vectorize(IterVar var);  // NOLINT(*)
  /*!
   * \brief Replace computation of the current stage by tensor intrinsic f.
   * \param var The axis marks beginning of tensorization.
   *  Every operations inside the axis(include axis itself is tensorized).
   * \param f The Tensor compute intrinsics.
   * \return reference to self.
   */
  TVM_DLL Stage& tensorize(IterVar var, TensorIntrin f);  // NOLINT(*)
  /*!
   * \brief Unroll iteration.
   * \param var The axis to be unrolled.
   * \return reference to self.
   */
  TVM_DLL Stage& unroll(IterVar var);  // NOLINT(*)
  /*!
   * \brief Parallelize iteration.
   * \param var The axis to be parallelized.
   * \return reference to self.
   */
  TVM_DLL Stage& parallel(IterVar var);  // NOLINT(*)
  /*!
   * \brief Annotate the iteration with pragma
   *
   * \param var The axis to be parallelized.
   * \param pragma_type The pragma type.
   * \param pragma_value The pragma value
   *
   * \return reference to self.
   */
  TVM_DLL Stage& pragma(IterVar var, const std::string& pragma_type,
                        const PrimExpr& pragma_value = PrimExpr());  // NOLINT(*)
  /*!
   * \brief Fetch data in advance.
   * \param domain the tensor to be prefetched
   * \param var the iteration point at which to apply prefetching
   * \param offset the number of iterations be to fetched in advance
   * \return reference to self
   */
  TVM_DLL Stage& prefetch(const Tensor& domain, IterVar var, PrimExpr offset);  // NOLINT(*)
  /*!
   * \brief Set alignment requirement for specific dimension.
   *
   *  Such that stride[axis] == k * factor + offset for some k.
   *
   * \param axis The dimension to be specified for alignment.
   * \param factor The factor multiple of alignment
   * \param offset The required offset factor.
   * \return reference to self
   */
  TVM_DLL Stage& storage_align(IterVar axis, int factor, int offset);  // NOLINT(*)
  /*!
   * \brief Compute current stage with double buffering.
   * \return reference to self.
   */
  TVM_DLL Stage& double_buffer();  // NOLINT(*)
  /*!
   * \brief Compute current stage with rolling buffering.
   * \return reference to self.
   */
  TVM_DLL Stage& rolling_buffer();  // NOLINT(*)
  /*!
   * \brief Defines a layout transformation to be applied to the buffer.
   *
   * The map from initial_index to final_index must be an
   * invertible affine transformation.
   *
   * \param initial_indices An array of variables to represent a
   * value's location in the tensor, using the pre-transformation
   * layout.  These variables are used as binding occurrences to
   * represent the initial indices when applying the initial->final
   * mapping, and should not occur elsewhere in the
   * Schedule. (i.e. Pass in newly constructed variables, not the
   * initial IterVar::var)
   *
   * \param final_indices An array of expressions, giving the
   * value's location in the tensor, using the post-transformation layout.
   * Expressions should be in terms of the variables given in
   * initial_indices.
   *
   * \param out_iter_vars An optional output location for the updated
   * loop iteration variables.
   *
   * \return reference to self
   */
  TVM_DLL Stage& transform_layout(const Array<Var>& initial_indices,
                                  const Array<PrimExpr>& final_indices,
                                  Array<IterVar>* out_iter_vars = nullptr);
  /*! \brief Defines separators between groups of axes.
   *
   * Used to define `BufferNode::axis_separators`, which has
   * additional details.
   *
   * \param axis_separators A list of axis separators.
   */
  TVM_DLL Stage& set_axis_separators(const Array<IntImm>& axis_separators);
  /*!
   * \brief whether the stage has been scheduled.
   * \return whether the stage has been scheduled.
   */
  bool is_scheduled() const;
  /*!
   * \brief Get attachment spec of current stage.
   *  If the stage compute at Group root, this function
   *  will traverse the group function to get the
   *  final spec from the group.
   * \return A stage representing the attach spec of the group.
   */
  Stage GetAttachSpec() const;
  // declare container type
  using ContainerType = StageNode;
};

/*!
 * \brief Global schedule container
 *  For operations and all the operations they depend on.
 *  The schedule per Operation is named as stage.
 */
class Schedule : public ObjectRef {
 public:
  Schedule() {}
  explicit Schedule(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief Create a schedule for array of ops(and their dependencies).
   * \param ops The ops to be scheduled.
   */
  TVM_DLL explicit Schedule(Array<Operation> ops);
  /*!
   * \brief Get a copy of current schedule.
   * \return The copied schedule.
   */
  Schedule copy() const;
  /*!
   * \brief Get the stage corresponds to the op
   * \param op The operation.
   */
  TVM_DLL Stage operator[](const Operation& op);
  /*!
   * \brief Short hand for getting the stage of tensor's operation.
   * \param tensor The tensor
   * \return The stage corresponding to the tensor's op
   */
  TVM_DLL Stage operator[](const Tensor& tensor) { return this->operator[](tensor->op); }
  /*!
   * \brief Create a new stage group for all intermediate
   *  operations between inputs and outputs.
   *
   * \param outputs The output boundary of the group.
   * \param inputs The input boundary of the group.
   * \param include_inputs Whether include inputs if they are reachable from outputs.
   * \return The new grouped stage.
   */
  TVM_DLL Stage create_group(const Array<Tensor>& outputs, const Array<Tensor>& inputs,
                             bool include_inputs = false);
  /*!
   * \brief create a cache read of original tensor for readers.
   *  This will mutate the body of the readers.
   *  A new stage will be created for the tensor.
   * \param tensor The tensor cached.
   * \param scope The scope of the cache.
   * \param readers The readers to redirect to the tensor.
   * \return The created tensor.
   */
  TVM_DLL Tensor cache_read(const Tensor& tensor, const std::string& scope,
                            const Array<Operation>& readers);
  /*!
   * \brief Create a cache write tensor for producing tensor.
   *  The tensor will take over body of original tensor op.
   *
   *  This function can be used to do data layout transformation.
   *  If there is a split/fuse/reorder on the data parallel axis of tensor
   *  before cache_write is called. The intermediate cache stores
   *  the data in the layout as the iteration order of leave axis.
   *  The data will be transformed back to the original layout in the original tensor.
   *  User can further call compute_inline to inline the original layout and keep
   *  the data stored in the transformed layout.
   *
   * \param tensor The tensors to be produced.
   * \param scope The scope of the storage.
   * \return The created tensor.
   */
  TVM_DLL Array<Tensor> cache_write(const Array<Tensor>& tensor, const std::string& scope);
  /*!
   * \brief Create a cache write tensor for producing tensor.
   *  The tensor will take over body of original tensor op.
   *
   *  This function can be used to do data layout transformation.
   *  If there is a split/fuse/reorder on the data parallel axis of tensor
   *  before cache_write is called. The intermediate cache stores
   *  the data in the layout as the iteration order of leave axis.
   *  The data will be transformed back to the original layout in the original tensor.
   *  User can further call compute_inline to inline the original layout and keep
   *  the data stored in the transformed layout.
   *
   * \param tensor The tensor to be produced.
   * \param scope The scope of the storage.
   * \return The created tensor.
   */
  TVM_DLL Tensor cache_write(const Tensor& tensor, const std::string& scope);
  /*!
   * \brief Factor a reduction axis in tensor's schedule to be an explicit axis.
   * This will create a new stage that generated the new tensor with axis
   * as the first dimension. The tensor's body will be rewritten as a reduction
   * over the factored tensor.
   *
   *  P. Suriana, A. Adams and S. Kamil. Parallel associative reductions in halide. CGO'17
   *
   * \param tensor The tensor to be factored.
   * \param axis The reduction axis in tensor's schedule to be factored.
   * \param factor_axis The position where the new axis is placed.
   * \return The created factored tensors.
   */
  TVM_DLL Array<Tensor> rfactor(const Tensor& tensor, const IterVar& axis, int factor_axis = 0);
  /*!
   * \brief Normalize the schedule.
   *  This is needed before bound inference.
   *  Insert necessary RebaseNode to make sure all leaf_iter_vars
   *  are in form [0, extent)
   *
   * \return A normalized schedule, can be same as current one.
   */
  Schedule normalize();

  /*!
   * \brief Normalize the schedule for feature extraction in auto-scheduler.
   * This is similar to `Schedule::normalize`, but we do aggressive simplification
   * to the TE compute with const_matrix=True for faster compilation and feature extraction.
   * The resulted schedule may be wrong, but it is good enough for feature extraction
   * purposes.
   *
   * \return A normalized schedule, can be same as current one.
   */
  Schedule normalize_for_feature_extraction();

  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const ScheduleNode* operator->() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline ScheduleNode* operator->();
  // declare container type
  using ContainerType = ScheduleNode;
};

/*!
 * \brief Context helper to collect debug information of Schedule.
 *
 *  Attach With<ScheduleContext>(schedule_instance, primitive_name)
 *  inside function body of schedule primitives to collect the
 *  snapshot of schedule status and corresponding primitive name
 */
class ScheduleContext {
 private:
  friend class With<ScheduleContext>;
  ScheduleContext(const ScheduleNode* sch_node, String current_primitive_name);
  void EnterWithScope();
  void ExitWithScope();

  /*! \brief Schedule instance to store information for debug */
  Schedule sch_;
  /*! \brief String representing which primitive has been applied to sch_ */
  String current_primitive_name_;
};

/*!
 * \brief The schedule relation between IterVars
 *  can be Split, Fuse.
 */
class IterVarRelation : public ObjectRef {
 public:
  IterVarRelation() {}
  explicit IterVarRelation(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IterVarRelationNode* operator->() const;
};

/*!
 * \brief Additional scheduable attributes about IterVar.
 */
class IterVarAttr : public ObjectRef {
 public:
  IterVarAttr() {}
  explicit IterVarAttr(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IterVarAttrNode* operator->() const;
};

/*!
 * \brief represents a stage.
 *
 *  relations form a Directed acylic hypergraph in bipartite manner.
 *  With each node is represented by a IterVar,
 *  and each hyper-edge is represented by a IterVarRelation.
 *  The relations connects the IterVars in the graph.
 *
 *  Besides typical stage that corresponds to operations.
 *  There is also group stage, which groups stages together.
 *  Each stage's group(given by group) represent an constraint,
 *  the stage can only be attached to stages within the group.
 *
 *  The group stage node can be attached to IterVars as in normal stage.
 */
class StageNode : public Object {
 public:
  /*!
   * \brief The operation of stage, can be different from original op.
   *  If it is null, then this stage is a group stage.
   */
  Operation op;
  /*!
   * \brief The original operator.
   *  The op field can change during schedule to alternate the dataflow,
   *  while origin_op remains fixed.
   */
  Operation origin_op;
  /*! \brief All the nodes in the iter var
   *
   * Each element of all_iter_vars represents an iteration variable
   * that may appear within this stage's computation.  Any element
   * of `all_iter_vars` that is in `leaf_iter_vars` represents a
   * variable that is directly defined and usable within the stage's
   * computation.  All other elements of `all_iter_vars` represent
   * variables whose value must be computed from the variables in
   * `leaf_iter_vars`.  (e.g. Support index k has been split by
   * ``ko, ki = s.split(k, factor=4)``.  ko and ki will appear in
   * `leaf_iter_vars`, while k will not, and must be computed as
   * `4*ko + ki`.
   */
  Array<IterVar> all_iter_vars;
  /*! \brief The current active leaf iter vars in the stage.
   *
   * Each element of leaf_iter_vars will either be replaced with the
   * bound index (e.g. threadIdx.x), or will be expanded into a loop
   * over the variable's extent.  `leaf_iter_vars` is a subset of
   * `all_iter_vars`.
   */
  Array<IterVar> leaf_iter_vars;
  /*!
   * \brief Specify threads to be launched at the stage.
   *  This is only valid for composite ops such as Scan.
   * \note Experimental primitive: used for thread persistence.
   */
  Array<IterVar> env_threads;
  /*!
   * \brief The predicate under which store can happen
   *  Use this when there can be duplicated threads doing the same store.
   * \note Experimental primitive: used by cross thread-reduction.
   */
  PrimExpr store_predicate;
  /*! \brief The relation bwteen of IterVars */
  Array<IterVarRelation> relations;
  /*! \brief additional attributes about iter var. */
  Map<IterVar, IterVarAttr> iter_var_attrs;
  /*! \brief The attachment type of the schedule */
  AttachType attach_type{kGroupRoot};
  /*! \brief The attach point of this schedule. */
  IterVar attach_ivar;
  /*! \brief The stage this node attaches to */
  Stage attach_stage;
  /*! \brief The schedule current stage is attached to */
  const ScheduleNode* attach_sch;
  /*! \brief The thread storage scope level of the stage */
  std::string scope;
  /*! \brief Whether this is an output stage */
  bool is_output{false};
  /*! \brief Whether apply double buffer optimization to this stage */
  bool double_buffer{false};
  /*! \brief Whether apply rolling buffer optimization to this stage */
  bool rolling_buffer{false};
  /*! \brief Layout transformations to be applied onto the stage's tensors. */
  Array<IndexMap> layout_transforms;
  /*! \brief List of axes after which to divide physical axes.
   *
   * Used to populate `BufferNode::axis_separators`, which has
   * additional details.
   */
  Array<IntImm> axis_separators;
  /*!
   * \brief The parent group of the current stage.
   *  The stage cannot be assigned to stages outside the group.
   */
  Stage group;
  /*! \brief Number of direct child stages, only used for group stage.*/
  int num_child_stages{0};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("origin_op", &origin_op);
    v->Visit("all_iter_vars", &all_iter_vars);
    v->Visit("leaf_iter_vars", &leaf_iter_vars);
    v->Visit("env_threads", &env_threads);
    v->Visit("relations", &relations);
    v->Visit("iter_var_attrs", &iter_var_attrs);
    v->Visit("attach_type", &attach_type);
    v->Visit("attach_ivar", &attach_ivar);
    v->Visit("attach_stage", &attach_stage);
    v->Visit("scope", &scope);
    v->Visit("is_output", &is_output);
    v->Visit("double_buffer", &double_buffer);
    v->Visit("layout_transforms", &layout_transforms);
    v->Visit("axis_separators", &axis_separators);
    v->Visit("group", &group);
    v->Visit("num_child_stages", &num_child_stages);
  }

  static constexpr const char* _type_key = "Stage";
  TVM_DECLARE_FINAL_OBJECT_INFO(StageNode, Object);
};

/*! \brief node container for schedule */
class ScheduleNode : public Object {
 public:
  /*! \brief The output operations in original data flow graph */
  Array<Operation> outputs;
  /*!
   * \brief list of all stages for ops.
   * The stages are sorted in dependency order.
   */
  Array<Stage> stages;
  /*!
   * \brief List of all stage groups.
   */
  Array<Stage> groups;
  /*! \brief map of original operation to the stages */
  Map<Operation, Stage> stage_map;
  /*!
   * \brief Internal stage map to map internal ops to stages.
   *  This is created on demand and can be invalidated.
   */
  std::unordered_map<const Object*, Stage> op2stage_cache_;
  /*!
   * \brief list of all transformed schedules
   * User can display the optimization strategy via TEDD step by step to check
   * the order and effect of primitives. Set "te.keep_schedule_record" in
   * PassContext config as true to enable recording.
   */
  Array<Schedule> schedule_record;
  /*!
   * \brief list of all applied primitive names.
   */
  Array<String> primitive_record;
  /*!
   * \brief Flag to keep schedule record or not.
   */
  Optional<Bool> keep_schedule_record;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("outputs", &outputs);
    v->Visit("stages", &stages);
    v->Visit("groups", &groups);
    v->Visit("stage_map", &stage_map);
    v->Visit("schedule_record", &schedule_record);
    v->Visit("primitive_record", &primitive_record);
    v->Visit("keep_schedule_record", &keep_schedule_record);
  }

  /*! \brief Initialize temp cache. */
  void InitCache();
  /*! \brief Invalidate temp cache. */
  void InvalidateCache();

  /*!
   * \brief Check if the schedule contains an Operation.
   * \param op The candidate Operation.
   * \return true if the schedule has the Operation. Otherwise, false.
   */
  TVM_DLL bool Contain(const Operation& op) const;

  /*!
   * \brief Check if the schedule contains a Tensor.
   * \param tensor The candidate tensor.
   * \return true if the schedule has the tensor. Otherwise, false.
   */
  TVM_DLL bool Contain(const Tensor& tensor) const { return Contain(tensor->op); }

  static constexpr const char* _type_key = "Schedule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleNode, Object);
};

/*!
 * \brief Create a schedule for array of ops(and their dependencies).
 * \param ops The ops to be scheduled.
 * \return sch The created Schedule.
 */
inline Schedule create_schedule(Array<Operation> ops) { return Schedule(ops); }

/*! \brief node container for IterVar attr */
class IterVarAttrNode : public Object {
 public:
  /*! \brief The iteration type. */
  IterVarType iter_type{kDataPar};
  /*! \brief The thread this iter Var binds, can be null */
  IterVar bind_thread;
  /*! \brief List of tensor to be prefetched in this loop */
  Array<Tensor> prefetch_data;
  /*! \brief The offset used in each prefetch */
  Array<PrimExpr> prefetch_offset;
  /*!
   * \brief Tensor intrinsic used in tensorization,
   *   when the axis is marked as Tensorized
   */
  TensorIntrin tensor_intrin;
  /*! \brief Alignment factor of buffer dimension */
  int dim_align_factor{0};
  /*! \brief Alignment offset of buffer dimension */
  int dim_align_offset{0};
  /*!
   * \brief Additional pragma keys, array of StringImm
   */
  Array<PrimExpr> pragma_keys;
  /*!
   * \brief Additional values of pragma, if any
   */
  Array<PrimExpr> pragma_values;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("iter_type", &iter_type);
    v->Visit("bind_thread", &bind_thread);
    v->Visit("prefetch_data", &prefetch_data);
    v->Visit("prefetch_offset", &prefetch_offset);
    v->Visit("tensor_intrin", &tensor_intrin);
    v->Visit("dim_align_factor", &dim_align_factor);
    v->Visit("dim_align_offset", &dim_align_offset);
    v->Visit("pragma_keys", &pragma_keys);
    v->Visit("pragma_values", &pragma_values);
  }

  static constexpr const char* _type_key = "IterVarAttr";
  TVM_DECLARE_FINAL_OBJECT_INFO(IterVarAttrNode, Object);
};

/*! \brief base node of iteration var */
class IterVarRelationNode : public Object {
 public:
  static constexpr const char* _type_key = "IterVarRelation";
  TVM_DECLARE_BASE_OBJECT_INFO(IterVarRelationNode, Object);
};

/*!
 * \brief Split the parent domain into product of
 *  outer and iter.
 */
class SplitNode : public IterVarRelationNode {
 public:
  /*! \brief The parent domain */
  IterVar parent;
  /*! \brief The outer domain */
  IterVar outer;
  /*! \brief The inner domain */
  IterVar inner;
  /*! \brief The split factor */
  PrimExpr factor;
  /*! \brief Number of parts, only factor or nparts can be given */
  PrimExpr nparts;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("parent", &parent);
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
    v->Visit("factor", &factor);
    v->Visit("nparts", &nparts);
  }

  static constexpr const char* _type_key = "Split";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitNode, IterVarRelationNode);
};

/*!
 * \brief Managed reference to SplitNode
 * \sa SplitNode
 */
class Split : public IterVarRelation {
 public:
  TVM_DLL Split(IterVar parent, IterVar outer, IterVar inner, PrimExpr factor, PrimExpr nparts);

  TVM_DEFINE_OBJECT_REF_METHODS(Split, IterVarRelation, SplitNode);
};

/*!
 * \brief Fuse two domains into one domain.
 */
class FuseNode : public IterVarRelationNode {
 public:
  /*! \brief The outer domain */
  IterVar outer;
  /*! \brief The inner domain */
  IterVar inner;
  /*! \brief The target domain */
  IterVar fused;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
    v->Visit("fused", &fused);
  }

  static constexpr const char* _type_key = "Fuse";
  TVM_DECLARE_FINAL_OBJECT_INFO(FuseNode, IterVarRelationNode);
};

/*!
 * \brief Managed reference to FuseNode
 * \sa FuseNode
 */
class Fuse : public IterVarRelation {
 public:
  TVM_DLL Fuse(IterVar outer, IterVar inner, IterVar fused);

  TVM_DEFINE_OBJECT_REF_METHODS(Fuse, IterVarRelation, FuseNode);
};

/*!
 * \brief Rebase the iteration to make min to be 0.
 *  This is useful to normalize the Schedule
 *  to make every leaf variable's min to be 0.
 */
class RebaseNode : public IterVarRelationNode {
 public:
  /*! \brief The parent domain */
  IterVar parent;
  /*! \brief The inner domain */
  IterVar rebased;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("parent", &parent);
    v->Visit("rebased", &rebased);
  }

  static constexpr const char* _type_key = "Rebase";
  TVM_DECLARE_FINAL_OBJECT_INFO(RebaseNode, IterVarRelationNode);
};

/*!
 * \brief Managed reference to RebaseNode
 * \sa RebaseNode
 */
class Rebase : public IterVarRelation {
 public:
  TVM_DLL Rebase(IterVar parent, IterVar rebased);

  TVM_DEFINE_OBJECT_REF_METHODS(Rebase, IterVarRelation, RebaseNode);
};

/*!
 * \brief Singleton iterator [0, 1)
 */
class SingletonNode : public IterVarRelationNode {
 public:
  /*! \brief The singleton iterator */
  IterVar iter;

  void VisitAttrs(AttrVisitor* v) { v->Visit("iter", &iter); }

  static constexpr const char* _type_key = "Singleton";
  TVM_DECLARE_FINAL_OBJECT_INFO(SingletonNode, IterVarRelationNode);
};

/*!
 * \brief Managed reference to SingletonNode
 * \sa SingletonNode
 */
class Singleton : public IterVarRelation {
 public:
  TVM_DLL explicit Singleton(IterVar iter);

  TVM_DEFINE_OBJECT_REF_METHODS(Singleton, IterVarRelation, SingletonNode);
};

/*!
 * \brief Transform iterator according to some arbitrary expression.
 */
class TransformNode : public IterVarRelationNode {
 public:
  /*! \brief The loop variables that were replaced by the transformation.
   *
   * Prior to applying a layout transformation, these represent the
   * loops to iterate over a tensor as it is being computed, following
   * a row-major traversal of the tensor's original shape in the
   * compute definition.
   */
  Array<IterVar> original_variables;

  /*! \brief The variables generated by the transformation.
   *
   * After to applying a layout transformation, these represent the
   * loops to iterate over a tensor as it is being computed, following
   * a row-major traversal of the transformed shape of the tensor.
   */
  Array<IterVar> transformed_variables;

  /*! \brief Map from the original variables to the transformed variables.
   *
   * Used to determine iterator ranges over the transformed variables.
   */
  IndexMap forward_transformation;

  /*! \brief Map from transformed variables to the original variables
   *
   * Used to rewrite expressions containing the original loop iterators
   * in terms of the transformed loop iterators.
   */
  IndexMap inverse_transformation;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("original_variables", &original_variables);
    v->Visit("transformed_variables", &transformed_variables);
    v->Visit("forward_transformation", &forward_transformation);
    v->Visit("inverse_transformation", &inverse_transformation);
  }

  static constexpr const char* _type_key = "Transform";
  TVM_DECLARE_FINAL_OBJECT_INFO(TransformNode, IterVarRelationNode);
};

class Transform : public IterVarRelation {
 public:
  TVM_DLL explicit Transform(Array<IterVar> original_variables,
                             Array<IterVar> transformed_variables, IndexMap forward_transformation,
                             IndexMap inverse_transformation);

  TVM_DEFINE_OBJECT_REF_METHODS(Transform, IterVarRelation, TransformNode);
};

/*! \brief Container for specialization conditions. */
class SpecializedConditionNode : public Object {
 public:
  /*!
   * \brief List of conditions in conjunctive joint form (CNF).
   *   Each condition should be a simple expression, e.g., n > 16, m % 8 == 0, etc.,
   *   where n, m are tvm::Var that represents a dimension in the tensor shape.
   */
  Array<PrimExpr> clauses;

  void VisitAttrs(AttrVisitor* v) { v->Visit("clauses", &clauses); }

  static constexpr const char* _type_key = "SpecializedCondition";
  TVM_DECLARE_FINAL_OBJECT_INFO(SpecializedConditionNode, Object);
};

/*!
 * \brief Specialized condition to enable op specialization
 */
class SpecializedCondition : public ObjectRef {
 public:
  /*!
   * \brief construct from conditions
   * \param conditions The clauses in the specialized condition.
   */
  TVM_DLL SpecializedCondition(Array<PrimExpr> conditions);  // NOLINT(*)

  /*!
   * \brief Get the current specialized condition.
   * \return the current specialized condition.
   */
  TVM_DLL static SpecializedCondition Current();

  TVM_DEFINE_OBJECT_REF_METHODS(SpecializedCondition, ObjectRef, SpecializedConditionNode);
  class Internal;

 private:
  // enable with syntax.
  friend class Internal;
  friend class With<SpecializedCondition>;
  /*! \brief Push a new specialized condition onto the thread local stack. */
  TVM_DLL void EnterWithScope();
  /*! \brief Pop a specialized condition off the thread local context stack. */
  TVM_DLL void ExitWithScope();
};

// implementations
inline const StageNode* Stage::operator->() const { return static_cast<const StageNode*>(get()); }
inline StageNode* Stage::operator->() { return static_cast<StageNode*>(get_mutable()); }

inline const ScheduleNode* Schedule::operator->() const {
  return static_cast<const ScheduleNode*>(get());
}
inline ScheduleNode* Schedule::operator->() { return static_cast<ScheduleNode*>(get_mutable()); }

inline const IterVarRelationNode* IterVarRelation::operator->() const {
  return static_cast<const IterVarRelationNode*>(get());
}

inline const IterVarAttrNode* IterVarAttr::operator->() const {
  return static_cast<const IterVarAttrNode*>(get());
}

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_SCHEDULE_H_
