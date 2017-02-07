/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.h
 * \brief Define a schedule.
 */
#ifndef TVM_SCHEDULE_H_
#define TVM_SCHEDULE_H_

#include <string>
#include "./base.h"
#include "./operation.h"

namespace tvm {

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
  kNone = 0,
  kRoot = 1,
  kInline = 2,
  kScope = 3
};

/*! \brief IterVar type */
enum IterVarType : int {
  kUnrolled = 1,
  kVectorized = 2
};

/*! \brief Stage, contains scheduling for a stage of computation. */
class Stage : public NodeRef {
 public:
  Stage() {}
  explicit Stage(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief create a new schedule for op.
   * \param op The operator in the schedule
   */
  explicit Stage(Operation op);
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
  Stage& set_scope(std::string scope);  // NOLINT(*)
  /*!
   * \brief specify the schedule to be computed at the parent schedule's scope.
   * \param parent The parent schedule.
   * \param scope The iteration point to carry the schedule.
   * \return reference to self.
   */
  Stage& compute_at(Stage parent, IterVar scope);   // NOLINT(*)
  /*!
   * \brief Compute the function inline, attach it at parent.
   * \return reference to self.
   */
  Stage& compute_inline();   // NOLINT(*)
  /*!
   * \brief Compute the function at root, attach it to its parent.
   * \return reference to self.
   */
  Stage& compute_root();  // NOLINT(*)
  /*!
   * \brief Split the parent by factor, generate
   * \param parent The parent iteration domain.
   * \param p_outer The result outer domain
   * \param p_inner The result inner domain.
   * \param factor The split factor of the loop.
   * \return reference to self.
   */
  Stage& split(IterVar parent, IterVar* p_outer, IterVar* p_inner, Expr factor);  // NOLINT(*)
  /*!
   * \brief Split the iteration with a given outer domain,
   *  the outer domain must have a thread-tag.
   *
   * \param parent The parent domain.
   * \param outer The outer domain to be spliited, must have a thread_tag.
   * \param p_inner The result inner domain.
   * \param factor Optional, the factor of the split,
   *  factor must be provided such that factor * outer.extent >= parent.extent.
   * \return reference to self.
   */
  Stage& split(IterVar parent, IterVar outer, IterVar* p_inner, Expr factor = Expr());   // NOLINT(*)
  /*!
   * \brief Fuse the inner outer domain to the target
   * \param inner The inner domain to be fused
   * \param outer The outer domain to be fused.
   * \param p_target The result target domain.
   * \return reference to self.
   */
  Stage& fuse(IterVar inner, IterVar outer, IterVar* p_target);  // NOLINT(*)
  /*!
   * \brief Reorder the iteration
   * \param order The order of iteration variable.
   * \return reference to self.
   */
  Stage& reorder(const Array<IterVar>& order);   // NOLINT(*)
  /*!
   * \brief Perform tiling on two dimensions
   *  The final loop order from outmost to inner most are
   *  [x_outer, y_outer, x_inner, y_inner]
   *
   * \param x_parent The original x dimension
   * \param y_parent The original y dimension
   * \param p_x_outer Outer axis of x dimension
   * \param p_y_outer Outer axis of y dimension
   * \param p_x_inner Inner axis of x dimension
   * \param p_y_inner Inner axis of y dimension
   * \param x_factor The stride factor on x axis
   * \param y_factor The stride factor on y axis
   * \return reference to self.
   */
  Stage& tile(IterVar x_parent, IterVar y_parent,   // NOLINT(*)
              IterVar* p_x_outer, IterVar* p_y_outer,
              IterVar* p_x_inner, IterVar* p_y_inner,
              Expr x_factor, Expr y_factor);
  /*!
   * \brief Vectorize iteration.
   * \param var The axis to be vectorized.
   * \return reference to self.
   */
  Stage& vectorize(IterVar var);   // NOLINT(*)
  /*!
   * \brief Unroll iteration.
   * \param var The axis to be vectorized.
   * \return reference to self.
   */
  Stage& unroll(IterVar var);   // NOLINT(*)
  /*!
   * \brief whether the stage has been scheduled.
   * \return whether the stage has been scheduled.
   */
  inline bool is_scheduled() const;
  // declare container type
  using ContainerType = StageNode;
};

/*!
 * \brief Global schedule container
 *  For operations and all the operations they depend on.
 *  The schedule per Operation is named as stage.
 */
class Schedule : public NodeRef {
 public:
  Schedule() {}
  explicit Schedule(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief construct schedule for array of ops(and their dependencies).
   * \param ops The ops to be scheduled.
   */
  explicit Schedule(Array<Operation> ops);
  /*!
   * \brief Get the stage corresponds to the op
   * \param op The operation.
   */
  Stage operator[](const Operation& op);
  /*!
   * \brief Short hand for getting the stage of tensor's operation.
   * \param tensor The tensor
   * \return The stage corresponding to the tensor's op
   */
  Stage operator[](const Tensor& tensor) {
    return this->operator[](tensor->op);
  }
  /*!
   * \brief Normalize the schedule.
   *  This is needed before bound inference.
   *  Insert necessary RebaseNode to make sure all leaf_iter_vars
   *  are in form [0, extent)
   *
   * \return A normalized schedule, can be same as current one.
   */
  void normalize();
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const ScheduleNode* operator->() const;
  // declare container type
  using ContainerType = ScheduleNode;
};

/*!
 * \brief The schedule relation between IterVars
 *  can be Split, Fuse.
 */
class IterVarRelation : public NodeRef {
 public:
  IterVarRelation() {}
  explicit IterVarRelation(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IterVarRelationNode* operator->() const;
};

/*!
 * \brief Additional scheduable attributes about IterVar.
 */
class IterVarAttr : public NodeRef {
 public:
  IterVarAttr() {}
  explicit IterVarAttr(IterVarType t);
  explicit IterVarAttr(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IterVarAttrNode* operator->() const;
};

// defintion of node containers
/*!
 * \brief represents the schedule of the tensor
 *
 *  A schedule is a Directed acylic hypergraph.
 *  With each node is represented by a IterVar,
 *  and each hyper-edge is represented by a IterVarRelation.
 *
 *  The relations can be Split/Fuse.
 *
 *  The current data structure stores the hyper graph in its
 *  bipartite representation.
 *
 *  The relations connects the IterVars in the graph.
 */
class StageNode : public Node {
 public:
  /*! \brief The operation to be scheduled */
  Operation op;
  /*! \brief The thread scope level of the stage */
  std::string scope;
  /*! \brief All the nodes in the iter var */
  Array<IterVar> all_iter_vars;
  /*!
   * \brief The current leafs in the schedule.
   *  Operations can only be performed in leaves.
   */
  Array<IterVar> leaf_iter_vars;
  /*! \brief The relation bwteen of IterVars */
  Array<IterVarRelation> relations;
  /*! \brief additional attributes about iter var. */
  Map<IterVar, IterVarAttr> iter_var_attrs;
  /*! \brief The attachment type of the schedule */
  AttachType attach_type{kNone};
  /*! \brief The attach point of this schedule. */
  IterVar attach_ivar;
  /*! \brief The stage this node attaches to */
  Stage attach_stage;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("scope", &scope);
    v->Visit("op", &op);
    v->Visit("all_iter_vars", &all_iter_vars);
    v->Visit("leaf_iter_vars", &leaf_iter_vars);
    v->Visit("relations", &relations);
    v->Visit("iter_var_attrs", &iter_var_attrs);
    v->Visit("attach_type", &attach_type);
    v->Visit("attach_ivar", &attach_ivar);
    v->Visit("attach_stage", &attach_stage);
  }

  static constexpr const char* _type_key = "Stage";
  TVM_DECLARE_NODE_TYPE_INFO(StageNode);
};

/*! \brief node container for schedule */
class ScheduleNode : public Node {
 public:
  /*! \brief The root operations */
  Array<Operation> roots;
  /*!
   * \brief list of all stages for non-placeholder ops
   *  The stage are ordered in PostDFS order of their op.
   */
  Array<Stage> stages;
  /*! \brief map of operation to the stages */
  Map<Operation, Stage> stage_map;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("roots", &roots);
    v->Visit("stages", &stages);
    v->Visit("stage_map", &stage_map);
  }

  static constexpr const char* _type_key = "Schedule";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleNode);
};

/*! \brief node container for IterVar attr */
class IterVarAttrNode : public Node {
 public:
  /*! \brief The iteration type. */
  IterVarType iter_type;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("iter_type", &iter_type);
  }

  static constexpr const char* _type_key = "IterVarAttr";
  TVM_DECLARE_NODE_TYPE_INFO(IterVarAttrNode);
};

/*! \brief base node of iteration var */
class IterVarRelationNode : public Node {
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
  Expr factor;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("parent", &parent);
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
    v->Visit("factor", &factor);
  }

  static IterVarRelation make(
      IterVar parent, IterVar outer,
      IterVar inner, Expr factor);

  static constexpr const char* _type_key = "Split";
  TVM_DECLARE_NODE_TYPE_INFO(SplitNode);
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

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("outer", &outer);
    v->Visit("inner", &inner);
    v->Visit("fused", &fused);
  }

  static IterVarRelation make(
      IterVar outer, IterVar inner, IterVar fused);

  static constexpr const char* _type_key = "Fuse";
  TVM_DECLARE_NODE_TYPE_INFO(FuseNode);
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

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("parent", &parent);
    v->Visit("rebased", &rebased);
  }

  static IterVarRelation make(IterVar parent, IterVar rebased);

  static constexpr const char* _type_key = "Rebase";
  TVM_DECLARE_NODE_TYPE_INFO(RebaseNode);
};


// implementations
inline const StageNode* Stage::operator->() const {
  return static_cast<const StageNode*>(node_.get());
}
inline StageNode* Stage::operator->() {
  return static_cast<StageNode*>(node_.get());
}

inline bool Stage::is_scheduled() const {
  const StageNode* n = operator->();
  return !(n->relations.empty() && n->attach_type == kNone);
}

inline const ScheduleNode* Schedule::operator->() const {
  return static_cast<const ScheduleNode*>(node_.get());
}

inline const IterVarRelationNode* IterVarRelation::operator->() const {
  return static_cast<const IterVarRelationNode*>(node_.get());
}

inline const IterVarAttrNode* IterVarAttr::operator->() const {
  return static_cast<const IterVarAttrNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_SCHEDULE_H_
