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

// Node container for Schedule
class ScheduleNode;
// Node container for IterVarRelation
class IterVarRelationNode;

/*! \brief the attachment type */
enum AttachType : int {
  kNone = 0,
  kRoot = 1,
  kInline = 2,
  kScope = 3
};

/*! \brief schedule container */
class Schedule : public NodeRef {
 public:
  Schedule() {}
  explicit Schedule(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief create a new schedule for op.
   * \param op The operator in the schedule
   * \param scope The scope of the schedule
   */
  Schedule(Operation op, std::string scope);
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
  /*!
   * \brief specify the schedule to be computed at the parent schedule's scope.
   * \param parent The parent schedule.
   * \param scope The iteration point to carry the schedule.
   * \return reference to self.
   */
  Schedule& compute_at(Schedule parent, IterVar scope);   // NOLINT(*)
  /*!
   * \brief Compute the function inline, attach it at parent.
   * \param parent The parent schedule to be attached to.
   * \return reference to self.
   */
  Schedule& compute_inline(Schedule parent);   // NOLINT(*)
  /*!
   * \brief Compute the function at root, attach it to its parent.
   * \param parent The parent schedule to be attached to.
   * \return reference to self.
   */
  Schedule& compute_root(Schedule parent);  // NOLINT(*)
  /*!
   * \brief Split the parent by factor, generate
   * \param parent The parent iteration domain.
   * \param p_outer The result outer domain
   * \param p_inner The result inner domain.
   * \param factor The split factor of the loop.
   * \return reference to self.
   */
  Schedule& split(IterVar parent, IterVar* p_outer, IterVar* p_inner, Expr factor);  // NOLINT(*)
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
  Schedule& split(IterVar parent, IterVar outer, IterVar* p_inner, Expr factor = Expr());   // NOLINT(*)
  /*!
   * \brief Fuse the inner outer domain to the target
   * \param inner The inner domain to be fused
   * \param outer The outer domain to be fused.
   * \param p_target The result target domain.
   * \return reference to self.
   */
  Schedule& fuse(IterVar inner, IterVar outer, IterVar* p_target);  // NOLINT(*)
  /*!
   * \brief Reorder the iteration
   * \param order The order of iteration variable.
   * \return reference to self.
   */
  Schedule& reorder(const Array<IterVar>& order);   // NOLINT(*)
  Schedule& tile(IterVar x_parent, IterVar y_parent, IterVar* p_x_outer,
                 IterVar* p_y_outer, IterVar* p_x_inner, IterVar* p_y_inner,
                 Expr x_factor, Expr y_factor);   // NOLINT(*)
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
class ScheduleNode : public Node {
 public:
  /*! \brief The operation to be scheduled */
  Operation op;
  /*! \brief The thread scope level of the schedule */
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
  /*! \brief The attachment type of the schedule */
  AttachType attach_type{kNone};
  /*!
   * \brief The attach point of this schedule.
   */
  IterVar attach_parent;
  /*! \brief the schedules that this schedule depend on */
  Array<Schedule> children;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("scope", &scope);
    v->Visit("op", &op);
    v->Visit("all_iter_vars", &all_iter_vars);
    v->Visit("leaf_iter_vars", &leaf_iter_vars);
    v->Visit("relations", &relations);
    v->Visit("attach_type", &attach_type);
    v->Visit("attach_parent", &attach_parent);
    v->Visit("children", &children);
  }

  static constexpr const char* _type_key = "Schedule";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleNode);
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

// implementations
inline const ScheduleNode* Schedule::operator->() const {
  return static_cast<const ScheduleNode*>(node_.get());
}
inline ScheduleNode* Schedule::operator->() {
  return static_cast<ScheduleNode*>(node_.get());
}

inline const IterVarRelationNode* IterVarRelation::operator->() const {
  return static_cast<const IterVarRelationNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_SCHEDULE_H_
