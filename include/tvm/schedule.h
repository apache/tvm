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
  kRoot = 0,
  kInline = 1,
  kSplit = 2
};

/*! \brief schedule container */
class Schedule : public NodeRef {
 public:
  Schedule() {}
  explicit Schedule(std::shared_ptr<Node> n) : NodeRef(n) {}
  Schedule(Operation op, std::string scope);
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const ScheduleNode* operator->() const;
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
  AttachType attach_type;
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

inline const IterVarRelationNode* IterVarRelation::operator->() const {
  return static_cast<const IterVarRelationNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_SCHEDULE_H_
