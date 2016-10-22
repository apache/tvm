/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.h
 * \brief Define a schedule.
 */
#ifndef TVM_SCHEDULE_H_
#define TVM_SCHEDULE_H_

#include <string>
#include "./base.h"
#include "./split.h"
#include "./tensor.h"

namespace tvm {

// Node container for Schedule
class ScheduleNode;
// Node container for AttachSpec
class AttachSpecNode;

/*! \brief the attachment type */
enum AttachType : int {
  kRoot = 0,
  kInline = 1,
  kSplit = 2
};

/*! \brief schedule container */
class Schedule : public NodeRef {
 public:
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const ScheduleNode* operator->() const;
};

/*! \brief schedule container */
class AttachSpec : public NodeRef {
 public:
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const AttachSpecNode* operator->() const;
};

// defintion of node containers

/*! \brief The attach specification of each subschedule */
class AttachSpecNode : public Node {
 public:
  /*! \brief The attachment type */
  AttachType attach_type;
  /*!
   * \brief The split to be attached to,
   *  only valid when attach_type is kRoot
   */
  Split attach_split;
  /*! \brief the child schedule to be attached. */
  Schedule schedule;
  const char* type_key() const override {
    return "AttachSpecNode";
  }
  void VisitAttrs(AttrVisitor* visitor) override {
    visitor->Visit("attach_type", &attach_type);
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("attach_split", &attach_split);
    fvisit("schedule", &schedule);
  }
};

/*! \brief represents the schedule of the tensor */
class ScheduleNode : public Node {
 public:
  /*! \brief Tensor to be scheduled */
  Tensor tensor;
  /*! \brief The thread scope level of the schedule */
  std::string scope;
  /*! \brief Splits over domains or rdomains */
  Array<Split> splits;
  /*! \brief attach specifications */
  Array<AttachSpec> attachs;
  const char* type_key() const override {
    return "AttachSpecNode";
  }
  void VisitAttrs(AttrVisitor* visitor) override {
    visitor->Visit("scope", &scope);
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("tensor", &tensor);
    fvisit("splits", &splits);
    fvisit("attachs", &attachs);
  }
};

// implementations
inline const ScheduleNode* Schedule::operator->() const {
  return static_cast<const ScheduleNode*>(node_.get());
}

inline const AttachSpecNode* AttachSpec::operator->() const {
  return static_cast<const AttachSpecNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_SCHEDULE_H_
