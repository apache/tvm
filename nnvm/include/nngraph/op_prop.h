/*!
 *  Copyright (c) 2016 by Contributors
 * \file op_prop.h
 * \brief Data structure about property of operators
 */
#ifndef NNGRAPH_OP_PROP_H_
#define NNGRAPH_OP_PROP_H_

namespace nngraph {

/*!
 * \brief operator specific data structure
 */
struct OpProperty {
  /*! \brief name of the operator */
  std::string name;
  /*! \brief number of inputs to the operator */
  int num_inputs;
  /*! \brief number of outputs to the operator */
  int num_outputs;
};


}  // namespace nngraph

#endif  // NNGRAPH_OP_PROP_H_
