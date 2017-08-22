/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Configuration of nnvm as well as basic data structure.
 */
#ifndef NNVM_BASE_H_
#define NNVM_BASE_H_

#include <dmlc/base.h>
#include <dmlc/common.h>
#include <dmlc/any.h>
#include <dmlc/memory.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <dmlc/array_view.h>

namespace nnvm {

/*! \brief any type */
using dmlc::any;

/*! \brief array_veiw type  */
using dmlc::array_view;

/*!\brief getter function of any type */
using dmlc::get;

}  // namespace nnvm

#endif  // NNVM_BASE_H_
