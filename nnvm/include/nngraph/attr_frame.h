/*!
 *  Copyright (c) 2016 by Contributors
 * \file attr_frame.h
 * \brief Attribute frame data structure for properties in the graph.
 *   This data structure is inspired by data_frame for general.
 */
#include "./base.h"

namespace nngraph {


struct AttrFrame {
  std::unique_ptr<std::unordered_map<std::string, any> > info;
};

}  // namespace nngraph
