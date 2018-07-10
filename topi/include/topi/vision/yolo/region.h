/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Region op constructions
 * \file vision/yolo/region.h
 */
#ifndef TOPI_VISION_YOLO_REGION_H_
#define TOPI_VISION_YOLO_REGION_H_

#include <algorithm>
#include <string>

#include "topi/detail/constant_utils.h"
#include "topi/reduction.h"
#include "topi/tags.h"
#include "topi/transform.h"
#include "topi/nn/softmax.h"
#include "tvm/tvm.h"


namespace topi {
namespace vision {
namespace yolo {
using namespace tvm;
using namespace nn;

/*!
* \brief region operation
*
* \param data The input tensor. Can be any dimension
* \param num Darknet layer parameter n
* \param classes number of classes in the yolo model
* \param coords Darknet layer parameter coords
* \param background Darknet layer parameter background
* \param l_softmax if true apply softmax
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the region operation
*/
inline Tensor region(const Tensor &data,
                     int num,
                     int classes,
                     int coords,
                     int background,
                     int l_softmax,
                     std::string name = "tensor",
                     std::string tag = "region_output") {
  auto input_shape = data->shape;
  int split_size = classes + coords + 1;
  Array <Expr> intermediate_shape = {input_shape[0],
                                     num,
                                     split_size,
                                     input_shape[2],
                                     input_shape[3]};
  auto data_block = reshape(data, intermediate_shape);
  Array <Expr> split_indices;
  for (int i = 1; i < split_size; ++i) {
    split_indices.push_back(i);
  }
  Array <Tensor> split_res = split(data_block, split_indices, 2);
  split_res.Set(0, sigmoid(split_res[0]));
  split_res.Set(1, sigmoid(split_res[1]));
  if (!background) {
    split_res.Set(coords, sigmoid(split_res[coords]));
  }

  if (l_softmax) {
    int offset = coords + static_cast<int>(!background);
    Array <Tensor> softmax_input(split_res.begin() + offset, split_res.end());
    auto softmax_output = softmax(concatenate(softmax_input, 2), 2);
    Array <Tensor> data_block_1(split_res.begin(), split_res.begin() + offset);
    data_block_1.push_back(softmax_output);
    split_res = data_block_1;
  }
  Tensor out = concatenate(split_res, 2);
  return reshape(out, input_shape);
}
}  // namespace yolo
}  // namespace vision
}  // namespace topi
#endif  // TOPI_VISION_YOLO_REGION_H_
