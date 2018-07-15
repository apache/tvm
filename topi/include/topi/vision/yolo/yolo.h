/*!
 *  Copyright (c) 2018 by Contributors
 * \brief YOLO op constructions
 * \file vision/yolo/yolo.h
 */
#ifndef TOPI_VISION_YOLO_YOLO_H_
#define TOPI_VISION_YOLO_YOLO_H_

#include <algorithm>
#include <string>

#include "topi/detail/constant_utils.h"
#include "topi/tags.h"
#include "topi/transform.h"
#include "tvm/tvm.h"


namespace topi {
namespace vision {
namespace yolo {
using namespace tvm;
using namespace nn;

/*!
* \brief yolo operation
*
* \param data The input tensor.
* \param num Darknet layer parameter n
* \param classes number of classes in the yolo model
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the yolo operation
*/
inline Tensor yolo(const Tensor &data,
                   int num,
                   int classes,
                   std::string name = "tensor",
                   std::string tag = "yolo_output") {
  auto input_shape = data->shape;
  int split_size = classes + 5;
  Array <Expr> intermediate_shape = {input_shape[0],
                                     num,
                                     split_size,
                                     input_shape[2],
                                     input_shape[3]};
  auto data_block = reshape(data, intermediate_shape);
  Array <Expr> split_indices = {2, 4};
  Array <Tensor> split_res = split(data_block, split_indices, 2);
  split_res.Set(0, sigmoid(split_res[0]));
  split_res.Set(2, sigmoid(split_res[2]));
  Tensor out = concatenate(split_res, 2);
  return reshape(out, input_shape);
}
}  // namespace yolo
}  // namespace vision
}  // namespace topi
#endif  // TOPI_VISION_YOLO_YOLO_H_
