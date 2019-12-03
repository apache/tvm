# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tensorflow as tf
from tensorflow.python.framework import load_library


class Module():

  def __init__(self, lib_path):
    self.lib_path = lib_path

  def func(self, name, output_dtype=None, output_shape=None):
    return Func(self.lib_path, name, output_dtype, output_shape)

  def __getitem__(self, func_name):
    return self.func(func_name)


class Func():

  def __init__(self, lib_path, func_name, output_dtype, output_shape):
    self.lib_path = lib_path
    self.func_name = func_name
    self.output_dtype = output_dtype

    # const(0) indicate invalid dynamic shape
    self.dynamic_output_shape = tf.constant(0, tf.int64)
    self.static_output_shape = None
    self.has_static_output_shape = False  # extra flag is required

    if self._is_static_shape(output_shape):
      self.static_output_shape = output_shape
      self.has_static_output_shape = True
    elif output_shape is not None:
      self.dynamic_output_shape = self._pack_shape_tensor(output_shape)
    
    # TODO: support non-xpu device 
    #self.device = device
    # delay initialization to called first time, where num input arguments is known
    self.tvm_dso_op = None
    self.module = load_library.load_op_library('tvm_dso_op.so')
    
  def apply(self, *params):
    if self.tvm_dso_op is None:
      num_inputs = len(params)
      self.tvm_dso_op = getattr(self.module, "tvm_dso_op%s" % num_inputs)
    
    return self.tvm_dso_op(*params, 
                           dynamic_output_shape=self.dynamic_output_shape,
                           static_output_shape=self.static_output_shape,
                           has_static_output_shape=self.has_static_output_shape, 
                           lib_path=self.lib_path, 
                           func_name=self.func_name, 
                           output_dtype=self.output_dtype)

  def __call__(self, *params):
    return self.apply(*params)

  def _is_static_shape(self, shape):
    if shape is None or not isinstance(shape, list):
      return False
    for d in shape:
      if not isinstance(d, int):
        return False
      if d < 0:
        raise Exception("Negative dimension is illegal: %d" % d)
    return True

  def _pack_shape_tensor(self, shape):
    if isinstance(shape, tf.Tensor):
      if shape.dtype == tf.int32:
        shape = tf.cast(shape, tf.int64)
      return shape
    elif isinstance(shape, list):
      shape_dims = []
      for d in shape:
        if isinstance(d, int):
          shape_dims.append(tf.constant(d, tf.int64))
        elif isinstance(d, tf.Tensor) and len(d.shape) == 0:
          if d.dtype == tf.int32:
            d = tf.cast(d, tf.int64)
          shape_dims.append(d)
        else:
          raise TypeError("Input shape dimension is neither scala tensor nor int")
      return tf.stack(shape_dims) 
    else:
      raise TypeError("Input shape is neither tensor nor list")



