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
"""Module container of TensorFlow TVMDSO op"""
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python import platform


class OpModule:
    """Module container of TensorFlow TVMDSO op which wraps exported
    TVM op implementation library to be called on TensorFlow side"""

    def __init__(self, lib_path):
        self.lib_path = lib_path

    def func(self, name, output_dtype=None, output_shape=None):
        """Get tvm op function wrapped as TensorFlow tensor to tensor function

        Parameters
        ----------
        name: str
            function name
        output_dtype: str or TensorFlow datatype
            Output datatype, default is float32
        output_shape: List of integer/tf scalar tensor or tf shape tensor
            Output shape, default the same with first input's shape

        Returns
        ----------
        Func object that acts as TensorFlow tensor to tensor function.
        """
        return TensorFunc(self.lib_path, name, output_dtype, output_shape)

    def __getitem__(self, func_name):
        return self.func(func_name)


class TensorFunc:
    """Function object that acts as TensorFlow tensor to tensor function."""

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

        self.module = self._load_platform_specific_library("libtvm_dso_op")
        self.tvm_dso_op = self.module.tvm_dso_op

    def apply(self, *params):
        return self.tvm_dso_op(
            params,
            dynamic_output_shape=self.dynamic_output_shape,
            static_output_shape=self.static_output_shape,
            has_static_output_shape=self.has_static_output_shape,
            lib_path=self.lib_path,
            func_name=self.func_name,
            output_dtype=self.output_dtype,
        )

    def __call__(self, *params):
        return self.apply(*params)

    def _load_platform_specific_library(self, lib_name):
        system = platform.system()
        if system == "Darwin":
            lib_file_name = lib_name + ".dylib"
        elif system == "Windows":
            lib_file_name = lib_name + ".dll"
        else:
            lib_file_name = lib_name + ".so"
        return load_library.load_op_library(lib_file_name)

    def _is_static_shape(self, shape):
        if shape is None or not isinstance(shape, list):
            return False
        for dim_value in shape:
            if not isinstance(dim_value, int):
                return False
            if dim_value < 0:
                raise Exception("Negative dimension is illegal: %d" % dim_value)
        return True

    def _pack_shape_tensor(self, shape):
        if isinstance(shape, tf.Tensor):
            if shape.dtype == tf.int32:
                shape = tf.cast(shape, tf.int64)
        elif isinstance(shape, list):
            shape_dims = []
            for dim_value in shape:
                if isinstance(dim_value, int):
                    shape_dims.append(tf.constant(dim_value, tf.int64))
                elif isinstance(dim_value, tf.Tensor) and dim_value.shape.rank == 0:
                    if dim_value.dtype == tf.int32:
                        dim_value = tf.cast(dim_value, tf.int64)
                    shape_dims.append(dim_value)
                else:
                    raise TypeError("Input shape dimension is neither scalar tensor nor int")
            shape = tf.stack(shape_dims)
        else:
            raise TypeError("Input shape is neither tensor nor list")
        return shape
