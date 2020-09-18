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

"""Per channel implementation of Conv2DPattern, Conv2DBiasAddPattern, and DensePattern, using the
average max algorithm to pick scales."""

import numpy as np

from tvm import relay
from tvm.relay.transform.quantize import (
    Conv2DPattern,
    Conv2DBiasAddPattern,
    DensePattern,
    DenseBiasAddPattern,
    PerChannelPattern,
    CalibrationCallback,
    QuantizerPattern,
)


class AverageMaxPerChannelPattern(PerChannelPattern):
    """Per channel implementation of the AverageMax algorithm."""

    def calibrate_pattern(self, calibration_info):
        self.attr_callback(calibration_info.partition_info.expr)
        scale_zp_values = {}

        data_max_avg = 0
        weight_max_avg = np.zeros(shape=self.get_scale_size())
        num_inputs = (
            calibration_info.dataset_manager.num_batches()
            * calibration_info.dataset_manager.batch_size()
        )

        while not calibration_info.dataset_manager.is_empty():
            # Get the original input from dataset manger, run unquantized graph with those inputs
            image_list, _ = calibration_info.dataset_manager.get_next_batch()
            unquantized_inputs = calibration_info.get_unquantized_layer_inputs(image_list)

            data = unquantized_inputs[0]
            weight = unquantized_inputs[1]

            data_max_avg += np.max(np.abs(data)) / num_inputs

            axis = list(range(len(weight.shape))).remove(0)
            weight_max_avg += np.max(np.abs(weight), axis=axis) / num_inputs

        calibration_info.dataset_manager.reset()

        # Since this is a symmetric distribution and we are quantizing to int8, there are 256 bins,
        # and 128 are positive
        data_scale = data_max_avg / 128
        weight_scales = weight_max_avg / 128
        scales = np.array([data_scale, weight_scales])

        for i, scale in enumerate(scales):
            scale_name = calibration_info.partition_info.input_scale_zps[i][0].name_hint
            zp_name = calibration_info.partition_info.input_scale_zps[i][1].name_hint

            scale_zp_values[scale_name] = scale.astype("float32")
            scale_zp_values[zp_name] = np.array(0).astype("int32")

        return scale_zp_values


class AverageMaxPerChannelConv2DPattern(AverageMaxPerChannelPattern, Conv2DPattern):
    """Conv2DPattern with the per channel average max algorithm as the calibration method."""

    def extract_attrs(self, pre, post, node_map):
        conv2d = node_map[self.conv2d][0]
        weight = node_map[self.conv_weight][0]

        self.get_attrs(conv2d.attrs, weight.checked_type.shape)
        return post

    def get_scale_size(self):
        return (self.channels,)


class AverageMaxPerChannelConv2DBiasAddPattern(
    AverageMaxPerChannelConv2DPattern, Conv2DBiasAddPattern
):
    """Per channel version of Conv2DBiasAddPattern, implementing the average max algorithm to
    calculate scales and zero points."""


class AverageMaxPerChannelDensePattern(AverageMaxPerChannelPattern, DensePattern):
    """Per channel version of DensePattern, implementing the average max algorithm to
    calculate scales and zero points."""

    def __init__(self, calibration_callback: CalibrationCallback = None):
        super().__init__(calibration_callback)

    def extract_attrs(self, pre, post, node_map):
        dense = node_map[self.dense][0]
        weight = node_map[self.weight][0]

        self.get_attrs(dense.attrs, weight.checked_type.shape)
        self.units = self.attrs["units"]

        return post

    def get_scale_size(self):
        return (self.units,)


class AverageMaxPerChannelDenseBiasAddPattern(
    AverageMaxPerChannelDensePattern, DenseBiasAddPattern
):
    """Per channel version of DenseBiasAddPattern, implementing the average max algorithm to
    calculate scales and zero point."""
