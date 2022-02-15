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
# pylint: disable=invalid-name, unused-argument
import pytest

pytest.importorskip("ethosu.vela")

import tensorflow as tf
from tests.python.contrib.test_ethosu.test_end_to_end import comparison_infra


ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32", "ethos-u65-256"]


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,size,align_corners",
    [
        [(1, 2, 2, 1), (4, 4), False],
        [(1, 4, 7, 3), (8, 14), False],
        [(1, 2, 2, 1), (3, 3), True],
        [(1, 4, 7, 3), (7, 13), True],
        [(1, 3, 5, 3), (3, 5), False],
    ],
)
def test_tflite_resize2d_bilinear(accel_type, ifm_shape, size, align_corners):
    @tf.function
    def resize_model(x):
        return tf.compat.v1.image.resize_bilinear(
            x, size, align_corners=align_corners, half_pixel_centers=False
        )

    # TODO(lhutton1) For now output is not bit exact with TFLite.
    # This is because TFLite reference kernels are not being used.
    # For this, TFLite will need upgrading to 2.6.
    comparison_infra._compare_tvm_with_tflite(
        resize_model, [ifm_shape], accel_type, output_tolerance=1
    )


if __name__ == "__main__":
    pytest.main([__file__])
