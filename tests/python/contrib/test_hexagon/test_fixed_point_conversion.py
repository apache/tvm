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

"""
Test float to fixed-point conversion. We do it by constructing a numpy array with the
wide range of floating-point values. These values are converted into the
fixed-point value using topi.hexagon.utils.get_fixed_point_value. Then, these values are
converted back into float using scale_factor provided by the function. These converted
floating point values are then compared against the original values and an assertion is
raised if they happened to be outside of the expected tolerance.
"""

import math
import struct
import numpy as np
import tvm.topi.hexagon.utils as utils


class TestFixedPointConversion:
    """Fixed point conversation test class"""

    def test_fixed_point_conversion(self):
        """Test fixed point conversion"""
        # Construct array with wide range of values
        fp1 = np.random.uniform(0.00001, 0.0002, size=(10))
        fp2 = np.random.uniform(0.001, 0.02, size=(10))
        fp3 = np.random.uniform(1, 20, size=(10))
        fp4 = np.random.uniform(900, 1000, size=(10))
        fp5 = np.random.uniform(1e9, 1e10, size=(10))

        # Test for values with largest possible exponent as per IEEE-754 floating-point
        # standard (actual exp value = 127, stored exp value = 254).
        fp6 = np.random.uniform(2.4e38, 2.5e38, size=(1))

        # Test for very small floating-point values.
        fp7 = np.random.uniform(1.4e-34, 1.7e-34, size=(1))

        float_arr = np.concatenate((fp1, fp2, fp3, fp4, fp5, fp6, fp7))
        for flp in float_arr:
            fxp, rsh = utils.get_fixed_point_value(flp, "int16")
            # Compute scale_factor using rsh (rsh is log2 of the scale_factor). While doing this,
            # we use IEEE-754 floating-point representation since rsh can be negative or positive.

            scale = ((rsh + 127) & 0xFF) << 23  # Add bias (127) and position it into exponent bits
            scale_i = struct.pack("I", scale)  # Pack it as integer
            scale_f = struct.unpack("f", scale_i)  # Unpack as float

            converted_flp = fxp / scale_f[0]
            assert math.isclose(flp, converted_flp, rel_tol=1e-2)


if __name__ == "__main__":
    tvm.testing.main()
