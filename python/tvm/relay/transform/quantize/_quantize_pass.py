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

"""Relay pass wrapping the quantization and calibration workflow."""

from typing import List

import tvm
from tvm.relay.transform.quantize import (
    Quantizer,
    QuantizationCalibrator,
    Requantizer,
    QuantizerPattern,
)
from .. import function_pass


@function_pass(opt_level=5)
class QuantizePass:
    """Explicit relay pass wrapper around quantization workflow.

    Parameters
    ----------
    quantizer_pattern_list : List[QuantizerPattern]
        The patterns we want to quantize.

    params : dict of str to NDArray
            Constants needed to run the mod. We need params so that we can run parts of the
            graph during calibration.

    target : str
        Target to generate code for calibration on.

    skip_first : bool
        If True, we do not quantize the first quantizable pattern in the function. If False,
        we will quantize it.

    skip_last : bool
        If True, we do not quantize the last quantizable pattern in the function. If False,
        we will quantize it."""

    def __init__(
        self,
        quantizer_pattern_list: List[QuantizerPattern],
        params=None,
        target="llvm",
        device=tvm.cpu(0),
        skip_first=True,
        skip_last=False,
    ):
        self.quantizer_pattern_list = quantizer_pattern_list
        self.params = params
        self.target = target
        self.device = device
        self.skip_first = skip_first
        self.skip_last = skip_last

    def transform_function(self, func, mod, ctx):
        """Quantizes, calibrates and requantizes the function.
        Parameters
        ----------
        func : relay.Function
            Function to apply the transformation on.

        """
        params = {}
        # Extract params that are in this function
        for param in func.params:
            if param.name_hint in self.params.keys():
                params[param.name_hint] = self.params[param.name_hint]
        quantizer = Quantizer(
            func, params, self.quantizer_pattern_list, self.skip_first, self.skip_last
        )

        calibrator = QuantizationCalibrator(quantizer, target=self.target, ctx=self.device)
        transformed_func = calibrator.calibrate()
        transformed_func = Requantizer().requantize(transformed_func)
        return transformed_func
