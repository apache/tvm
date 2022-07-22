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

import pytest
import numpy as np
from typing import *
import collections
import tvm.testing


def get_test_id(*test_params, test_param_descs: List[Optional[str]] = None) -> str:
    """
    An opinionated alternative to pytest's default algorithm for generating a
    test's ID string.  Intended to make it easier for human readers to
    interpret the test IDs.

    'test_params': The sequence of pytest parameter values supplied to some unit
       test.

    'test_param_descs': An (optional) means to provide additional text for some/all of the
       paramuments in 'test_params'.

       If provided, then len(test_params) must equal len(test_param_descs).
       Each element test_param_descs that is a non-empty string will be used
       in some sensible way in this function's returned string.
    """

    assert len(test_params) > 0

    if test_param_descs is None:
        test_param_descs = [None] * len(test_params)
    else:
        assert len(test_param_descs) == len(test_params)

    def get_single_param_chunk(param_val, param_desc: Optional[str]):
        if type(param_val) == list:
            # Like str(list), but avoid the whitespace padding.
            val_str = "[" + ",".join(str(x) for x in param_val) + "]"
            need_prefix_separator = False

        elif type(param_val) == bool:
            if param_val:
                val_str = "T"
            else:
                val_str = "F"
            need_prefix_separator = True

        else:
            val_str = str(param_val)
            need_prefix_separator = True

        if param_desc and need_prefix_separator:
            return f"{param_desc}:{val_str}"
        elif param_desc and not need_prefix_separator:
            return f"{param_desc}{val_str}"
        else:
            return val_str

    chunks = [
        get_single_param_chunk(param_val, param_desc)
        for param_val, param_desc in zip(test_params, test_param_descs)
    ]
    return "-".join(chunks)


def get_multitest_ids(
    multitest_params_list: List[List], param_descs: Optional[List[Optional[str]]]
) -> List[str]:
    """
    A convenience function for classes that use both 'tvm.testing.parameters' and 'get_test_id'.

    This function provides a workaround for a specific quirk in Python, where list-comprehension
    can't necessarily access the value of another class-variable, discused here:
    https://stackoverflow.com/q/13905741
    """
    return [
        get_test_id(*single_test_param_list, test_param_descs=param_descs)
        for single_test_param_list in multitest_params_list
    ]
