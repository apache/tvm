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
This file contains utility functions for processing the outputs
of TVMC models. These utilities are likely to be task specific,
overtime more will be added to support more machine learning tasks.

Examples
--------
The following code shows how one might postprocess
the output of a classification model.

.. code-block:: python
    result = tvmc.run(tvmc_package, device="cpu")
    top_results = result_utils.get_top_results(max_results=5)
"""
import numpy as np
from .model import TVMCResult


def get_top_results(result: TVMCResult, max_results: int):
    """Return the top n results from the output tensor.

    This function is primarily for image classification and will
    not necessarily generalize.

    Parameters
    ----------
    result : TVMCResult
        The output of a TVMCModel
    max_results : int
        Number of results to return

    Returns
    -------
    top_results : np.array
        Results array of shape (2, n).
        The first row is the indices and the second is the values.

    """
    output = np.copy(result.outputs["output_0"])
    sorted_labels = output.argsort()[0][-max_results:][::-1]
    output.sort()
    sorted_values = output[0][-max_results:][::-1]
    top_results = np.array([sorted_labels, sorted_values])
    return top_results
