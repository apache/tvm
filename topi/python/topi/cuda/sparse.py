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

import tvm

from .. import generic 
from .injective import schedule_injective

@generic.schedule_sparse_dense.register(["gpu", "cuda"])
def _schedule_sparse_dense(outs):
    """Schedule for sparse_dense operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for sparse_dense.
    """
    target = tvm.target.current_target()

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    if target.target_name == "cuda" and "cusparse" in target.libs:
        return generic.schedule_extern(outs)

    s = schedule_injective(outs)
    return s
