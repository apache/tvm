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
"""Namespace of all tag system in tvm

Each operator can be tagged by a tag, which indicate its type.

Generic categories

- tag.ELEMWISE="elemwise":
   Elementwise operator, for example :code:`out[i, j] = input[i, j]`
- tag.BROADCAST="broadcast":
    Broadcasting operator, can always map output axis to the input in order.
    for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
    Note that the axis need to be in order so transpose is not a bcast operator.
    If an input of broadcast operator has same shape as output,
    we can ensure that it is elementwise relation.
- tag.INJECTIVE="injective":
    Injective operator, can always injectively map output axis to a single input axis.
    All injective operator can still be safely fused similar to ewise to reduction.

- tag.COMM_REDUCE="comm_reduce":
    Communicative reduction operator
- If an op does not belong to these generic categories, it should have a special tag.

Note
----
When we add a new topi operator, the op need to be tagged as generic as possible.
We can also compose tags like "injective,pad" to give generic and specific information.
When we use composed tags, we must always put generic tag in the first location.
"""

ELEMWISE = "elemwise"
BROADCAST = "broadcast"
INJECTIVE = "injective"
COMM_REDUCE = "comm_reduce"
COMM_REDUCE_IDX = "comm_reduce_idx"


def is_broadcast(tag):
    """Check if a tag is bcast

    Parameters
    ----------
    tag : str
        The input tag

    Returns
    -------
    ret : bool
        Whether a tag is broadcast
    """
    if tag in (ELEMWISE, BROADCAST):
        return True
    return tag.startswith(ELEMWISE) or tag.startswith(BROADCAST)


def is_injective(tag):
    """Check if a tag is injective

    Parameters
    ----------
    tag : str
        The input tag

    Returns
    -------
    ret : bool
        Whether a tag is injective
    """
    if tag in (ELEMWISE, BROADCAST, INJECTIVE):
        return True
    return tag.startswith(ELEMWISE) or tag.startswith(BROADCAST) or tag.startswith(INJECTIVE)
