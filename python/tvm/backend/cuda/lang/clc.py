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
"""Decoding a ``clusterlaunchcontrol.try_cancel`` response.

The hardware writes an opaque 16-byte response into shared memory; reading a
scheduled CTA id out of it takes three instructions (load, test, extract) plus
a proxy fence, which is why it lives here rather than at every call site.
"""

from tvm.script import tirx as T


@T.inline
def query_cancel_first_ctaid_x(first_ctaid_x, handle, *, use_ld_acquire=True):
    """Decode the response at ``handle`` into ``first_ctaid_x``.

    Writes ``0xFFFFFFFF`` when the cancellation request did not succeed: the
    extraction is predicated on it, and ``get_first_ctaid`` accumulates into
    its destination, so an unsuccessful query leaves the sentinel in place.

    ``use_ld_acquire`` orders the (generic-proxy, weak) response read against
    the mbarrier wait that announced it; pass False only where the caller has
    already established that order.
    """
    response = T.local_scalar("uint128")
    canceled = T.local_scalar("uint32")

    T.ptx[f"ld{'.acquire.cta' if use_ld_acquire else ''}.shared.b128"](response, handle)
    T.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response)
    first_ctaid_x = T.uint32(0xFFFFFFFF)
    T.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        first_ctaid_x, response, pred=canceled
    )
    # Release the generic-proxy read of the handle before the next iteration's
    # async-proxy write to it (ISA 9.7.14.15's own example does the same).
    T.ptx.fence.proxy.async_.shared__cta()
