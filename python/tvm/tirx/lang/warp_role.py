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
"""Warp role helpers for SM100 kernels.

Simplifies the common pattern of dispatching warps to named roles
with register budgets.

Example::

    # Declare roles
    tma_warp = WarpRole(warp_id, 1, regs=48)
    store_warp = WarpRole(warp_id, 2, regs=48)
    mma_warp = WarpRole(warp_id, 0, regs=232, increase=True)

    # Use with context manager
    with tma_warp:
        # TMA load code
    with store_warp:
        # TMA store code
    with mma_warp:
        # MMA compute code
"""

from tvm.script import tirx as T


class WarpRole:
    """A warp-level role that guards a block of code by warp_id comparison
    with optional register budget.

    Generates::

        if <warp_id_var> == <warp_id_val>:
            T.ptx.setmaxnreg(<increase>, <regs>)  # if regs specified
            <user code>

    The ``if`` guard narrows the active set to the single warp; individual
    tile-primitive calls inside ``<user code>`` carry their own exec scope via
    a scope-namespace prefix (e.g. ``Tx.warp.copy(...)``).

    Parameters
    ----------
    warp_id_var : Var
        The warp_id variable (from ``T.warp_id(...)``).
    warp_id_val : int
        Which warp index this role corresponds to.
    regs : int, optional
        Register budget (passed to ``T.ptx.setmaxnreg``).
        If None, no setmaxnreg is emitted.
    increase : bool
        Direction for ``setmaxnreg`` (default False = decrease).
    """

    def __init__(self, warp_id_var, warp_id_val, regs=None, increase=False):
        self.warp_id_var = warp_id_var
        self.warp_id_val = warp_id_val
        self.regs = regs
        self.increase = increase

    def __enter__(self):
        self._if_frame = T.If(self.warp_id_var == self.warp_id_val)
        self._if_frame.__enter__()
        self._then_frame = T.Then()
        self._then_frame.__enter__()
        if self.regs is not None:
            T.evaluate(T.ptx.setmaxnreg(self.increase, self.regs))
        return self

    def __exit__(self, *exc):
        self._then_frame.__exit__(*exc)
        self._if_frame.__exit__(*exc)
        return False


class WarpgroupRole:
    """A warpgroup-level role that guards by wg_id comparison,
    with optional register budget.

    Generates (single wg_id)::

        if <wg_id_var> == <wg_id_val>:
            T.ptx.setmaxnreg(<increase>, <regs>)  # if regs specified
            <user code>

    Generates (range of wg_ids, e.g. ``wg_id_val=(0, 2)``)::

        if 0 <= <wg_id_var> and <wg_id_var> < 2:
            T.ptx.setmaxnreg(<increase>, <regs>)
            <user code>

    The ``if`` guard narrows the active set to the target warpgroup(s);
    individual tile-primitive calls inside ``<user code>`` carry their own exec
    scope via a scope-namespace prefix (e.g. ``Tx.wg.copy(...)``).

    Parameters
    ----------
    wg_id_var : Var
        The warpgroup_id variable (from ``T.warpgroup_id(...)``).
    wg_id_val : int or tuple[int, int]
        Which warpgroup index (int) or range ``(start, stop)`` this role
        corresponds to.
    regs : int, optional
        Register budget.
    increase : bool
        Direction for ``setmaxnreg`` (default False = decrease).
    """

    def __init__(self, wg_id_var, wg_id_val, regs=None, increase=False):
        self.wg_id_var = wg_id_var
        self.wg_id_val = wg_id_val
        self.regs = regs
        self.increase = increase

    def __enter__(self):
        if isinstance(self.wg_id_val, tuple):
            start, stop = self.wg_id_val
            self._if_frame = T.If(start <= self.wg_id_var and self.wg_id_var < stop)
        else:
            self._if_frame = T.If(self.wg_id_var == self.wg_id_val)
        self._if_frame.__enter__()
        self._then_frame = T.Then()
        self._then_frame.__enter__()
        if self.regs is not None:
            T.evaluate(T.ptx.setmaxnreg(self.increase, self.regs))
        return self

    def __exit__(self, *exc):
        self._then_frame.__exit__(*exc)
        self._if_frame.__exit__(*exc)
        return False
