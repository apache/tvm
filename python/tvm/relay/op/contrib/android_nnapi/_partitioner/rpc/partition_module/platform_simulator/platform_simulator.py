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
"""Simulate computation platform and compute runtime costs for a given Relay IR Function."""
import tvm
from . import compute_device
from . import _utils


class PlatformSimulator(tvm.relay.ExprVisitor):
    """Simulate computation platform and compute runtime costs for a given Relay IR Function.

    Parameters
    ----------
    tracker: tvm.rpc.TrackerSession
        The tracker client managing RPC device sessions.

    options: dict
        The partitioner option dict.
    """

    ENABLED_DEVICES = [compute_device.TvmDevice.DEV_NAME, compute_device.NnapiDevice.DEV_NAME]

    def __init__(self, tracker, options, branching_nodes):
        super().__init__()
        self._tracker = tracker
        self._options = options

        # DP artifacts
        self._node_costs = {dev: {} for dev in self.ENABLED_DEVICES}
        self._node_transfers = {dev: {} for dev in self.ENABLED_DEVICES}

        # node assignment exceptions
        self._pinned_nodes = {n: compute_device.TvmDevice.DEV_NAME for n in branching_nodes}

        # init platform components
        def _scope():
            self._compute_devices = {
                compute_device.TvmDevice.DEV_NAME: compute_device.TvmDevice(options, self._tracker),
                compute_device.NnapiDevice.DEV_NAME: compute_device.NnapiDevice(
                    options, self._tracker
                ),
            }

        _scope()
        assert all([dev in self._compute_devices for dev in self.ENABLED_DEVICES])

        # measure data movement costs
        self._data_movement_costs = {dev: {} for dev in self.ENABLED_DEVICES}
        for sdev in self.ENABLED_DEVICES:
            for tdev in self.ENABLED_DEVICES:
                self._data_movement_costs[sdev][tdev] = (
                    0
                    if sdev == tdev
                    else self._compute_devices[sdev].estimate_single_byte_read_cost_to_bus()
                    + self._compute_devices[tdev].estimate_single_byte_write_cost_to_bus()
                )

    @property
    def node_costs(self):
        return self._node_costs

    @property
    def node_transfers(self):
        return self._node_transfers

    def calculate_cost(self, func):
        """Compute runtime costs for a given Relay IR Function.

        Parameters
        ----------
        func: tvm.relay.Function
            The function whose cost is to be evaluated.
        """
        self.visit(func)

    def visit_tuple(self, tup):
        super().visit_tuple(tup)

        for tdev in self.ENABLED_DEVICES:
            if self._skip_node_on_dev(tup, tdev):
                continue
            t_cost = 0
            for f in tup.fields:
                if f in self._node_costs[tdev]:
                    t_cost += self._node_costs[tdev][f]
                else:
                    t_cost = None
                    break
            if t_cost is None:
                continue
            self._node_costs[tdev][tup] = t_cost
            self._node_transfers[tdev][tup] = tdev

    def visit_call(self, call):
        super().visit_call(call)

        for tdev in self.ENABLED_DEVICES:
            c_cost = None
            for cdev in self.ENABLED_DEVICES:  # compute device
                if self._skip_node_on_dev(call, cdev):
                    continue
                cost = 0
                for a in call.args:
                    if a in self._node_costs[cdev]:
                        cost += self._node_costs[cdev][a]
                    else:
                        cost = None
                        break
                if cost is None:
                    continue

                if isinstance(call.op, tvm.ir.Op):
                    op_cost = self._compute_devices[cdev].estimate_call_op_cost(call)
                    if op_cost is None:
                        continue
                    cost += op_cost
                elif isinstance(call.op, (tvm.relay.Function, tvm.relay.GlobalVar)):
                    if call.op not in self._node_costs[cdev]:
                        continue
                    cost += self._node_costs[cdev][call.op]
                else:
                    raise NotImplementedError(call.op.type_key)
                cost += self.get_transfer_cost(call, cdev, tdev)
                if c_cost is None or c_cost > cost:
                    c_cost = cost
                    if isinstance(call.op, (tvm.relay.Function, tvm.relay.GlobalVar)):
                        assert cdev == compute_device.TvmDevice.DEV_NAME
                    self._node_transfers[tdev][call] = cdev
                    if isinstance(call.op, tvm.ir.Op):
                        self._node_transfers[tdev][call.op] = cdev
            assert c_cost is not None
            self._node_costs[tdev][call] = c_cost

    def visit_var(self, var):
        super().visit_var(var)
        if isinstance(var.checked_type, tvm.relay.TupleType):
            self._node_costs[compute_device.TvmDevice.DEV_NAME][var] = 0
            self._node_transfers[compute_device.TvmDevice.DEV_NAME][
                var
            ] = compute_device.TvmDevice.DEV_NAME
        else:
            for tdev in self.ENABLED_DEVICES:
                if self._skip_node_on_dev(var, tdev):
                    continue
                self._node_costs[tdev][var] = self.get_transfer_cost(
                    var, compute_device.TvmDevice.DEV_NAME, tdev
                )
                self._node_transfers[tdev][var] = compute_device.TvmDevice.DEV_NAME

    def visit_let(self, let):
        raise NotImplementedError(let.type_key)

    def visit_function(self, f):
        super().visit_function(f)
        assert f not in self._pinned_nodes
        f_cost = None
        for sdev in self.ENABLED_DEVICES:
            if f.body in self._node_costs[sdev]:
                cost = self._node_costs[sdev][f.body] + self.get_transfer_cost(
                    f.body, sdev, compute_device.TvmDevice.DEV_NAME
                )
                if f_cost is None or f_cost > cost:
                    f_cost = cost
                    fb_dev = sdev
        assert f_cost is not None
        self._node_costs[compute_device.TvmDevice.DEV_NAME][f] = f_cost
        self._node_transfers[compute_device.TvmDevice.DEV_NAME][f] = fb_dev

    def visit_if(self, i):
        raise NotImplementedError(i.type_key)

    def visit_global_var(self, gv):
        super().visit_global_var(gv)
        assert gv not in self._pinned_nodes
        self._node_costs[compute_device.TvmDevice.DEV_NAME][gv] = 0
        self._node_transfers[compute_device.TvmDevice.DEV_NAME][
            gv
        ] = compute_device.TvmDevice.DEV_NAME

    def visit_constructor(self, c):
        raise NotImplementedError(c.type_key)

    def visit_constant(self, const):
        for tdev in self.ENABLED_DEVICES:
            if self._skip_node_on_dev(const, tdev):
                continue
            self._node_costs[tdev][const] = 0
            self._node_transfers[tdev][const] = tdev

    def visit_ref_create(self, r):
        raise NotImplementedError(r.type_key)

    def visit_ref_read(self, r):
        raise NotImplementedError(r.type_key)

    def visit_ref_write(self, r):
        raise NotImplementedError(r.type_key)

    def visit_tuple_getitem(self, t):
        super().visit_tuple_getitem(t)
        if isinstance(t.tuple_value, tvm.relay.Tuple):
            for tdev in self.ENABLED_DEVICES:
                if self._skip_node_on_dev(t, tdev):
                    continue
                self._node_costs[tdev][t] = self._node_costs[tdev][t.tuple_value]
                self._node_transfers[tdev][t] = tdev
        elif isinstance(t.tuple_value, (tvm.relay.Call, tvm.relay.Var)):
            for tdev in self.ENABLED_DEVICES:
                self._node_costs[tdev][t] = self._node_costs[compute_device.TvmDevice.DEV_NAME][
                    t.tuple_value
                ] + self.get_transfer_cost(t, compute_device.TvmDevice.DEV_NAME, tdev)
                self._node_transfers[tdev][t] = compute_device.TvmDevice.DEV_NAME
        else:
            raise NotImplementedError(t.tuple_value.type_key)

    def visit_match(self, m):
        raise NotImplementedError(m.type_key)

    def get_transfer_cost(self, node, sdev, tdev):
        if sdev == tdev:
            return 0
        return _utils.get_node_size(node) * self._data_movement_costs[sdev][tdev]

    def _skip_node_on_dev(self, node, dev):
        if node in self._pinned_nodes:
            if self._pinned_nodes[node] == dev:
                return False
            return True
        return False
