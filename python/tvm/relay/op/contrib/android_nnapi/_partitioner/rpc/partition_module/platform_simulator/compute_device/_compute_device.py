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
"""Base class for computation device
"""


class ComputeDevice:
    """Base class for computation device"""

    def estimate_call_op_cost(self, call):
        """Estimate the runtime cost of executing a given call

        Parameters
        ----------
        call: tvm.relay.Call
            The Relay call expression whose runtime cost is to be estimated

        """
        raise NotImplementedError()

    def estimate_single_byte_read_cost_to_bus(self):  # pylint: disable=invalid-name
        """Estimate the runtime cost of reading a single byte to the bus from
        the internal memory managed by this compute device
        """
        raise NotImplementedError()

    def estimate_single_byte_write_cost_to_bus(self):  # pylint: disable=invalid-name
        """Estimate the runtime cost of writing a single byte to the bus from
        the internal memory managed by this compute device
        """
        raise NotImplementedError()
