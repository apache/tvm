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
"""Registration of profiling objects in python."""

from .. import _ffi
from . import Object

_ffi._init_api("runtime.profiling", __name__)


@_ffi.register_object("runtime.profiling.Report")
class Report(Object):
    """A container for information gathered during a profiling run.

    Attributes
    ----------
    calls : Array[Dict[str, Object]]
        Per-call profiling metrics (function name, runtime, device, ...).

    device_metrics : Dict[Device, Dict[str, Object]]
        Per-device metrics collected over the entire run.
    """

    def csv(self):
        """Convert this profiling report into CSV format.

        This only includes calls and not overall metrics.

        Returns
        -------
        csv : str
            `calls` in CSV format.
        """
        return AsCSV(self)
