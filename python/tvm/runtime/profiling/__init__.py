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

from typing import Dict, Sequence, Optional
from ... import _ffi
from . import _ffi_api
from .. import Object, Device


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
        return _ffi_api.AsCSV(self)

    def table(self, sort=True, aggregate=True, col_sums=True):
        """Generate a human-readable table

        Parameters
        ----------
        sort : bool

            If aggregate is true, whether to sort call frames by
            descending duration.  If aggregate is False, whether to
            sort frames by order of appearancei n the program.

        aggregate : bool

            Whether to join multiple calls to the same op into a
            single line.

        col_sums : bool

            Whether to include the sum of each column.

        Returns
        -------
        table : str

            A human-readable table

        """
        return _ffi_api.AsTable(self, sort, aggregate, col_sums)

    def json(self):
        """Convert this profiling report into JSON format.

        Example output:

        .. code-block:

            {
              "calls": [
                {
                  "Duration (us)": {
                    "microseconds": 12.3
                  },
                  "Name": "fused_dense",
                  "Count": {
                    "count": 1
                  },
                  "Percent": {
                    "percent": 10.3
                  }
                }
              ],
              "device_metrics": {
                "cpu": {
                  "Duration (us)": {
                    "microseconds": 334.2
                  },
                  "Percent": {
                    "percent": 100
                  }
                }
              }
            }

           {"calls":
              [
                {"Duration (us)": {"microseconds": 12.3}
                 ,"Name": "fused_dense"
                 ,"Count": {"count":1}
                 ,"Percent": {"percent": 10.3}
                 }
              ],
            "device_metrics":
              {"cpu":
                {"Duration (us)": {"microseconds": 334.2}
                ,"Percent": {"percent": 100.0}
                }
              }
           }

        Returns
        -------
        json : str
            Formatted JSON
        """
        return _ffi_api.AsJSON(self)

    @classmethod
    def from_json(cls, s):
        """Deserialize a report from JSON.

        Parameters
        ----------
        s : str
            Report serialize via :py:meth:`json`.

        Returns
        -------
        report : Report
            The deserialized report.
        """
        return _ffi_api.FromJSON(s)


@_ffi.register_object("runtime.profiling.MetricCollector")
class MetricCollector(Object):
    """Interface for user defined profiling metric collection."""


@_ffi.register_object("runtime.profiling.DeviceWrapper")
class DeviceWrapper(Object):
    """Wraps a tvm.runtime.Device"""

    def __init__(self, dev: Device):
        self.__init_handle_by_constructor__(_ffi_api.DeviceWrapper, dev)


def profile_function(mod, dev, collectors, func_name="main", warmup_iters=10):
    """Collect performance information of a function execution. Usually used with
    a compiled PrimFunc.

    This information can include performance counters like cache hits and FLOPs
    that are useful in debugging performance issues of individual PrimFuncs.
    Different metrics can be collected depending on which MetricCollector is
    used.

    Example
    -------

    .. code-block: python
        f = tvm.build(my_func, target="llvm", name="my_func")
        prof = tvm.runtime.profiling.profile_function(
            f,
            tvm.cpu(),
            [tvm.runtime.profiling.PAPIMetricCollector({tvm.cpu(): ["PAPI_FP_OPS"]}),
        )
        counters = prof(*args)
        print(counters)

    Parameters
    ----------
    mod: Module
        Module containing the function to profile.
    dev: Device
        Device to run the function on.

    collectors: List[MetricCollector]
        :py:class:`MetricCollector`s which will collect performance information.
    func_name: str
        Name of the function in `mod` to profile. Defaults to "main".
    warmup_iters: int
        Number of iterations to run the function before collecting performance
        information. Recommended to set this larger than 0 for consistent cache
        effects. Defaults to 10.

    Returns
    -------
    prof: PackedFunc[args, Dict[str, ObjectRef]]
        PackedFunc which takes the same arguments as the `mod[func_name]` and
        returns performance metrics as a `Dict[str, ObjectRef]` where values
        can be `CountNode`, `DurationNode`, `PercentNode`.
    """
    return _ffi_api.ProfileFunction(
        mod, func_name, dev.device_type, dev.device_id, warmup_iters, collectors
    )


# We only enable this class when TVM is build with PAPI support
if _ffi.get_global_func("runtime.profiling.PAPIMetricCollector", allow_missing=True) is not None:

    @_ffi.register_object("runtime.profiling.PAPIMetricCollector")
    class PAPIMetricCollector(MetricCollector):
        """Collects performance counter information using the Performance
        Application Programming Interface (PAPI).
        """

        def __init__(self, metric_names: Optional[Dict[Device, Sequence[str]]] = None):
            """
            Parameters
            ----------
            metric_names : Optional[Dict[Device, Sequence[str]]]
                List of per-device metrics to collect. You can find a list of valid
                metrics by runing `papi_native_avail` from the command line.
            """
            metric_names = {} if metric_names is None else metric_names
            wrapped = dict()
            for dev, names in metric_names.items():
                wrapped[DeviceWrapper(dev)] = names
            self.__init_handle_by_constructor__(_ffi_api.PAPIMetricCollector, wrapped)


if _ffi.get_global_func("runtime.profiling.LikwidMetricCollector", allow_missing=True) is not None:

    @_ffi.register_object("runtime.profiling.LikwidMetricCollector")
    class LikwidMetricCollector(MetricCollector):
        """Collects performance counter information using the Likwid library."""

        def __init__(self, group: str):
            """
            Parameters
            ----------
            group : str
                Likwid group name to profile. Available groups can be listed
                with `likwid-perfctr -a`.
            """
            self.__init_handle_by_constructor__(_ffi_api.LikwidMetricCollector, group)

    @_ffi.register_object("runtime.profiling.LikwidMarkerMetricCollector")
    class LikwidMarkerMetricCollector(MetricCollector):
        """
        Marks profiling regions with a Likwid Marker so that it can be used to
        profile an application at runtime. To use, prefix your script like so:
        `likwid-perfctr -m -g MY_GROUP python3 my_script.py` (the `-m` flag is
        required). See `likwid-perfctr -h` for more information.
        """

        def __init__(self):
            self.__init_handle_by_constructor__(_ffi_api.LikwidMarkerMetricCollector)
