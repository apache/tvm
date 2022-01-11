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
"""Testing utilities for the TensorIR schedule API"""
from typing import Union

from tvm.ir import IRModule, structural_equal
from tvm.tir import PrimFunc
from tvm.tir.schedule import Trace, Schedule


def verify_trace_roundtrip(
    sch: Schedule,
    mod: Union[PrimFunc, IRModule],
    *,
    debug_mask: Union[str, int] = "all",
) -> Schedule:
    """Serialize a traced schedule to JSON, then replay the JSON trace by applying to
    a fresh new schedule, verifying the reproducibility of scheduling.

    Parameters
    ----------
    sch : tir.Schedule
        The traced TensorIR schedule to be verified
    mod : Union[PrimFunc, IRModule]
        The IRModule or PrimFunc to construct the fresh new schedule
    debug_mask : Union[str, int]
        Do extra correctness checking after the class creation and each time
        after calling the Replace method.
        Possible choices of `debug_mask`:
        1) "all" - Turn on all the checks
        2) "none" - Turn off all the checks
        3) An integer - Turn on checks according to the bitmasks provided in ScheduleDebugMask
    """
    # Step 1. Serialize the trace to JSON
    trace = sch.trace
    assert trace is not None
    json_obj = trace.as_json()
    # Step 2. Apply the JSON trace to a new schedule, then check if it reproduces the scheduling
    new_sch = Schedule(mod=mod, debug_mask=debug_mask)
    Trace.apply_json_to_schedule(json_obj=json_obj, sch=new_sch)
    assert structural_equal(new_sch.mod, sch.mod)
    # Step 3. Check the consistency of the text format between the old and new traces
    py_repr = "\n".join(trace.as_python())
    new_py_repr = "\n".join(new_sch.trace.as_python())
    assert py_repr == new_py_repr
    # Step 4. Return the new schedule in case it could be useful
    return new_sch
