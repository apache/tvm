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
# pylint: disable=invalid-name,unused-argument
"""The default schedule used by various operators"""
import tvm
from tvm import te


def default_schedule(outs, auto_inline):
    """Default schedule for llvm."""
    target = tvm.target.Target.current(allow_none=False)
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    if target.id.name not in ("llvm", "c"):
        raise RuntimeError("schedule not registered for '%s'" % target)
    s = te.create_schedule([x.op for x in outs])
    if auto_inline:
        x = outs[0]
        te.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
    return s
