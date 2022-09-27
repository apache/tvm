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
"""Default schedule rules"""
from typing import List, Tuple, Union

from tvm.meta_schedule import default_config
from tvm.meta_schedule.schedule_rule import ScheduleRule


def get_rules(kind: str, types: Union[type, Tuple[type, ...]]) -> List[ScheduleRule]:
    """Get default schedule rules"""
    # pylint: disable=protected-access
    if kind == "llvm":
        rules = default_config._DefaultLLVM.schedule_rules()
    elif kind == "cuda":
        rules = default_config._DefaultCUDA.schedule_rules()
    elif kind == "tensor_core":
        rules = default_config._DefaultCUDATensorCore.schedule_rules()
    else:
        raise NotImplementedError(f"{kind} is not supported")
    # pylint: enable=protected-access
    return [rule for rule in rules if isinstance(rule, types)]
