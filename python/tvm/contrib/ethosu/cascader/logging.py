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
"""A class to hold logging information about the cascader"""
from typing import Tuple
import datetime
import json
import os
import math


class Logging:
    """Cascader logging class"""

    def __init__(self):
        self.min_memory_usage = 0
        self.max_memory_usage = 0
        self.min_cycles = 0
        self.max_cycles = 0

        self.selected_proposal_idx = -1
        self.proposals = {}
        self.cascader_runtime = 0

    def add_proposal(self, idx: int, memory_usage: int, cycles: int):
        self.proposals[idx] = {"memory_usage": memory_usage, "cycles": cycles}

    def get_extreme_points(self) -> Tuple[int, int, int, int]:
        min_cycles, min_mem_usage = math.inf, math.inf
        max_cycles, max_mem_usage = 0, 0
        for proposal in self.proposals.values():
            min_mem_usage = min(proposal["memory_usage"], min_mem_usage)
            max_mem_usage = max(proposal["memory_usage"], max_mem_usage)
            min_cycles = min(proposal["cycles"], min_cycles)
            max_cycles = max(proposal["cycles"], max_cycles)

        return min_mem_usage, max_mem_usage, min_cycles, max_cycles

    def dump_json(self):
        min_mem_usage, max_mem_usage, min_cycles, max_cycles = self.get_extreme_points()
        with open(os.getcwd() + "/cascader_log.json", "w") as json_file:
            print(
                json.dumps(
                    {
                        "date": f"{datetime.datetime.now()}",
                        "cascader_runtime": self.cascader_runtime,
                        "min_cycles": min_cycles,
                        "max_cycles": max_cycles,
                        "min_memory_usage": min_mem_usage,
                        "max_memory_usage": max_mem_usage,
                        "selected_proposal": self.selected_proposal_idx,
                        "proposals": self.proposals,
                    },
                    indent=2,
                ),
                file=json_file,
            )
