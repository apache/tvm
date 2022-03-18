#!/usr/bin/env python3
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

import os
import argparse
import re
import datetime
import subprocess
import json
from typing import Dict, Any, List
from pathlib import Path

from cmd_utils import REPO_ROOT, Sh

def core_prefix(name: str) -> str:
    delimiters = [" ", "-", "_"]
    prefix = name
    for d in delimiters:
        prefix = prefix.split(d)[0]
    return prefix


def list_coredumps() -> List[Path]:
    p = Sh().run("sysctl kernel.core_pattern", stdout=subprocess.PIPE, encoding="utf-8")
    output = p.stdout.strip()
    core_pattern = output.split("= |")[1]
    core_pattern = Path(core_pattern)
    core_dir = core_pattern.parent
    prefix = core_prefix("core-%d-%d")

    print(f"Checking for core dumps matching '{prefix}*' in {core_dir}")
    return list(core_dir.glob(f"{prefix}*"))


if __name__ == "__main__":
    help = "List available coredumps"
    parser = argparse.ArgumentParser(description=help)
    args = parser.parse_args()
    
    cores = list_coredumps()
    if len(cores) == 0:
        print("No coredumps found")
    
    print("Found coredumps:", cores)
    exit(1)
  