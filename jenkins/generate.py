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

import json
import subprocess
import re
import jinja2
import argparse
from urllib import request
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import unittest



REPO_ROOT = Path(__file__).resolve().parent.parent
JENKINSFILE_TEMPLATE = REPO_ROOT / "jenkins" / "Jenkinsfile.j2"
JENKINSFILE = REPO_ROOT / "Jenkinsfile"


data = {
    "images": [
        { "name": "ci_arm", "platform": "ARM", },
        { "name": "ci_cpu", "platform": "CPU", },
        { "name": "ci_gpu", "platform": "CPU", },
        { "name": "ci_hexagon", "platform": "CPU", },
        { "name": "ci_i386", "platform": "CPU", },
        { "name": "ci_lint", "platform": "CPU", },
        { "name": "ci_qemu", "platform": "CPU", },
        { "name": "ci_wasm", "platform": "CPU", },
    ]
}


if __name__ == "__main__":
    help = "Regenerate Jenkinsfile from template"
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("--check", action="store_true", help="just verify the output didn't change")
    args = parser.parse_args()

    with open(JENKINSFILE) as f:
        content = f.read()

    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(REPO_ROOT),
        undefined=jinja2.StrictUndefined,
        lstrip_blocks=True,
        trim_blocks=True,
        keep_trailing_newline=True,
    )
    template = environment.get_template(str(JENKINSFILE_TEMPLATE.relative_to(REPO_ROOT)))
    new_content = template.render(**data)

    if args.check:
        tc = unittest.TestCase()
        tc.maxDiff = float("inf")
        tc.assertEqual(content, new_content)
    else:
        with open(JENKINSFILE, "w") as f:
            f.write(new_content)
        print(f"Wrote output to {JENKINSFILE.relative_to(REPO_ROOT)}")
