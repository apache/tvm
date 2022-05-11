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

import subprocess
import os
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class RelativePathFilter(logging.Filter):
    def filter(self, record):
        path = Path(record.pathname).resolve()
        record.relativepath = str(path.relative_to(REPO_ROOT))
        return True


def init_log():
    logging.basicConfig(
        format="[%(relativepath)s:%(lineno)d %(levelname)-1s] %(message)s", level=logging.INFO
    )

    # Flush on every log call (logging and then calling subprocess.run can make
    # the output look confusing)
    logging.root.handlers[0].addFilter(RelativePathFilter())
    logging.root.handlers[0].flush = sys.stderr.flush


class Sh:
    def __init__(self, env=None):
        self.env = os.environ.copy()
        if env is not None:
            self.env.update(env)

    def run(self, cmd: str, **kwargs):
        logging.info(f"+ {cmd}")
        if "check" not in kwargs:
            kwargs["check"] = True
        if "shell" not in kwargs:
            kwargs["shell"] = True
        if "env" not in kwargs:
            kwargs["env"] = self.env

        return subprocess.run(cmd, **kwargs)
