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
import pytest
import io
import argparse

from contextlib import redirect_stdout


class NodeidsCollector:
    def pytest_collection_modifyitems(self, items):
        self.nodeids = [item.nodeid for item in items]


def main(folder):
    collector = NodeidsCollector()
    f = io.StringIO()
    with redirect_stdout(f):
        pytest.main(["-qq", "--collect-only", folder], plugins=[collector])
    for nodeid in collector.nodeids:
        print(nodeid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List pytest nodeids for a folder")
    parser.add_argument("--folder", required=True, help="test folder to inspect")
    args = parser.parse_args()
    main(args.folder)
