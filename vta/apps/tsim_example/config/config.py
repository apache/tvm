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

import os.path as osp
import sys
import json
import argparse

cur = osp.abspath(osp.dirname(__file__))
cfg = json.load(open(osp.join(cur, 'config.json')))

def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--get-target", action="store_true",
                        help="Get target language, i.e. verilog or chisel")
    parser.add_argument("--get-top-name", action="store_true",
                        help="Get hardware design top name")
    parser.add_argument("--get-build-name", action="store_true",
                        help="Get build folder name")
    parser.add_argument("--get-use-trace", action="store_true",
                        help="Get use trace")
    parser.add_argument("--get-trace-name", action="store_true",
                        help="Get trace filename")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.get_target:
        print(cfg['TARGET'])

    if args.get_top_name:
        print(cfg['TOP_NAME'])

    if args.get_build_name:
        print(cfg['BUILD_NAME'])

    if args.get_use_trace:
        print(cfg['USE_TRACE'])

    if args.get_trace_name:
        print(cfg['TRACE_NAME'])

if __name__ == "__main__":
    main()
