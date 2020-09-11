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
# pylint: disable=invalid-name
"""Pick best log entries from a large file and store them to a small file"""

import argparse
import os
import logging
import warnings

from .. import autotvm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--act", type=str, choices=["pick-best"], required=True, help="The action")
    parser.add_argument("--i", type=str, help="The input file or directory", required=True)
    parser.add_argument("--o", type=str, help="The output file")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.act == "pick-best":
        if os.path.isfile(args.i):
            args.o = args.o or args.i + ".best.log"
            autotvm.record.pick_best(args.i, args.o)
        elif os.path.isdir(args.i):
            args.o = args.o or "best.log"
            tmp_filename = args.o + ".tmp"

            with open(tmp_filename, "w") as tmp_fout:
                for filename in os.listdir(args.i):
                    if filename.endswith(".log"):
                        try:
                            autotvm.record.pick_best(filename, tmp_fout)
                        except Exception:  # pylint: disable=broad-except
                            warnings.warn("Ignore invalid file %s" % filename)

            logging.info("Run final filter...")
            autotvm.record.pick_best(tmp_filename, args.o)
            os.remove(tmp_filename)
            logging.info("Output to %s ...", args.o)
        else:
            raise ValueError("Invalid input file: " + args.i)
    else:
        raise ValueError("Invalid action " + args.act)
