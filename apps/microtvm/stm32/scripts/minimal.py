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

"""
Minimal example - Invoke a model with the random data
"""

import sys
import argparse
from statistics import mean
from ai_runner import AiRunner

_DEFAULT = "serial"


def example(args):

    runner = AiRunner(debug=args.debug)

    if not runner.connect(args.desc):
        return 1

    print(runner, flush=True)

    runner.summary(level=args.verbosity)

    inputs = runner.generate_rnd_inputs(batch_size=args.batch)

    print("Invoking the model with random data (b={})..".format(inputs[0].shape[0]), flush=True)
    _, profile = runner.invoke(inputs)

    print("")
    print("host execution time      : {:.3f}ms".format(profile["debug"]["host_duration"]))
    print("number of samples        : {}".format(len(profile["c_durations"])))
    print("inference time by sample : {:.3f}ms (average)".format(mean(profile["c_durations"])))
    print("")

    runner.disconnect()

    return 0


def main():
    """ script entry point """

    parser = argparse.ArgumentParser(description="Minimal example")

    parser.add_argument(
        "--desc",
        "-d",
        metavar="STR",
        type=str,
        help="description for the connection",
        default=_DEFAULT,
    )
    parser.add_argument(
        "--batch", "-b", metavar="INT", type=int, help="number of sample", default=1
    )
    parser.add_argument("--debug", action="store_true", help="debug option")
    parser.add_argument(
        "--verbosity",
        "-v",
        nargs="?",
        const=1,
        type=int,
        choices=range(0, 3),
        help="set verbosity level",
        default=0,
    )

    args = parser.parse_args()

    return example(args)


if __name__ == "__main__":
    sys.exit(main())
