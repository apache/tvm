#!/usr/bin/env python

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
TVMC - TVM driver command-line interface
"""
import argparse
import logging
import sys

import tvm

from tvm.driver.tvmc import TVMCException, TVMCImportError
from tvm.driver.tvmc.config_options import (
    read_and_convert_json_into_dict,
    convert_config_json_to_cli,
)

REGISTERED_PARSER = []


def register_parser(make_subparser):
    """
    Utility function to register a subparser for tvmc.

    Functions decorated with `tvm.driver.tvmc.main.register_parser` will be invoked
    with a parameter containing the subparser instance they need to add itself to,
    as a parser.

    Example
    -------

        @register_parser
        def _example_parser(main_subparser):
            subparser = main_subparser.add_parser('example', help='...')
            ...

    """
    REGISTERED_PARSER.append(make_subparser)
    return make_subparser


def _main(argv):
    """TVM command line interface."""

    parser = argparse.ArgumentParser(
        prog="tvmc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="TVM compiler driver",
        epilog=__doc__,
        # Help action will be added later, after all subparsers are created,
        # so it doesn't interfere with the creation of the dynamic subparsers.
        add_help=False,
    )

    parser.add_argument("--config", default="default", help="configuration json file")
    config_arg, argv = parser.parse_known_args(argv)

    json_param_dict = read_and_convert_json_into_dict(config_arg)
    json_config_values = convert_config_json_to_cli(json_param_dict)

    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity")
    parser.add_argument("--version", action="store_true", help="print the version and exit")

    subparser = parser.add_subparsers(title="commands")
    for make_subparser in REGISTERED_PARSER:
        make_subparser(subparser, parser, json_config_values)

    # Finally, add help for the main parser.
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit.")

    args = parser.parse_args(argv)
    if args.verbose > 4:
        args.verbose = 4

    logging.getLogger("TVMC").setLevel(40 - args.verbose * 10)

    if args.version:
        sys.stdout.write("%s\n" % tvm.__version__)
        return 0

    if not hasattr(args, "func"):
        # In case no valid subcommand is provided, show usage and exit
        parser.print_help(sys.stderr)
        return 1

    try:
        return args.func(args)
    except TVMCImportError as err:
        sys.stderr.write(
            f'Package "{err}" is not installed. ' f'Hint: "pip install tlcpack[tvmc]".'
        )
        return 5
    except TVMCException as err:
        sys.stderr.write("Error: %s\n" % err)
        return 4


def main():
    sys.exit(_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
