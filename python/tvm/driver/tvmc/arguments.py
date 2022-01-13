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
TVMC Argument Parsing
"""

import argparse

from tvm.driver.tvmc import TVMCException


class TVMCSuppressedArgumentParser(argparse.ArgumentParser):
    """
    A silent ArgumentParser class.
    This class is meant to be used as a helper for creating dynamic parsers in
    TVMC. It will create a "supressed" parser based on an existing one (parent)
    which does not include a help message, does not print a usage message (even
    when -h or --help is passed) and does not exit on invalid choice parse
    errors but rather throws a TVMCException so it can be handled and the
    dynamic parser construction is not interrupted prematurely.
    """

    def __init__(self, parent, **kwargs):
        # Don't add '-h' or '--help' options to the newly created parser. Don't print usage message.
        # 'add_help=False' won't supress existing '-h' and '--help' options from the parser (and its
        # subparsers) present in 'parent'. However that class is meant to be used with the main
        # parser, which is created with `add_help=False` - the help is added only later. Hence it
        # the newly created parser won't have help options added in its (main) root parser. The
        # subparsers in the main parser will eventually have help activated, which is enough for its
        # use in TVMC.
        super().__init__(parents=[parent], add_help=False, usage=argparse.SUPPRESS, **kwargs)

    def exit(self, status=0, message=None):
        # Don't exit on error when parsing the command line.
        # This won't catch all the errors generated when parsing tho. For instance, it won't catch
        # errors due to missing required arguments. But this will catch "error: invalid choice",
        # which is what it's necessary for its use in TVMC.
        raise TVMCException()
