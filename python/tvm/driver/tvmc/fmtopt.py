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
Utils to format help text for project options.
"""
from textwrap import TextWrapper


# Maximum column length for accommodating option name and its choices.
# Help text is placed after it in a new line.
MAX_OPTNAME_CHOICES_TEXT_COL_LEN = 80


# Maximum column length for accommodating help text.
# 0 turns off formatting for the help text.
MAX_HELP_TEXT_COL_LEN = 0


# Justification of help text placed below option name + choices text.
HELP_TEXT_JUST = 2


def format_option(option_text, help_text, default_text, required=True):
    """Format option name, choices, and default text into a single help text.

    Parameters
    ----------
    options_text: str
        String containing the option name and option's choices formatted as:
        optname={opt0, opt1, ...}

    help_text: str
        Help text string.

    default_text: str
        Default text string.

    required: bool
        Flag that controls if a "(required)" text mark needs to be added to the final help text to
        inform if the option is a required one.

    Returns
    -------
    help_text_just: str
       Single justified help text formatted as:
       optname={opt0, opt1, ... }
         HELP_TEXT. "(required)" | "Defaults to 'DEFAULT'."

    """
    optname, choices_text = option_text.split("=", 1)

    # Prepare optname + choices text chunck.

    optname_len = len(optname)
    wrapper = TextWrapper(width=MAX_OPTNAME_CHOICES_TEXT_COL_LEN - optname_len)
    choices_lines = wrapper.wrap(choices_text)

    # Set first choices line which merely appends to optname string.
    # No justification is necessary for the first line since first
    # line was wrapped based on MAX_OPTNAME_CHOICES_TEXT_COL_LEN - optname_len,
    # i.e. considering optname_len, hence only append justified choices_lines[0] line.
    choices_just_lines = [optname + "=" + choices_lines[0]]

    # Justify the remaining lines based on first optname + '='.
    for line in choices_lines[1:]:
        line_len = len(line)
        line_just = line.rjust(
            optname_len + 1 + line_len
        )  # add 1 to align after '{' in the line above
        choices_just_lines.append(line_just)

    choices_text_just_chunk = "\n".join(choices_just_lines)

    # Prepare help text chunck.

    help_text = help_text[0].lower() + help_text[1:]
    if MAX_HELP_TEXT_COL_LEN > 0:
        wrapper = TextWrapper(width=MAX_HELP_TEXT_COL_LEN)
        help_text_lines = wrapper.wrap(help_text)
    else:
        # Don't format help text.
        help_text_lines = [help_text]

    help_text_just_lines = []
    for line in help_text_lines:
        line_len = len(line)
        line_just = line.rjust(HELP_TEXT_JUST + line_len)
        help_text_just_lines.append(line_just)

    help_text_just_chunk = "\n".join(help_text_just_lines)

    # An option might be required for one method but optional for another one.
    # If the option is required for one method it means there is no default for
    # it when used in that method, hence suppress default text in that case.
    if default_text and not required:
        help_text_just_chunk += " " + default_text

    if required:
        help_text_just_chunk += " (required)"

    help_text_just = choices_text_just_chunk + "\n" + help_text_just_chunk
    return help_text_just
