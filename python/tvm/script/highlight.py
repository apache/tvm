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
"""Highlight printed TVM script.
"""

from typing import Union

from pygments import highlight
from pygments.lexers import Python3Lexer
from pygments.formatters import Terminal256Formatter
from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Number, Operator

from tvm.ir import IRModule
from tvm.tir import PrimFunc


class VSCDark(Style):
    """A VSCode-Dark-like Pygments style configuration"""

    styles = {
        Keyword: "bold #c586c0",
        Keyword.Namespace: "#4ec9b0",
        Keyword.Type: "#82aaff",
        Name.Function: "bold #dcdcaa",
        Name.Class: "bold #569cd6",
        Name.Decorator: "italic #fe4ef3",
        String: "#ce9178",
        Number: "#b5cea8",
        Operator: "#bbbbbb",
        Operator.Word: "#569cd6",
        Comment: "italic #6a9956",
    }


class JupyterLight(Style):
    """A Jupyter-Notebook-like Pygments style configuration"""

    styles = {
        Keyword: "bold #008000",
        Keyword.Type: "nobold #008000",
        Name.Function: "#0000FF",
        Name.Class: "bold #0000FF",
        Name.Decorator: "#AA22FF",
        String: "#BA2121",
        Number: "#008000",
        Operator: "bold #AA22FF",
        Operator.Word: "bold #008000",
        Comment: "italic #007979",
    }


def cprint(printable: Union[IRModule, PrimFunc], style="light") -> None:
    """
    Print highlighted TVM script string with Pygments
    Parameters
    ----------
    printable : Union[IRModule, PrimFunc]
        The TVM script to be printed
    style : str, optional
        Style of the printed script
    Notes
    -----
    The style parameter follows the Pygments style names or Style objects. Two
    built-in styles are extended: "light" (default) and "dark". Other styles
    can be found in https://pygments.org/styles/
    """
    if style == "light":
        style = JupyterLight
    elif style == "dark":
        style = VSCDark
    print(highlight(printable.script(), Python3Lexer(), Terminal256Formatter(style=style)))
