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

from typing import Union, Optional
import warnings
import sys

from tvm.ir import IRModule
from tvm.tir import PrimFunc


def cprint(printable: Union[IRModule, PrimFunc], style: Optional[str] = None) -> None:
    """
    Print highlighted TVM script string with Pygments
    Parameters
    ----------
    printable : Union[IRModule, PrimFunc]
        The TVM script to be printed
    style : str, optional
        Printing style, auto-detected if None.
    Notes
    -----
    The style parameter follows the Pygments style names or Style objects. Three
    built-in styles are extended: "light", "dark" and "ansi". By default, "light"
    will be used for notebook environment and terminal style will be "ansi" for
    better style consistency. As an fallback when the optional Pygment library is
    not installed, plain text will be printed with a one-time warning to suggest
    installing the Pygment library. Other Pygment styles can be found in
    https://pygments.org/styles/
    """

    try:
        # pylint: disable=import-outside-toplevel
        import pygments
        from pygments import highlight
        from pygments.lexers.python import Python3Lexer
        from pygments.formatters import Terminal256Formatter, HtmlFormatter
        from pygments.style import Style
        from pygments.token import Keyword, Name, Comment, String, Number, Operator
        from packaging import version

        if version.parse(pygments.__version__) < version.parse("2.4.0"):
            raise ImportError("Required Pygments version >= 2.4.0 but got " + pygments.__version__)
    except ImportError as err:
        with warnings.catch_warnings():
            warnings.simplefilter("once", UserWarning)
            install_cmd = sys.executable + ' -m pip install "Pygments>=2.4.0" --upgrade --user'
            warnings.warn(
                str(err)
                + "\n"
                + "To print highlighted TVM script, please install Pygments:\n"
                + install_cmd,
                category=UserWarning,
            )
        print(printable.script())
    else:

        class JupyterLight(Style):
            """A Jupyter-Notebook-like Pygments style configuration (aka. "light")"""

            background_color = ""
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

        class VSCDark(Style):
            """A VSCode-Dark-like Pygments style configuration (aka. "dark")"""

            background_color = ""
            styles = {
                Keyword: "bold #c586c0",
                Keyword.Type: "#82aaff",
                Keyword.Namespace: "#4ec9b0",
                Name.Class: "bold #569cd6",
                Name.Function: "bold #dcdcaa",
                Name.Decorator: "italic #fe4ef3",
                String: "#ce9178",
                Number: "#b5cea8",
                Operator: "#bbbbbb",
                Operator.Word: "#569cd6",
                Comment: "italic #6a9956",
            }

        class AnsiTerminalDefault(Style):
            """The default style for terminal display with ANSI colors (aka. "ansi")"""

            background_color = ""
            styles = {
                Keyword: "bold ansigreen",
                Keyword.Type: "nobold ansigreen",
                Name.Class: "bold ansiblue",
                Name.Function: "bold ansiblue",
                Name.Decorator: "italic ansibrightmagenta",
                String: "ansiyellow",
                Number: "ansibrightgreen",
                Operator: "bold ansimagenta",
                Operator.Word: "bold ansigreen",
                Comment: "italic ansibrightblack",
            }

        is_in_notebook = "ipykernel" in sys.modules  # in notebook env (support html display).

        if style is None:
            # choose style automatically according to the environment:
            style = JupyterLight if is_in_notebook else AnsiTerminalDefault
        elif style == "light":
            style = JupyterLight
        elif style == "dark":
            style = VSCDark
        elif style == "ansi":
            style = AnsiTerminalDefault

        if is_in_notebook:  # print with HTML display
            from IPython.display import display, HTML  # pylint: disable=import-outside-toplevel

            formatter = HtmlFormatter(style=JupyterLight)
            formatter.noclasses = True  # inline styles
            html = highlight(printable.script(), Python3Lexer(), formatter)
            display(HTML(html))
        else:
            print(highlight(printable.script(), Python3Lexer(), Terminal256Formatter(style=style)))
