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

import os
import sys
import warnings
from typing import Optional, Union

from tvm.ir import IRModule
from tvm.tir import PrimFunc


def cprint(
    printable: Union[IRModule, PrimFunc, str],
    style: Optional[str] = None,
    black_format: bool = True,
) -> None:
    """Print TVMScript string with Pygments highlight and Black auto-formatting.

    Parameters
    ----------
    printable : Union[IRModule, PrimFunc, str]

        The TVMScript to be printed

    style : str, optional

        Pygmentize printing style, auto-detected if None.

    black_format: bool

        If true (default), use the formatter Black to format the TVMScript

    Notes
    -----

    The style parameter follows the Pygments style names or Style objects. Three
    built-in styles are extended: "light", "dark" and "ansi". By default, "light"
    will be used for notebook environment and terminal style will be "ansi" for
    better style consistency. As an fallback when the optional Pygment library is
    not installed, plain text will be printed with a one-time warning to suggest
    installing the Pygment library. Other Pygment styles can be found in
    https://pygments.org/styles/

    The default pygmentize style can also be set with the environment
    variable "TVM_PYGMENTIZE_STYLE".
    """
    if isinstance(printable, (IRModule, PrimFunc)):
        printable = printable.script()

    if black_format:
        printable = _format(printable)

    is_in_notebook = "ipykernel" in sys.modules  # in notebook env (support html display).

    style = _get_pygments_style(style, is_in_notebook)

    if style is None:
        print(printable)
        return

    # pylint: disable=import-outside-toplevel
    from pygments import highlight
    from pygments.formatters import HtmlFormatter, Terminal256Formatter
    from pygments.lexers.python import Python3Lexer

    if is_in_notebook:
        from IPython import display  # pylint: disable=import-outside-toplevel

        formatter = HtmlFormatter(style=style)
        formatter.noclasses = True  # inline styles
        html = highlight(printable, Python3Lexer(), formatter)
        display.display(display.HTML(html))
    else:
        print(highlight(printable, Python3Lexer(), Terminal256Formatter(style=style)))


def _format(code_str: str) -> str:
    """Format a code string using Black.

    Parameters
    ----------
    code_str: str

        The string containing Python/TVMScript code to format

    Returns
    -------
    formatted: str

        The formatted Python/TVMScript code
    """
    try:
        # pylint: disable=import-outside-toplevel
        import black
    except ImportError as err:
        with warnings.catch_warnings():
            warnings.simplefilter("once", UserWarning)
            install_cmd = sys.executable + ' -m pip install "black==22.3.0" --upgrade --user'
            warnings.warn(
                str(err)
                + "\n"
                + "To print formatted TVM script, please install the formatter 'Black':\n"
                + install_cmd,
                category=UserWarning,
            )
        return code_str
    else:
        return black.format_str(code_str, mode=black.FileMode())


def _get_pygments_style(
    style: Optional[str], is_in_notebook: bool
) -> Optional[Union["pygments.style.Style", str]]:
    """Select a pygments style to use

    Parameters
    ----------
    style: str

        The style specifier to use.  If None, auto-select a style.

    is_in_notebook: bool

        Whether python is currently running in a jupyter notebook.
        Used for automatic selection.

    Returns
    -------
    style: Optional[Union['pygments.style.Style',str]]

        If pygments is installed, the style object or string, suitable
        for use as the "style" argument to pygments formatters.  If
        pygments is not installed, returns None.

    """
    try:
        # pylint: disable=import-outside-toplevel
        import pygments
        from packaging import version
        from pygments.style import Style
        from pygments.token import Comment, Keyword, Name, Number, Operator, String

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
        return None

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

    if style == "light":
        return JupyterLight
    elif style == "dark":
        return VSCDark
    elif style == "ansi":
        return AnsiTerminalDefault

    if style is not None:
        return style

    style_from_environment = os.environ.get("TVM_PYGMENTIZE_STYLE", "").strip()
    if style_from_environment:
        return style_from_environment

    if is_in_notebook:
        return JupyterLight

    return AnsiTerminalDefault
