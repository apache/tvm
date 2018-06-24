"""Shared functions and classes for tvmdbg command-line interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import os
import six

import numpy as np

from tvm.tools.debug.ui import command_parser
from tvm.tools.debug.ui import ui_common
from tvm.tools.debug.ui import tensor_data
from tvm.tools.debug.util import common

RL = ui_common.RichLine

# Default threshold number of elements above which ellipses will be used
# when printing the value of the tensor.
DEFAULT_NDARRAY_DISPLAY_THRESHOLD = 2000

COLOR_BLACK = "black"
COLOR_BLUE = "blue"
COLOR_CYAN = "cyan"
COLOR_GRAY = "gray"
COLOR_GREEN = "green"
COLOR_MAGENTA = "magenta"
COLOR_RED = "red"
COLOR_WHITE = "white"
COLOR_YELLOW = "yellow"

TIME_UNIT_US = "us"
TIME_UNIT_MS = "ms"
TIME_UNIT_S = "s"
TIME_UNITS = [TIME_UNIT_US, TIME_UNIT_MS, TIME_UNIT_S]


def bytes_to_readable_str(num_bytes, include_b=False):
    """Generate a human-readable string representing number of bytes.

    The units B, kB, MB and GB are used.

    Args:
      num_bytes: (`int` or None) Number of bytes.
      include_b: (`bool`) Include the letter B at the end of the unit.

    Returns:
      (`str`) A string representing the number of bytes in a human-readable way,
        including a unit at the end.
    """

    if num_bytes is None:
        return str(num_bytes)
    if num_bytes < 1024:
        result = "%d" % num_bytes
    elif num_bytes < 1048576:
        result = "%.2fk" % (num_bytes / 1024.0)
    elif num_bytes < 1073741824:
        result = "%.2fM" % (num_bytes / 1048576.0)
    else:
        result = "%.2fG" % (num_bytes / 1073741824.0)

    if include_b:
        result += "B"
    return result


def time_to_readable_str(value_us, force_time_unit=None):
    """Convert time value to human-readable string.

    Args:
      value_us: time value in microseconds.
      force_time_unit: force the output to use the specified time unit. Must be
        in TIME_UNITS.

    Returns:
      Human-readable string representation of the time value.

    Raises:
      ValueError: if force_time_unit value is not in TIME_UNITS.
    """
    if not value_us:
        return "0"
    if force_time_unit:
        if force_time_unit not in TIME_UNITS:
            raise ValueError("Invalid time unit: %s" % force_time_unit)
        order = TIME_UNITS.index(force_time_unit)
        time_unit = force_time_unit
        return "{:.10g}{}".format(value_us / math.pow(10.0, 3 * order), time_unit)
    else:
        order = min(len(TIME_UNITS) - 1, int(math.log(value_us, 10) / 3))
        time_unit = TIME_UNITS[order]
        return "{:.3g}{}".format(value_us / math.pow(10.0, 3 * order), time_unit)


def parse_ranges_highlight(ranges_string):
    """Process ranges highlight string.

    Args:
      ranges_string: (str) A string representing a numerical range of a list of
        numerical ranges. See the help info of the -r flag of the view_tensor
        command for more details.

    Returns:
      An instance of tensor_data.HighlightOptions, if range_string is a valid
        representation of a range or a list of ranges.
    """

    ranges = None

    def ranges_filter(ndarray):
        """Determine which elements of the tensor to be highlighted.

        Args:
          ndarray: tensor

        Returns: A boolean ndarray of the same shape as ndarray.
        """

        ret_val = np.zeros(ndarray.shape, dtype=bool)
        for range_start, range_end in ranges:
            ret_val = np.logical_or(ret_val,
                                    np.logical_and(ndarray >= range_start, ndarray <= range_end))

        return ret_val

    if ranges_string:
        ranges = command_parser.parse_ranges(ranges_string)
        return tensor_data.HighlightOptions(
            ranges_filter, description=ranges_string)
    return None

def get_np_printoptions_frm_scr(screen_info):
    """Retreive np.set_printoptions() to set the text format for display numpy ndarrays.

    Args:
      screen_info: Optional dict input containing screen information such as
            cols.

    Returns: A dictionary of keyword.
    """

    if screen_info and "cols" in screen_info:
        return {"linewidth": screen_info["cols"]}
    return {}


def format_tensor(tensor,
                  tensor_name,
                  np_printoptions,
                  print_all=False,
                  tensor_slicing=None,
                  highlight_options=None,
                  include_numeric_summary=False,
                  write_path=None):
    """Generate formatted str to represent a tensor or its slices.

    Args:
      tensor: (numpy ndarray) The tensor value.
      tensor_name: (str) Name of the tensor, e.g., the tensor's debug watch key.
      np_printoptions: (dict) Numpy tensor formatting options.
      print_all: (bool) Whether the tensor is to be displayed in its entirety,
        instead of printing ellipses, even if its number of elements exceeds
        the default numpy display threshold.
        (Note: Even if this is set to true, the screen output can still be cut
         off by the UI frontend if it consist of more lines than the frontend
         can handle.)
      tensor_slicing: (str or None) Slicing of the tensor, e.g., "[:, 1]". If
        None, no slicing will be performed on the tensor.
      highlight_options: (tensor_data.HighlightOptions) options to highlight
        elements of the tensor. See the doc of tensor_data.format_tensor()
        for more details.
      include_numeric_summary: Whether a text summary of the numeric values (if
        applicable) will be included.
      write_path: A path to save the tensor value (after any slicing) to
        (optional). `numpy.save()` is used to save the value.

    Returns:
      An instance of `ui_common.RichTextLines` representing the
      (potentially sliced) tensor.
    """

    if tensor_slicing:
        # Validate the indexing.
        value = command_parser.evaluate_tensor_slice(tensor, tensor_slicing)
        sliced_name = tensor_name + tensor_slicing
    else:
        value = tensor
        sliced_name = tensor_name

    auxiliary_message = None
    if write_path:
        with open(write_path, "wb") as output_file:
            np.save(output_file, value)
        line = ui_common.RichLine("Saved value to: ")
        line += ui_common.RichLine(write_path, font_attr="bold")
        line += " (%sB)" % bytes_to_readable_str(os.stat(write_path).st_size)
        auxiliary_message = ui_common.rich_text_lines_frm_line_list(
            [line, ui_common.RichLine("")])

    if print_all:
        np_printoptions["threshold"] = value.size
    else:
        np_printoptions["threshold"] = DEFAULT_NDARRAY_DISPLAY_THRESHOLD

    return tensor_data.format_tensor(
        value,
        sliced_name,
        include_metadata=True,
        include_numeric_summary=include_numeric_summary,
        auxiliary_message=auxiliary_message,
        np_printoptions=np_printoptions,
        highlight_options=highlight_options)


def error(msg):
    """Generate a RichTextLines output for error.

    Args:
      msg: (str) The error message.

    Returns:
      (ui_common.RichTextLines) A representation of the error message
        for screen output.
    """

    return ui_common.rich_text_lines_frm_line_list([
        RL("ERROR: " + msg, COLOR_RED)])


def _recommend_command(command, description, indent=2, create_link=False):
    """Generate a RichTextLines object that describes a recommended command.

    Args:
      command: (str) The command to recommend.
      description: (str) A description of what the command does.
      indent: (int) How many spaces to indent in the beginning.
      create_link: (bool) Whether a command link is to be applied to the command
        string.

    Returns:
      (RichTextLines) Formatted text (with font attributes) for recommending the
        command.
    """

    indent_str = " " * indent

    if create_link:
        font_attr = [ui_common.MenuItem("", command), "bold"]
    else:
        font_attr = "bold"

    lines = [RL(indent_str) + RL(command, font_attr) + ":",
             indent_str + "   " + description]

    return ui_common.rich_text_lines_frm_line_list(lines)


def get_tvmdbg_logo():
    """Make an ASCII representation of the tvmdbg logo."""

    lines = []
    return ui_common.RichTextLines(lines)


_HORIZONTAL_BAR = " ======================================"


def get_run_start_intro(graph_node_count,
                        outputs,
                        input_dict,
                        is_callable_runner=False):
    """Generate formatted intro for run-start UI.

    Args:
      run_call_count: (int) Run call counter.
      outputs: Outputs of the `GraphRuntime.run()` call. See doc of `GraphRuntime.run()`
        for more details.
      input_dict: Inputs to the `GraphRuntime.run()` call. See doc of `GraphRuntime.run()`
        for more details.
      tensor_filters: (dict) A dict from tensor-filter name to tensor-filter
        callable.
      is_callable_runner: (bool) whether a runner returned by
          GraphRuntime.make_callable is being run.

    Returns:
      (RichTextLines) Formatted intro message about the `GraphRuntime.run()` call.
    """

    output_lines = common.get_flattened_names(outputs)

    if not input_dict:
        input_dict_lines = [ui_common.RichLine("  (Empty)")]
    else:
        input_dict_lines = []
        for input_key in input_dict:
            input_key_name = common.get_graph_element_name(input_key)
            input_dict_line = ui_common.RichLine("  ")
            input_dict_line += ui_common.RichLine(
                input_key_name,
                ui_common.MenuItem(None, "pi '%s'" % input_key_name))
            # Surround the name string with quotes, because input_key_name may contain
            # spaces in some cases, e.g., SparseTensors.
            input_dict_lines.append(input_dict_line)
    input_dict_lines = ui_common.rich_text_lines_frm_line_list(
        input_dict_lines)

    out = ui_common.RichTextLines(_HORIZONTAL_BAR)
    out.append("")
    out.append("")
    out.append(" Choose any of the below option to continue...")
    out.append(" ---------------------------------------------")

    out.extend(
        _recommend_command(
            "run",
            "Run the NNVM graph with debug",
            create_link=True))
    out.extend(
        _recommend_command(
            "run -nodebug",
            "Run the NNVM graph without debug",
            create_link=True))

    out.append("")
    out.append(_HORIZONTAL_BAR)
    if is_callable_runner:
        out.append(" Running a runner returned by GraphRuntime.make_callable()")
    else:
        out.append("")
        out.append(" TVM Graph details")# % run_call_count)
        out.append(" -----------------")
        out.append("")
        out.append(" Node count:")
        out.append("  " + str(graph_node_count))
        out.append("")
        out.append(" Input(s):")
        out.extend(input_dict_lines)
        out.append("")
        out.append(" Output(s):")
        out.extend(ui_common.RichTextLines(
            ["  " + line for line in output_lines]))
        out.append("")
    out.append(_HORIZONTAL_BAR)

    # Make main menu for the run-start intro.
    menu = ui_common.Menu()
    menu.append(ui_common.MenuItem("run", "run"))
    out.annotations[ui_common.MAIN_MENU_KEY] = menu

    return out


def get_run_short_description(run_call_count,
                              outputs,
                              input_dict,
                              is_callable_runner=False):
    """Get a short description of the run() call.

    Args:
      run_call_count: (int) Run call counter.
      outputs: Outputs of the `GraphRuntime.run()` call. See doc of `GraphRuntime.run()`
        for more details.
      input_dict: Inputs to the `GraphRuntime.run()` call. See doc of `GraphRuntime.run()`
        for more details.
      is_callable_runner: (bool) whether a runner returned by
          GraphRuntime.make_callable is being run.

    Returns:
      (str) A short description of the run() call, including information about
        the output(s) and input(s).
    """
    if is_callable_runner:
        return "runner from make_callable()"

    description = "run #%d: " % run_call_count

    if ';' not in outputs:
        description += "1 input (%s); " % common.get_graph_element_name(outputs)
    else:
        # Could be (nested) list, tuple, dict or namedtuple.
        num_outputs = len(common.get_flattened_names(outputs))
        if num_outputs > 1:
            description += "%d outputs; " % num_outputs
        else:
            description += "%d output; " % num_outputs

    if not input_dict:
        description += "0 inputs"
    else:
        if len(input_dict) == 1:
            for key in input_dict:
                description += "1 input (%s)" % (
                    key if isinstance(key, six.string_types) or not hasattr(key, "name")
                    else key.name)
        else:
            description += "%d inputs" % len(input_dict)

    return description


def get_error_intro(tvm_error):
    """Generate formatted intro for TVM run-time error.

    Args:
      tvm_error: (errors.OpError) TVM run-time error object.

    Returns:
      (RichTextLines) Formatted intro message about the run-time OpError, with
        sample commands for debugging.
    """

    op_name = tvm_error.op.name

    intro_lines = [
        "--------------------------------------",
        RL("!!! An error occurred during the run !!!", "blink"),
        "",
        "You may use the following commands to debug:",
    ]

    out = ui_common.rich_text_lines_frm_line_list(intro_lines)

    out.extend(
        _recommend_command("nd -a -d -t %s" % op_name,
                           "Inspect information about the failing op.",
                           create_link=True))
    out.extend(
        _recommend_command("gi -r %s" % op_name,
                           "List inputs to the failing op, recursively.",
                           create_link=True))

    out.extend(
        _recommend_command(
            "lg",
            "List all graphnodes dumped during the failing run() call.",
            create_link=True))

    more_lines = [
        "",
        "Op name:    " + op_name,
        "Error type: " + str(type(tvm_error)),
        "",
        "Details:",
        str(tvm_error),
        "",
        "WARNING: Using client GraphDef due to the error, instead of "
        "executor GraphDefs.",
        "--------------------------------------",
        "",
    ]

    out.extend(ui_common.RichTextLines(more_lines))

    return out
