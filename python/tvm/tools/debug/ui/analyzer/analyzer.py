# pylint: disable=too-many-lines
"""CLI Backend for the Analyzer Part of the Debugger.

The analyzer performs post hoc analysis of dumped intermediate tensors and
graph structure information from debugged GraphRuntime.debug_run() calls.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import curses
import re

from tvm.tools.debug.ui import ui_config
from tvm.tools.debug.ui import ui_shared
from tvm.tools.debug.ui import command_parser
from tvm.tools.debug.ui import ui_common
from tvm.tools.debug.ui import ui_factory
from tvm.tools.debug.util import graph_dump


RL = ui_common.RichLine

# String constants for the depth-dependent hanging indent at the beginning
# of each line.
HANG_UNFINISHED = "|  "  # Used for unfinished recursion depths.
HANG_FINISHED = "   "
HANG_SUFFIX = "|- "

# String constant for displaying depth and op type.
DEPTH_TEMPLATE = "(%d) "
OP_TYPE_TEMPLATE = "[%s] "

# String constants for control inputs/outputs, etc.
CTRL_LABEL = "(Ctrl) "
ELLIPSIS = "..."

SORT_TENSORS_BY_TIMESTAMP = "timestamp"
SORT_TENSORS_BY_DUMP_SIZE = "dump_size"
SORT_TENSORS_BY_OP_TYPE = "op_type"
SORT_TENSORS_BY_INPUT_SIZE = "input_size"
SORT_TENSORS_BY_OUTPUT_SIZE = "output_size"
SORT_TENSORS_BY_TENSOR_NAME = "tensor_name"


def _add_main_menu(output,
                   node_name=None,
                   enable_list_graphnodes=True,
                   enable_node_details=True,
                   enable_view_tensor=True,
                   enable_graphnode_inputs=True,
                   enable_graphnode_outputs=True):
    """Generate main menu for the screen output from a command.

    Args:
      output: (ui_common.RichTextLines) the output object to modify.
      node_name: (str or None) name of the node involved (if any). If None,
        the menu items node_details, graphnode_inputs and graphnode_outputs will be
        automatically disabled, overriding the values of arguments
        enable_node_details, enable_graphnode_inputs and enable_graphnode_outputs.
      enable_list_graphnodes: (bool) whether the list_tensor menu item will be
        enabled.
      enable_node_details: (bool) whether the node_details item will be enabled.
      enable_view_tensor: (bool) whether the view_tensor item will be enabled.
      enable_graphnode_inputs: (bool) whether the item graphnode_inputs will be enabled.
      enable_graphnode_outputs: (bool) whether the item graphnode_outputs will be enabled.
    """

    menu = ui_common.Menu()

    menu.append(
        ui_common.MenuItem(
            "list_graphnodes", "list_graphnodes", enabled=enable_list_graphnodes))

    if node_name:
        menu.append(
            ui_common.MenuItem(
                "node_details",
                "node_details -a -d -t %s" % node_name,
                enabled=enable_node_details))
        menu.append(
            ui_common.MenuItem(
                "view_tensor",
                "view_tensor %s" % node_name,
                enabled=enable_view_tensor))
        menu.append(
            ui_common.MenuItem(
                "graphnode_inputs",
                "graphnode_inputs -r %s" % node_name,
                enabled=enable_graphnode_inputs))
        menu.append(
            ui_common.MenuItem(
                "graphnode_outputs",
                "graphnode_outputs -r %s" % node_name,
                enabled=enable_graphnode_outputs))
    else:
        menu.append(
            ui_common.MenuItem(
                "node_details", None, enabled=False))
        menu.append(
            ui_common.MenuItem("view_tensor", None, enabled=False))
        menu.append(
            ui_common.MenuItem("graphnode_inputs", None, enabled=False))
        menu.append(
            ui_common.MenuItem("graphnode_outputs", None, enabled=False))

    #menu.append(
    #    ui_common.MenuItem("home", "home"))
    #menu.append(
    #    ui_common.MenuItem("help", "help"))

    output.annotations[ui_common.MAIN_MENU_KEY] = menu

class DebugAnalyzer(object):
    """Analyzer for debug data from dump directories."""

    _TIMESTAMP_COLUMN_HEAD = "Time (us)"
    _DUMP_SIZE_COLUMN_HEAD = "Size (B)"
    _OP_TYPE_COLUMN_HEAD = "Op type"
    _OP_NUMBER_INPUTS_HEAD = "Input(s)"
    _OP_NUMBER_OUTPUTS_HEAD = "Output(s)"
    _TENSOR_NAME_COLUMN_HEAD = "Node name"

    def __init__(self, debug_dump, config):
        """DebugAnalyzer constructor.

        Args:
          debug_dump: A DebugDumpDir object.
          config: A `ui_config.CLIConfig` object that carries user-facing
            configurations.
        """
        self._debug_dump = debug_dump

        # Initialize tensor filters state.
        self._tensor_filters = {}

        self._build_argument_parsers(config)
        config.set_callback("graph_recursion_depth",
                            self._build_argument_parsers)

    def _build_argument_parsers(self, config):
        """Build argument parsers for DebugAnalayzer.

        Args:
          config: A `ui_config.CLIConfig` object.

        Returns:
          A dict mapping command handler name to `ArgumentParser` instance.
        """
        # Argument parsers for command handlers.
        self._arg_parsers = {}

        # Parser for list_graphnodes.
        arg_p = argparse.ArgumentParser(
            description="List dumped intermediate tensors.",
            usage=argparse.SUPPRESS)
        arg_p.add_argument(
            "-f",
            "--tensor_filter",
            dest="tensor_filter",
            type=str,
            default="",
            help="List only Tensors passing the filter of the specified name")
        arg_p.add_argument(
            "-n",
            "--node_name_filter",
            dest="node_name_filter",
            type=str,
            default="",
            help="filter node name by regex.")
        arg_p.add_argument(
            "-t",
            "--op_type_filter",
            dest="op_type_filter",
            type=str,
            default="",
            help="filter op type by regex.")
        arg_p.add_argument(
            "-s",
            "--sort_by",
            dest="sort_by",
            type=str,
            default=SORT_TENSORS_BY_TIMESTAMP,
            help=("the field to sort the data by: (%s | %s | %s | %s | %s | %s)" %
                  (SORT_TENSORS_BY_TENSOR_NAME, SORT_TENSORS_BY_TIMESTAMP,
                   SORT_TENSORS_BY_DUMP_SIZE, SORT_TENSORS_BY_INPUT_SIZE,
                   SORT_TENSORS_BY_OUTPUT_SIZE, SORT_TENSORS_BY_OP_TYPE)))
        arg_p.add_argument(
            "-r",
            "--reverse",
            dest="reverse",
            action="store_true",
            help="sort the data in reverse (descending) order")
        self._arg_parsers["list_graphnodes"] = arg_p

        # Parser for node_details.
        arg_p = argparse.ArgumentParser(
            description="Show information about a node.", usage=argparse.SUPPRESS)
        arg_p.add_argument(
            "node_name",
            type=str,
            help="Name of the node or an associated tensor.")
        arg_p.add_argument(
            "-a",
            "--attributes",
            dest="attributes",
            action="store_true",
            help="Also list attributes of the node.")
        arg_p.add_argument(
            "-d",
            "--dumps",
            dest="dumps",
            action="store_true",
            help="Also list dumps available from the node.")
        arg_p.add_argument(
            "-t",
            "--traceback",
            dest="traceback",
            action="store_true",
            help="Also include the traceback of the node's creation "
                 "(if available in Python).")
        self._arg_parsers["node_details"] = arg_p

        # Parser for graphnode_inputs.
        arg_p = argparse.ArgumentParser(
            description="Show inputs to a node.", usage=argparse.SUPPRESS)
        arg_p.add_argument(
            "node_name",
            type=str,
            help="Name of the node or an output tensor from the node.")
        arg_p.add_argument(
            "-d",
            "--depth",
            dest="depth",
            type=int,
            default=config.get("graph_recursion_depth"),
            help="Maximum depth of recursion used when showing the input tree.")
        arg_p.add_argument(
            "-r",
            "--recursive",
            dest="recursive",
            action="store_true",
            help="Show inputs to the node recursively, i.e., the input tree.")
        arg_p.add_argument(
            "-t",
            "--op_type",
            action="store_true",
            help="Show op types of input nodes.")
        self._arg_parsers["graphnode_inputs"] = arg_p

        # Parser for graphnode_outputs.
        arg_p = argparse.ArgumentParser(
            description="Show the nodes that receive the outputs of given node.",
            usage=argparse.SUPPRESS)
        arg_p.add_argument(
            "node_name",
            type=str,
            help="Name of the node or an output tensor from the node.")
        arg_p.add_argument(
            "-d",
            "--depth",
            dest="depth",
            type=int,
            default=config.get("graph_recursion_depth"),
            help="Maximum depth of recursion used when showing the output tree.")
        arg_p.add_argument(
            "-r",
            "--recursive",
            dest="recursive",
            action="store_true",
            help="Show recipients of the node recursively, i.e., the output "
                 "tree.")
        arg_p.add_argument(
            "-t",
            "--op_type",
            action="store_true",
            help="Show op types of recipient nodes.")
        self._arg_parsers["graphnode_outputs"] = arg_p

        # Parser for view_tensor.
        self._arg_parsers["view_tensor"] = (
            command_parser.get_view_tensor_argparser(
                "Print the value of a dumped tensor."))


    def add_tensor_filter(self, filter_name, filter_callable):
        """Add a tensor filter.

        A tensor filter is a named callable of the signature:
          filter_callable(dump_datum, tensor),

        wherein dump_datum is an instance of data_dump.DebugTensorDatum carrying
        metadata about the dumped tensor, including tensor name, timestamps, etc.
        tensor is the value of the dumped tensor as an numpy.ndarray object.
        The return value of the function is a bool.
        This is the same signature as the input argument to
        dbg_dump.DebugDumpDir.find().

        Args:
          filter_name: (str) name of the filter. Cannot be empty.
          filter_callable: (callable) a filter function of the signature described
            as above.

        Raises:
          ValueError: If filter_name is an empty str.
          TypeError: If filter_name is not a str.
                     Or if filter_callable is not callable.
        """

        if not isinstance(filter_name, str):
            raise TypeError("Input argument filter_name is expected to be str, "
                            "but is not.")

        # Check that filter_name is not an empty str.
        if not filter_name:
            raise ValueError("Input argument filter_name cannot be empty.")

        # Check that filter_callable is callable.
        if not callable(filter_callable):
            raise TypeError(
                "Input argument filter_callable is expected to be callable, "
                "but is not.")

        self._tensor_filters[filter_name] = filter_callable

    def get_tensor_filter(self, filter_name):
        """Retrieve filter function by name.

        Args:
          filter_name: Name of the filter set during add_tensor_filter() call.

        Returns:
          The callable associated with the filter name.

        Raises:
          ValueError: If there is no tensor filter of the specified filter name.
        """

        if filter_name not in self._tensor_filters:
            raise ValueError("There is no tensor filter named \"%s\"" % filter_name)

        return self._tensor_filters[filter_name]

    def get_help(self, handler_name):
        """ Retrieve ArgumentParser full help text.

        Args:
          handler_name: argument name.
        Returns:
          Formats the full help text and returns it as a string.
        """
        return self._arg_parsers[handler_name].format_help()

    def list_graphnodes(self, args, screen_info=None):
        """Command handler for list_graphnodes.

        List tensors dumped during debugged Session.run() call.

        Args:
          args: Command-line arguments, excluding the command prefix, as a list of
            str.
          screen_info: Optional dict input containing screen information such as
            cols.

        Returns:
          Output text lines as a RichTextLines object.
        """

        _ = screen_info
        parsed = self._arg_parsers["list_graphnodes"].parse_args(args)

        output = []

        filter_strs = []
        if parsed.op_type_filter:
            op_type_regex = re.compile(parsed.op_type_filter)
            filter_strs.append("Op type regex filter: \"%s\"" % parsed.op_type_filter)
        else:
            op_type_regex = None

        if parsed.node_name_filter:
            node_name_regex = re.compile(parsed.node_name_filter)
            filter_strs.append("Node name regex filter: \"%s\"" %
                               parsed.node_name_filter)
        else:
            node_name_regex = None

        output = ui_common.RichTextLines(filter_strs)
        output.append("")

        if parsed.tensor_filter:
            try:
                filter_callable = self.get_tensor_filter(parsed.tensor_filter)
            except ValueError:
                output = ui_shared.error("There is no tensor filter named \"%s\"." %
                                         parsed.tensor_filter)
                _add_main_menu(output, node_name=None, enable_list_graphnodes=False)
                return output

            data_to_show = self._debug_dump.find(filter_callable)
        else:
            data_to_show = self._debug_dump.dumped_tensor_data

        ts_width, dump_size_width, op_type_width, no_ip_width, no_op_width, no_tensor_width = (
            self._measure_tensor_lst_col_width(data_to_show))

        # Sort the data.
        data_to_show = self._sort_dump_data_by(
            data_to_show, parsed.sort_by, parsed.reverse)

        output.extend(
            self._tensor_list_column_heads(parsed, ts_width,
                                           dump_size_width, op_type_width,
                                           no_ip_width, no_op_width, no_tensor_width))

        dump_count = 0
        for dump in data_to_show:
            if node_name_regex and not node_name_regex.match(dump.node_name):
                continue

            if op_type_regex:
                op_type = self._debug_dump.node_op_type(dump.node_name)
                if not op_type_regex.match(op_type):
                    continue

            rel_time = (dump.timestamp - self._debug_dump.ts0) / 1000.0
            dump_size_str = ui_shared.bytes_to_readable_str(dump.dump_size_bytes)
            dumped_tensor_name = "%s:%d" % (dump.node_name, dump.output_slot)
            op_type = self._debug_dump.node_op_type(dump.node_name)
            no_ips = str(len(self._debug_dump.node_inputs(dump.node_name)))
            no_ops = str(len(self._debug_dump.node_recipients(dump.node_name)))

            line = dumped_tensor_name
            line += " " * (no_tensor_width - len(line))

            line += "%.3f" % rel_time
            line += " " * (no_tensor_width + ts_width - len(line))
            line += dump_size_str
            line += " " * (no_tensor_width + ts_width + dump_size_width - len(line))

            line += no_ips
            line += " " * (no_tensor_width + no_ip_width + ts_width +
                           dump_size_width - len(line))

            line += no_ops
            line += " " * (no_tensor_width + no_op_width + no_ip_width + ts_width +
                           dump_size_width - len(line))

            line += op_type
            line += " " * (no_tensor_width + ts_width + dump_size_width +
                           no_op_width + no_ip_width + op_type_width - len(line))

            output.append(
                line,
                font_attr_segs=[(
                    0, len(dumped_tensor_name),
                    ui_common.MenuItem("", "vt %s" % dumped_tensor_name))])
            dump_count += 1

        if parsed.tensor_filter:
            output.prepend([
                "%d dumped tensor(s) passing filter \"%s\":" %
                (dump_count, parsed.tensor_filter)
            ])
        else:
            output.prepend(["Dumped tensor(s):%d" % dump_count])

        _add_main_menu(output, node_name=None, enable_list_graphnodes=False)
        return output

    def _measure_tensor_lst_col_width(self, data):
        """Determine the maximum widths of the timestamp and op-type column.

        This method assumes that data is sorted in the default order, i.e.,
        by ascending timestamps.

        Args:
          data: (list of DebugTensorDaum) the data based on which the maximum
            column widths will be determined.

        Returns:
          (int) maximum width of the timestamp column. 0 if data is empty.
          (int) maximum width of the dump size column. 0 if data is empty.
          (int) maximum width of the op type column. 0 if data is empty.
        """

        max_timestamp_width = 0
        if data:
            max_rel_time_ms = (data[-1].timestamp - self._debug_dump.ts0) / 1000.0
            max_timestamp_width = len("[%.3f] " % max_rel_time_ms) + 1
        max_timestamp_width = max(max_timestamp_width,
                                  len(self._TIMESTAMP_COLUMN_HEAD) + 1)

        max_dump_size_width = 0
        for dump in data:
            dump_size_str = ui_shared.bytes_to_readable_str(dump.dump_size_bytes)
            if len(dump_size_str) + 1 > max_dump_size_width:
                max_dump_size_width = len(dump_size_str) + 2
        max_dump_size_width = max(max_dump_size_width,
                                  len(self._DUMP_SIZE_COLUMN_HEAD) + 2)

        max_op_type_width = 0
        for dump in data:
            op_type = self._debug_dump.node_op_type(dump.node_name)
            if len(op_type) + 1 > max_op_type_width:
                max_op_type_width = len(op_type) + 2
        max_op_type_width = max(max_op_type_width,
                                len(self._OP_TYPE_COLUMN_HEAD) + 2)

        max_no_ip_width = 0
        for dump in data:
            no_ips = str(len(self._debug_dump.node_inputs(dump.node_name)))
            if len(no_ips) + 1 > max_no_ip_width:
                max_no_ip_width = len(no_ips) + 2
        max_no_ip_width = max(max_no_ip_width,
                              len(self._OP_NUMBER_INPUTS_HEAD) + 2)

        max_no_op_width = 0
        for dump in data:
            no_ops = str(len(self._debug_dump.node_recipients(dump.node_name)))
            if len(no_ops) + 1 > max_no_op_width:
                max_no_op_width = len(no_ops) + 2
        max_no_op_width = max(max_no_op_width,
                              len(self._OP_NUMBER_OUTPUTS_HEAD) + 2)

        max_no_tensorname_width = 0
        for dump in data:
            nodename = dump.node_name
            if len(nodename) + 1 > max_no_tensorname_width:
                max_no_tensorname_width = len(nodename) + 4
        max_no_tensorname_width = max(max_no_tensorname_width,
                                      len(self._TENSOR_NAME_COLUMN_HEAD) + 4)

        return (max_timestamp_width, max_dump_size_width, max_op_type_width,
                max_no_ip_width, max_no_op_width, max_no_tensorname_width)

    def _sort_dump_data_by(self, data, sort_by, reverse):
        """Sort a list of DebugTensorDatum in specified order.

        Args:
          data: (list of DebugTensorDatum) the data to be sorted.
          sort_by: The field to sort data by.
          reverse: (bool) Whether to use reversed (descending) order.

        Returns:
          (list of DebugTensorDatum) in sorted order.

        Raises:
          ValueError: given an invalid value of sort_by.
        """

        if sort_by == SORT_TENSORS_BY_TIMESTAMP:
            return sorted(
                data,
                reverse=reverse,
                key=lambda x: x.timestamp)
        elif sort_by == SORT_TENSORS_BY_DUMP_SIZE:
            return sorted(data, reverse=reverse, key=lambda x: x.dump_size_bytes)
        elif sort_by == SORT_TENSORS_BY_OP_TYPE:
            return sorted(
                data,
                reverse=reverse,
                key=lambda x: self._debug_dump.node_op_type(x.node_name))

        elif sort_by == SORT_TENSORS_BY_INPUT_SIZE:
            return sorted(
                data,
                reverse=reverse,
                key=lambda x: len(self._debug_dump.node_inputs(x.node_name)))
        elif sort_by == SORT_TENSORS_BY_OUTPUT_SIZE:
            return sorted(
                data,
                reverse=reverse,
                key=lambda x: len(self._debug_dump.node_recipients(x.node_name)))

        elif sort_by == SORT_TENSORS_BY_TENSOR_NAME:
            return sorted(
                data,
                reverse=reverse,
                key=lambda x: "%s:%d" % (x.node_name, x.output_slot))
        else:
            raise ValueError("Unsupported key to sort tensors by: %s" % sort_by)

    def _tensor_list_column_heads(self, parsed, max_timestamp_width,
                                  max_dump_size_width, max_op_type_width,
                                  max_no_ip_width, max_no_op_width, no_tensor_width):
        """Generate a line containing the column heads of the tensor list.

        Args:
          parsed: Parsed arguments (by argparse) of the list_graphnodes command.
          max_timestamp_width: (int) maximum width of the timestamp column.
          max_dump_size_width: (int) maximum width of the dump size column.
          max_op_type_width: (int) maximum width of the op type column.

        Returns:
          A RichTextLines object.
        """

        base_command = "list_graphnodes"
        if parsed.tensor_filter:
            base_command += " -f %s" % parsed.tensor_filter
        if parsed.op_type_filter:
            base_command += " -t %s" % parsed.op_type_filter
        if parsed.node_name_filter:
            base_command += " -n %s" % parsed.node_name_filter

        attr_segs = {0: []}

        row = self._TENSOR_NAME_COLUMN_HEAD
        command = "%s -s %s" % (base_command, SORT_TENSORS_BY_TENSOR_NAME)
        if parsed.sort_by == SORT_TENSORS_BY_TENSOR_NAME and not parsed.reverse:
            command += " -r"
        attr_segs[0].append((0, len(row),
                             [ui_common.MenuItem("", command), "bold"]))
        row += " " * (no_tensor_width - len(row))

        prev_len = len(row)
        row += self._TIMESTAMP_COLUMN_HEAD
        command = "%s -s %s" % (base_command, SORT_TENSORS_BY_TIMESTAMP)
        if parsed.sort_by == SORT_TENSORS_BY_TIMESTAMP and not parsed.reverse:
            command += " -r"
        attr_segs[0].append(
            (prev_len, len(row), [ui_common.MenuItem(None, command), "bold"]))
        row += " " * (no_tensor_width + max_timestamp_width - len(row))

        prev_len = len(row)
        row += self._DUMP_SIZE_COLUMN_HEAD
        command = "%s -s %s" % (base_command, SORT_TENSORS_BY_DUMP_SIZE)
        if parsed.sort_by == SORT_TENSORS_BY_DUMP_SIZE and not parsed.reverse:
            command += " -r"
        attr_segs[0].append((prev_len, len(row),
                             [ui_common.MenuItem(None, command), "bold"]))
        row += " " * (no_tensor_width + max_dump_size_width + max_timestamp_width - len(row))

        prev_len = len(row)
        row += self._OP_NUMBER_INPUTS_HEAD
        command = "%s -s %s" % (base_command, SORT_TENSORS_BY_INPUT_SIZE)
        if parsed.sort_by == SORT_TENSORS_BY_INPUT_SIZE and not parsed.reverse:
            command += " -r"
        attr_segs[0].append((prev_len, len(row),
                             [ui_common.MenuItem(None, command), "bold"]))
        row += " " * (no_tensor_width + max_no_ip_width + max_dump_size_width +
                      max_timestamp_width - len(row))

        prev_len = len(row)
        row += self._OP_NUMBER_OUTPUTS_HEAD
        command = "%s -s %s" % (base_command, SORT_TENSORS_BY_OUTPUT_SIZE)
        if parsed.sort_by == SORT_TENSORS_BY_OUTPUT_SIZE and not parsed.reverse:
            command += " -r"
        attr_segs[0].append((prev_len, len(row),
                             [ui_common.MenuItem(None, command), "bold"]))
        row += " " * (no_tensor_width + max_no_op_width + max_no_ip_width + max_dump_size_width +
                      max_timestamp_width - len(row))

        prev_len = len(row)
        row += self._OP_TYPE_COLUMN_HEAD
        command = "%s -s %s" % (base_command, SORT_TENSORS_BY_OP_TYPE)
        if parsed.sort_by == SORT_TENSORS_BY_OP_TYPE and not parsed.reverse:
            command += " -r"
        attr_segs[0].append((prev_len, len(row),
                             [ui_common.MenuItem(None, command), "bold"]))
        row += " " * (no_tensor_width + max_no_op_width + max_no_ip_width + max_op_type_width +
                      max_dump_size_width + max_timestamp_width - len(row))

        return ui_common.RichTextLines([row], font_attr_segs=attr_segs)

    def node_details(self, args, screen_info=None):
        """Command handler for node_details.

        Query information about a given node.

        Args:
          args: Command-line arguments, excluding the command prefix, as a list of
            str.
          screen_info: Optional dict input containing screen information such as
            cols.

        Returns:
          Output text lines as a RichTextLines object.
        """

        _ = screen_info
        parsed = self._arg_parsers["node_details"].parse_args(args)

        # Get a node name, regardless of whether the input is a node name (without
        # output slot attached) or a tensor name (with output slot attached).
        node_name, _ = graph_dump.parse_node_or_tensor_name(
            parsed.node_name)

        if not self._debug_dump.node_exists(node_name):
            output = ui_shared.error(
                "There is no node named \"%s\" in the partition graphs" % node_name)
            _add_main_menu(
                output,
                node_name=None,
                enable_list_graphnodes=True,
                enable_node_details=False,
                enable_graphnode_inputs=False,
                enable_graphnode_outputs=False)
            return output

        lines = ["Node %s" % node_name]
        font_attr_segs = {
            0: [(len(lines[-1]) - len(node_name), len(lines[-1]), "blue")]
        }
        lines.append("")
        lines.append("  Op: %s" % self._debug_dump.node_op_type(node_name))
        node_device_name = self._debug_dump.node_device(node_name).split("device:", 1)[1]
        lines.append("  Device: %s" % node_device_name)
        output = ui_common.RichTextLines(
            lines, font_attr_segs=font_attr_segs)

        # List node inputs (non-control and control).
        inputs = self._debug_dump.node_inputs(node_name)
        ctrl_inputs = self._debug_dump.node_inputs(node_name, is_control=True)
        output.extend(self._format_neighbors("input", inputs, ctrl_inputs))

        # List node output recipients (non-control and control).
        recs = self._debug_dump.node_recipients(node_name)
        ctrl_recs = self._debug_dump.node_recipients(node_name, is_control=True)
        output.extend(self._format_neighbors("recipient", recs, ctrl_recs))

        # Optional: List attributes of the node.
        if parsed.attributes:
            output.extend(self._list_node_attributes(node_name))

        # Optional: List dumps available from the node.
        if parsed.dumps:
            output.extend(self._list_node_dumps(node_name))

        _add_main_menu(output, node_name=node_name, enable_node_details=False)
        return output

    def _render_node_traceback(self, node_name):
        """Render traceback of a node's creation in Python, if available.

        Args:
          node_name: (str) name of the node.

        Returns:
          A RichTextLines object containing the stack trace of the node's
          construction.
        """

        lines = [RL(""), RL(""), RL("Traceback of node construction:", "bold")]

        try:
            node_stack = self._debug_dump.node_traceback(node_name)
            for depth, (file_path, line, function_name, text) in enumerate(
                    node_stack):
                lines.append("%d: %s" % (depth, file_path))

                attribute = ui_common.MenuItem(
                    "", "ps %s -b %d" % (file_path, line)) if text else None
                line_number_line = RL("  ")
                line_number_line += RL("Line:     %d" % line, attribute)
                lines.append(line_number_line)

                lines.append("  Function: %s" % function_name)
                lines.append("  Text:     " + (("\"%s\"" % text) if text else "None"))
                lines.append("")
        except KeyError:
            lines.append("(Node unavailable in the loaded Python graph)")
        except LookupError:
            lines.append("(Unavailable because no Python graph has been loaded)")

        return ui_common.rich_text_lines_frm_line_list(lines)

    def graphnode_inputs(self, args, screen_info=None):
        """Command handler for inputs.

        Show inputs to a given node.

        Args:
          args: Command-line arguments, excluding the command prefix, as a list of
            str.
          screen_info: Optional dict input containing screen information such as
            cols.

        Returns:
          Output text lines as a RichTextLines object.
        """

        _ = screen_info
        parsed = self._arg_parsers["graphnode_inputs"].parse_args(args)

        output = self._graphnode_inputs_or_outputs(
            parsed.recursive,
            parsed.node_name,
            parsed.depth,
            parsed.op_type,
            do_outputs=False)

        node_name = graph_dump.get_node_name(parsed.node_name)
        _add_main_menu(output, node_name=node_name, enable_graphnode_inputs=False)

        return output

    def view_tensor(self, args, screen_info=None):
        """Command handler for view_tensor.

        Print value of a given dumped tensor.

        Args:
          args: Command-line arguments, excluding the command prefix, as a list of
            str.
          screen_info: Optional dict input containing screen information such as
            cols.

        Returns:
          Output text lines as a RichTextLines object.
        """

        parsed = self._arg_parsers["view_tensor"].parse_args(args)

        np_printoptions = ui_shared.get_np_printoptions_frm_scr(
            screen_info)

        # Determine if any range-highlighting is required.
        highlight_options = ui_shared.parse_ranges_highlight(parsed.ranges)

        tensor_name, tensor_slicing = (
            command_parser.parse_tensor_name_with_slicing(parsed.tensor_name))

        node_name, output_slot = graph_dump.parse_node_or_tensor_name(tensor_name)
        if (self._debug_dump.loaded_partition_graphs() and
                not self._debug_dump.node_exists(node_name)):
            output = ui_shared.error(
                "Node \"%s\" does not exist in partition graphs" % node_name)
            _add_main_menu(
                output,
                node_name=None,
                enable_list_graphnodes=True,
                enable_view_tensor=False)
            return output

        watch_keys = self._debug_dump.debug_watch_keys(node_name)
        if output_slot is None:
            output_slots = set()
            for watch_key in watch_keys:
                output_slots.add(int(watch_key.split(":")[1]))

            if len(output_slots) == 1:
                # There is only one dumped tensor from this node, so there is no
                # ambiguity. Proceed to show the only dumped tensor.
                output_slot = list(output_slots)[0]
            else:
                # There are more than one dumped tensors from this node. Indicate as
                # such.
                lines = [
                    "Node \"%s\" generated debug dumps from %s output slots:" %
                    (node_name, len(output_slots)),
                    "Please specify the output slot: %s:x." % node_name
                ]
                output = ui_common.RichTextLines(lines)
                _add_main_menu(
                    output,
                    node_name=node_name,
                    enable_list_graphnodes=True,
                    enable_view_tensor=False)
                return output

        # Find debug dump data that match the tensor name (node name + output
        # slot).
        matching_data = []
        for watch_key in watch_keys:
            debug_tensor_data = self._debug_dump.watch_key_to_data(watch_key)
            for datum in debug_tensor_data:
                if datum.output_slot == output_slot:
                    matching_data.append(datum)

        if not matching_data:
            # No dump for this tensor.
            output = ui_shared.error("Tensor \"%s\" did not generate any dumps." %
                                     parsed.tensor_name)
        elif len(matching_data) == 1:
            # There is only one dump for this tensor.
            if parsed.number <= 0:
                output = ui_shared.format_tensor(
                    matching_data[0].get_tensor(),
                    matching_data[0].watch_key,
                    np_printoptions,
                    print_all=parsed.print_all,
                    tensor_slicing=tensor_slicing,
                    highlight_options=highlight_options,
                    include_numeric_summary=parsed.numeric_summary,
                    write_path=parsed.write_path)
            else:
                output = ui_shared.error(
                    "Invalid number (%d) for tensor %s, which generated one dump." %
                    (parsed.number, parsed.tensor_name))

            _add_main_menu(output, node_name=node_name, enable_view_tensor=False)
        else:
            # There are more than one dumps for this tensor.
            if parsed.number < 0:
                lines = [
                    "Tensor \"%s\" generated %d dumps:" % (parsed.tensor_name,
                                                           len(matching_data))
                ]
                font_attr_segs = {}

                for i, datum in enumerate(matching_data):
                    rel_time = (datum.timestamp - self._debug_dump.ts0) / 1000.0
                    lines.append("#%d [%.3f ms] %s" % (i, rel_time, datum.watch_key))
                    command = "view_tensor %s -n %d" % (parsed.tensor_name, i)
                    font_attr_segs[len(lines) - 1] = [(
                        len(lines[-1]) - len(datum.watch_key), len(lines[-1]),
                        ui_common.MenuItem(None, command))]

                lines.append("")
                lines.append(
                    "You can use the -n (--number) flag to specify which dump to "
                    "print.")
                lines.append("For example:")
                lines.append("  view_tensor %s -n 0" % parsed.tensor_name)

                output = ui_common.RichTextLines(
                    lines, font_attr_segs=font_attr_segs)
            elif parsed.number >= len(matching_data):
                output = ui_shared.error(
                    "Specified number (%d) exceeds the number of available dumps "
                    "(%d) for tensor %s" %
                    (parsed.number, len(matching_data), parsed.tensor_name))
            else:
                output = ui_shared.format_tensor(
                    matching_data[parsed.number].get_tensor(),
                    matching_data[parsed.number].watch_key + " (dump #%d)" %
                    parsed.number,
                    np_printoptions,
                    print_all=parsed.print_all,
                    tensor_slicing=tensor_slicing,
                    highlight_options=highlight_options,
                    write_path=parsed.write_path)
            _add_main_menu(output, node_name=node_name, enable_view_tensor=False)

        return output

    def graphnode_outputs(self, args, screen_info=None):
        """Command handler for inputs.

        Show inputs to a given node.

        Args:
          args: Command-line arguments, excluding the command prefix, as a list of
            str.
          screen_info: Optional dict input containing screen information such as
            cols.

        Returns:
          Output text lines as a RichTextLines object.
        """

        _ = screen_info
        parsed = self._arg_parsers["graphnode_outputs"].parse_args(args)

        output = self._graphnode_inputs_or_outputs(
            parsed.recursive,
            parsed.node_name,
            parsed.depth,
            parsed.op_type,
            do_outputs=True)

        node_name = graph_dump.get_node_name(parsed.node_name)
        _add_main_menu(output, node_name=node_name, enable_graphnode_outputs=False)

        return output

    def _graphnode_inputs_or_outputs(self,
                                     recursive,
                                     node_name,
                                     depth,
                                     op_type,
                                     do_outputs=False):
        """Helper function used by graphnode_inputs and graphnode_outputs.

        Format a list of lines to display the inputs or output recipients of a
        given node.

        Args:
          recursive: Whether the listing is to be done recursively, as a boolean.
          node_name: The name of the node in question, as a str.
          depth: Maximum recursion depth, applies only if recursive == True, as an
            int.
          op_type: Whether the op types of the nodes are to be included, as a
            boolean.
          do_outputs: Whether recipients, instead of input nodes are to be
            listed, as a boolean.

        Returns:
          Input or recipient tree formatted as a RichTextLines object.
        """

        if do_outputs:
            tracker = self._debug_dump.node_recipients
            type_str = "Recipients of"
        else:
            tracker = self._debug_dump.node_inputs
            type_str = "Inputs to"

        lines = []
        font_attr_segs = {}

        # Check if this is a tensor name, instead of a node name.
        node_name, _ = graph_dump.parse_node_or_tensor_name(node_name)

        # Check if node exists.
        if not self._debug_dump.node_exists(node_name):
            return ui_shared.error(
                "There is no node named \"%s\" in the partition graphs" % node_name)

        if recursive:
            max_depth = depth
        else:
            max_depth = 1

        line = "%s node \"%s\"" % (type_str, node_name)
        font_attr_segs[0] = [(len(line) - 1 - len(node_name), len(line) - 1, "blue")]
        lines.append(line + " (Depth limit = %d):" % (max_depth))

        command_template = "go -r %s" if do_outputs else "gi -r %s"
        self._dfs_from_node(
            lines,
            font_attr_segs,
            node_name,
            tracker,
            max_depth,
            1, [],
            op_type,
            command_template=command_template)

        # Include legend.
        lines.append("")
        lines.append("Legend:")
        lines.append("  (d): recursion depth = d.")

        if op_type:
            lines.append("  [Op]: Input node has op type Op.")

        return ui_common.RichTextLines(
            lines, font_attr_segs=font_attr_segs, additional_attr=curses.A_BOLD)

    def _dfs_from_node(self,
                       lines,
                       attr_segs,
                       node_name,
                       tracker,
                       max_depth,
                       depth,
                       unfinished,
                       show_op_type=False,
                       command_template=None):
        """Perform depth-first search (DFS) traversal of a node's input tree.

        It recursively tracks the inputs (or output recipients) of the node called
        node_name, and append these inputs (or output recipients) to a list of text
        lines (lines) with proper indentation that reflects the recursion depth,
        together with some formatting attributes (to attr_segs). The formatting
        attributes can include command shortcuts, for example.

        Args:
          lines: Text lines to append to, as a list of str.
          attr_segs: (dict) Attribute segments dictionary to append to.
          node_name: Name of the node, as a str. This arg is updated during the
            recursion.
          tracker: A callable that takes one str as the node name input and
            returns a list of str as the inputs/outputs.
            This makes it this function general enough to be used with both
            node-input and node-output tracking.
          max_depth: Maximum recursion depth, as an int.
          depth: Current recursion depth. This arg is updated during the
            recursion.
          unfinished: A stack of unfinished recursion depths, as a list of int.
          include_control: Whether control dependencies are to be included as
            inputs (and marked as such).
          show_op_type: Whether op type of the input nodes are to be displayed
            alongside the nodes' names.
          command_template: (str) Template for command shortcut of the node names.
        """

        # Make a shallow copy of the list because it may be extended later.
        all_inputs = copy.copy(tracker(node_name, is_control=False))
        is_ctrl = [False] * len(all_inputs)

        if not all_inputs:
            if depth == 1:
                lines.append("  [None]")

            return

        unfinished.append(depth)

        # Create depth-dependent hanging indent for the line.
        hang = ""
        for k in range(depth):
            if k < depth - 1:
                if k + 1 in unfinished:
                    hang += HANG_UNFINISHED
                else:
                    hang += HANG_FINISHED
            else:
                hang += HANG_SUFFIX

        if all_inputs and depth > max_depth:
            lines.append(hang + ELLIPSIS)
            unfinished.pop()
            return

        hang += DEPTH_TEMPLATE % depth

        len_all_inputs = len(all_inputs)
        for i in range(len_all_inputs):
            inp = all_inputs[i]
            op_type = self._debug_dump.node_op_type(graph_dump.get_node_name(inp))

            if is_ctrl[i]:
                ctrl_str = CTRL_LABEL
            else:
                ctrl_str = ""

            op_type_str = ""
            if show_op_type:
                op_type_str = OP_TYPE_TEMPLATE % op_type

            if i == len(all_inputs) - 1:
                unfinished.pop()

            line = hang + ctrl_str + op_type_str + inp
            lines.append(line)
            if command_template:
                attr_segs[len(lines) - 1] = [(
                    len(line) - len(inp), len(line),
                    ui_common.MenuItem(None, command_template % inp))]

            # Recursive call.
            # The input's/output's name can be a tensor name, in the case of node
            # with >1 output slots.
            inp_node_name, _ = graph_dump.parse_node_or_tensor_name(inp)
            self._dfs_from_node(
                lines,
                attr_segs,
                inp_node_name,
                tracker,
                max_depth,
                depth + 1,
                unfinished,
                show_op_type=show_op_type,
                command_template=command_template)

    def _format_neighbors(self, neighbor_type, non_ctrls, ctrls, neighbors_display=False):
        """List neighbors (inputs or recipients) of a node.

        Args:
          neighbor_type: ("input" | "recipient")
          non_ctrls: Non-control neighbor node names, as a list of str.
          ctrls: Control neighbor node names, as a list of str.

        Returns:
          A RichTextLines object.
        """

        lines = []
        font_attr_segs = {}

        lines.append("")
        if neighbors_display:
            lines.append("  %d %s(s) + %d control %s(s):" %
                         (len(non_ctrls), neighbor_type, len(ctrls), neighbor_type))
            lines.append("    %d %s(s):" % (len(non_ctrls), neighbor_type))
        else:
            lines.append("  %d %s(s):" % (len(non_ctrls), neighbor_type))
        for non_ctrl in non_ctrls:
            if neighbors_display:
                line = "      [%s] %s" % (self._debug_dump.node_op_type(non_ctrl),
                                          non_ctrl)
            else:
                line = "    [%s] %s" % (self._debug_dump.node_op_type(non_ctrl),
                                        non_ctrl)
            lines.append(line)
            font_attr_segs[len(lines) - 1] = [(
                len(line) - len(non_ctrl), len(line),
                ui_common.MenuItem(None, "nd -a -d -t %s" % non_ctrl))]

        if ctrls and neighbors_display:
            lines.append("")
            lines.append("    %d control %s(s):" % (len(ctrls), neighbor_type))
            for ctrl in ctrls:
                line = "      [%s] %s" % (self._debug_dump.node_op_type(ctrl), ctrl)
                lines.append(line)
                font_attr_segs[len(lines) - 1] = [(
                    len(line) - len(ctrl), len(line),
                    ui_common.MenuItem(None, "nd -a -d -t %s" % ctrl))]

        return ui_common.RichTextLines(
            lines, font_attr_segs=font_attr_segs)

    def _list_node_attributes(self, node_name):
        """List neighbors (inputs or recipients) of a node.

        Args:
          node_name: Name of the node of which the attributes are to be listed.

        Returns:
          A RichTextLines object.
        """

        lines = []
        lines.append("")
        lines.append("Node attributes:")

        attrs = self._debug_dump.node_attributes(node_name)
        for attr_key in attrs:
            lines.append("  %s:" % attr_key)
            attr_val_str = repr(attrs[attr_key]).strip().replace("\n", " ")
            lines.append("    %s" % attr_val_str)
            lines.append("")

        return ui_common.RichTextLines(lines)

    def _list_node_dumps(self, node_name):
        """List dumped tensor data from a node.

        Args:
          node_name: Name of the node of which the attributes are to be listed.

        Returns:
          A RichTextLines object.
        """

        lines = []
        font_attr_segs = {}

        watch_keys = self._debug_dump.debug_watch_keys(node_name)

        dump_count = 0
        for watch_key in watch_keys:
            debug_tensor_data = self._debug_dump.watch_key_to_data(watch_key)
            for datum in debug_tensor_data:
                if not datum.debug_op:
                    line = "  Slot %d @ %.3f ms" % (
                        datum.output_slot,
                        (datum.timestamp - self._debug_dump.ts0) / 1000.0)
                else:
                    line = "  Slot %d @ %s @ %.3f ms" % (
                        datum.output_slot, datum.debug_op,
                        (datum.timestamp - self._debug_dump.ts0) / 1000.0)
                lines.append(line)
                command = "vt %s:%d -n %d" % (node_name, datum.output_slot, dump_count)
                font_attr_segs[len(lines) - 1] = [(
                    2, len(line), ui_common.MenuItem(None, command))]
                dump_count += 1

        output = ui_common.RichTextLines(
            lines, font_attr_segs=font_attr_segs)
        output_with_header = ui_common.RichTextLines(
            ["Dumped tensor(s):%d" % dump_count, ""])
        output_with_header.extend(output)
        return output_with_header


def create_analyzer_ui(debug_dump,
                       tensor_filters=None,
                       ui_type="curses",
                       on_ui_exit=None,
                       config=None):
    """Create an instance of CursesUI based on a DebugDumpDir object.

    Args:
      debug_dump: (dbg_dump.DebugDumpDir) The debug dump to use.
      tensor_filters: (dict) A dict mapping tensor filter name (str) to tensor
        filter (Callable).
      ui_type: (str) requested UI type, e.g., "curses", "readline".
      on_ui_exit: (`Callable`) the callback to be called when the UI exits.
      config: A `ui_config.CLIConfig` object.

    Returns:
      (ui_base.BaseUI) A BaseUI subtype object with a set of standard analyzer
        commands and tab-completions registered.
    """
    if config is None:
        config = ui_config.CLIConfig()

    analyzer = DebugAnalyzer(debug_dump, config=config)
    if tensor_filters:
        for tensor_filter_name in tensor_filters:
            analyzer.add_tensor_filter(
                tensor_filter_name, tensor_filters[tensor_filter_name])

    cli = ui_factory.get_ui(ui_type, on_ui_exit=on_ui_exit, config=config)
    cli.register_command_handler(
        "list_graphnodes",
        analyzer.list_graphnodes,
        analyzer.get_help("list_graphnodes"),
        prefix_aliases=["lg"])
    cli.register_command_handler(
        "node_details",
        analyzer.node_details,
        analyzer.get_help("node_details"),
        prefix_aliases=["nd"])
    cli.register_command_handler(
        "graphnode_inputs",
        analyzer.graphnode_inputs,
        analyzer.get_help("graphnode_inputs"),
        prefix_aliases=["gi"])
    cli.register_command_handler(
        "graphnode_outputs",
        analyzer.graphnode_outputs,
        analyzer.get_help("graphnode_outputs"),
        prefix_aliases=["go"])
    cli.register_command_handler(
        "view_tensor",
        analyzer.view_tensor,
        analyzer.get_help("view_tensor"),
        prefix_aliases=["vt"])

    dumped_tensor_names = []
    for datum in debug_dump.dumped_tensor_data:
        dumped_tensor_names.append("%s:%d" % (datum.node_name, datum.output_slot))

    # Tab completions for command "view_tensors".
    cli.register_tab_comp_context(["view_tensor", "vt"], dumped_tensor_names)

    return cli
