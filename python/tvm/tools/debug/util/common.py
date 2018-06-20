"""Common values and methods for TVM Debugger."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_graph_element_name(elem):
    """Obtain the name or string representation of a graph element.

    If the graph element has the attribute "name", return name. Otherwise, return
    a __str__ representation of the graph element. Certain graph elements, such as
    `SparseTensor`s, do not have the attribute "name".

    Args:
      elem: The graph element in question.

    Returns:
      If the attribute 'name' is available, return the name. Otherwise, return
      str(output).
    """
    if hasattr(elem, "attr"):
        val = elem.attr("name")
    else:
        val = elem.name if hasattr(elem, "name") else str(elem)
    return val


def get_flattened_names(inputs_or_outputs):
    """Get a flattened list of the names in run() call inputs or outputs.

    Args:
      inputs_or_outputs: Inputs or outputs of the `Session.run()` call. It maybe
        a Tensor, an Operation or a Variable. It may also be nested lists, tuples
        or dicts. See doc of `Session.run()` for more details.

    Returns:
      (list of str) A flattened list of output names from `inputs_or_outputs`.
    """

    lines = []
    if isinstance(inputs_or_outputs, (list, tuple)):
        for item in inputs_or_outputs:
            lines.extend(get_flattened_names(item))
    elif isinstance(inputs_or_outputs, dict):
        for key in inputs_or_outputs:
            lines.extend(get_flattened_names(inputs_or_outputs[key]))
    elif ';' in inputs_or_outputs:
        names = inputs_or_outputs.split(";")
        for name in names:
            if name:
                lines.extend(get_flattened_names(name))
    else:
        # This ought to be a Tensor, an Operation or a Variable, for which the name
        # attribute should be available. (Bottom-out condition of the recursion.)
        lines.append(get_graph_element_name(inputs_or_outputs))

    return lines

class CLIRunStartAction(object):
    """Enum-like values for possible action to take on start of a run() call."""

    # Run once with debug tensor-watching.
    DEBUG_RUN = "debug_run"

    # Run without debug tensor-watching.
    NON_DEBUG_RUN = "non_debug_run"
