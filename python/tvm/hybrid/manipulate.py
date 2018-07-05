"""Infrastructures for manipulating the IR directly."""

import sys
from .. import ir_pass
from .. import expr
from .. import stmt
from .. import make
from .. import api


def _axis(body):
    """A HalideIR body is given and return the loop variables in their occurance order.

    Parameters
    ----------
    body: HalideIR
        The HalideIR body to be inspected

    Returns
    -------
    res: list
        A list of loop variables in their order of occurance
    """
    res = []

    def preoprder(op):
        if isinstance(op, stmt.For):
            res.append(op.loop_var)
        return None

    ir_pass.IRTransform(body, preoprder, lambda op: None, ['For'])

    return res


def _split(body, var, factor=None, nparts=None):
    """Split a loop variable inside the HalideIR body

    Parameters
    ----------
    body: HalideIR
        The HalideIR body to be transformed

    var: variable
        The loop variable to be splited

    factor: int
        After transformation, the inner loop extent

    nparts: int
        After transformation, the outer loop extent

    Either excatly one of factor or nparts should be given!

    Returns
    -------
    (body, outer, inner): (HalideIR, variable, variable)
        body: The HalideIR body after transformation
        outer: The outer loop variable after transformation
        inner: The inner loop variable after transformation
    """

    did_transform = [False]
    loop_vars = []


    def get_outer_and_inner(value):
        """The loop extent is given. Get the inner and outer loop extent after splitting."""
        if factor is not None:
            if value % factor != 0:
                raise ValueError("The loop extent to be splitted should be dividable by factor!")
            outer = value / factor if sys.version_info[0] == 2 else value // factor
            return outer, factor
        if value % nparts != 0:
            raise ValueError("The loop extent to be splitted should be dividable by nparts!")
        inner = value / nparts if sys.version_info[0] == 2 else value // nparts
        return nparts, inner


    #pylint: disable=missing-docstring
    def preorder(op):
        if isinstance(op, stmt.For) and op.loop_var == var:
            if not isinstance(op.extent, (expr.IntImm, expr.UIntImm)):
                raise ValueError("The loop extent to be splitted should be a constant!")
            if op.for_type != stmt.For.Serial:
                raise ValueError("The loop to be splitted should be serial loop type!")
            extent_o, extent_i = get_outer_and_inner(op.extent.value)
            var_o = api.var(var.name + ".outer")
            var_i = api.var(var.name + ".inner")
            body = op.body
            body = ir_pass.Substitute(body, {var: var_o * extent_i + var_i})
            body = make.For(var_i, api.const(0, dtype='int32'), extent_i, stmt.For.Serial, 0, body)
            did_transform[0] = True
            loop_vars.append(var_o)
            loop_vars.append(var_i)
            return make.For(var_o, api.const(0, dtype='int32'), extent_o, stmt.For.Serial, 0, body)
        else:
            return None


    if (factor is None) + (nparts is None) != 1:
        raise ValueError("Between factor and nparts excatly one should be given!")

    res = ir_pass.IRTransform(body, preorder, None, ['For'])
    if not did_transform[0]:
        raise ValueError("Corresponding loop levle not found!")
    return res, loop_vars[0], loop_vars[1]


def _pragma(body, var, pragma_type, pragma_value=None):
    """Annotate the loop level with the given pragma

    Parameters
    ----------
    body: HalideIR
        The HalideIR body to be transformed

    var: IterVar
        The loop level to be annotated

    pragma_type: str
        The pragma type

    pragma_value: Expr, optional
        The value to be passed along with the pragma

    Returns
    -------
    res: HalideIR
        The HalideIR body after transformation
    """
    did_transform = [False]


    def preorder(op):
        if isinstance(op, stmt.For) and op.loop_var == var:
            did_transform[0] = True
            return make.AttrStmt(op.loop_var, pragma_type, pragma_value, op)
        return None


    res = ir_pass.IRTransform(body, preorder, None, ['For'])
    if not did_transform[0]:
        raise ValueError("Corresponding loop level not found!")
    return res


def _change_loop_type(body, var, for_type):
    """Change the given loop type.

    Parameters
    ----------
    body: HalideIR
        The HalideIR body to be transformed

    var: IterVar
        The loop level to change the for_type

    for_type: int
        The desired for_type

    Returns
    -------
    res: HalideIR
        The HalideIR body after transformation
    """
    if for_type < 0 or for_type > 3:
        raise ValueError("for_type should be an int in range [0, 3]!")

    did_transform = [False]


    def preorder(op):
        if isinstance(op, stmt.For) and op.loop_var == var:
            did_transform[0] = True
            return make.For(op.loop_var, op.min, op.extent, for_type, 0, op.body)
        return None


    res = ir_pass.IRTransform(body, preorder, None, ['For'])
    if not did_transform[0]:
        raise ValueError("Corresponding loop level not found!")
    return res


def _bind(body, var, thread_axis):
    """Bind a thread axis to the given loop level.

    Parameters
    ----------
    body: HalideIR
        The HalideIR body to be transformed

    var: IterVar
        The loop level to change the for_type

    thread_axis: thread_axis
        The thread_axis to be binded

    Returns
    -------
    res: HalideIR
        The HalideIR body after transformation
    """
    did_transform = [False]


    def preorder(op):
        if isinstance(op, stmt.For) and op.loop_var == var:
            did_transform[0] = True
            # Hack to convert an iter_var to an Expr
            body = ir_pass.Substitute(op.body, {var: thread_axis + 0})
            body = make.AttrStmt(thread_axis, 'thread_extent', op.extent, body)
            return body
        return None


    res = ir_pass.IRTransform(body, preorder, None, ['For'])
    if not did_transform[0]:
        raise ValueError("Corresponding loop level not found!")
    return res


def _reorder(body, *args):
    """Reorder the given loop levels.

    Parameters
    ----------
    body: HalideIR
        The HalideIR body to be transformed

    *args: IterVar s
        The loop levels to be reordered

    Returns
    -------
    res: HalideIR
        The HalideIR body after transformation
    """
    old_order = []
    loop_stack = []
    loop_info = {}
    all_on_one_chain = [False]


    def check_preorder(op):
        if isinstance(op, stmt.For) and op.loop_var in args:
            loop_info[op.loop_var] = (op.min, op.extent, op.for_type)
            old_order.append(op.loop_var)
            loop_stack.append(op.loop_var)
        return None


    def check_postorder(op):
        if isinstance(op, stmt.For) and op.loop_var in args:
            if len(args) == len(loop_stack):
                all_on_one_chain[0] = True
            loop_stack.pop()
        return None


    ir_pass.IRTransform(body, check_preorder, check_postorder, ['For'])

    if not all_on_one_chain[0]:
        raise ValueError("All the variables should be on the same chain of the 'for' tree!")
    reorder_map = {}
    for old, new in zip(old_order, args):
        reorder_map[old] = new


    def transform_postorder(op):
        if isinstance(op, stmt.For) and op.loop_var in reorder_map.keys():
            _new = reorder_map[op.loop_var]
            _min, _extent, _for_type = loop_info[_new]
            return make.For(_new, _min, _extent, _for_type, 0, op.body)
        return None

    return ir_pass.IRTransform(body, None, transform_postorder, ['For'])
