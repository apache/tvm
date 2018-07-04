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


    def postorder(op):
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

    res = ir_pass.IRTransform(body, postorder, None, ['For'])
    if not did_transform[0]:
        raise ValueError("Corresponding loop levle not found!")
    return res, loop_vars[0], loop_vars[1]

def _change_loop_type(body, var, for_type):
    """Change the given loop type.

    Parameters
    ----------
    body: HalideIR
        The HalideIR body to be transformed

    var: variable
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


    def postorder(op):
        if isinstance(op, stmt.For) and op.loop_var == var:
            did_transform[0] = True
            return make.For(op.loop_var, op.min, op.extent, for_type, 0, op.body)
        return None


    res = ir_pass.IRTransform(body, postorder, None, ['For'])
    if not did_transform[0]:
        raise ValueError("Corresponding loop level not found!")
    return res
