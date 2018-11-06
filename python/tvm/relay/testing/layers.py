"""Simple Layer DSL wrapper to ease creation of neural nets."""
from tvm import relay

def batch_norm_infer(data,
                     gamma=None,
                     beta=None,
                     moving_mean=None,
                     moving_var=None,
                     **kwargs):
    """Wrapper of batch_norm.

    This function automatically creates weights and return
    the first output(normalized result).

    Parameters
    ----------
    data : relay.Expr
        The input expression.

    gamma : relay.Expr
        The gamma scale factor.

    beta : relay.Expr
        The beta offset factor.

    moving_mean : relay.Expr
        Running mean of input,

    moving_var : relay.Expr
        Running variance of input.

    kwargs : dict
        Additional arguments.

    Returns
    -------
    result : relay.Expr
        The result.
    """
    name = kwargs.get("name")
    kwargs.pop("name")
    if not gamma:
        gamma = relay.var(name + "_gamma")
    if not beta:
        beta = relay.var(name + "_beta")
    if not moving_mean:
        moving_mean = relay.var(name + "_moving_mean")
    if not moving_var:
        moving_var = relay.var(name + "_moving_var")
    return relay.nn.batch_norm(data,
                               gamma=gamma,
                               beta=beta,
                               moving_mean=moving_mean,
                               moving_var=moving_var,
                               **kwargs)[0]


def conv2d(data, weight=None, **kwargs):
    """Wrapper of conv2d which automatically creates weights if not given.

    Parameters
    ----------
    data : relay.Expr
        The input expression.

    weight : relay.Expr
        The weight to conv2d.

    kwargs : dict
        Additional arguments.

    Returns
    -------
    result : relay.Expr
        The result.
    """
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    return relay.nn.conv2d(data, weight, **kwargs)

def conv2d_transpose(data, weight=None, **kwargs):
    """Wrapper of conv2d_transpose which automatically creates weights if not given.

    Parameters
    ----------
    data : relay.Expr
        The input expression.

    weight : relay.Expr
        The weight to conv2d_transpose.

    kwargs : dict
        Additional arguments.

    Returns
    -------
    result : relay.Expr
        The result.
    """
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    return relay.nn.conv2d_transpose(data, weight, **kwargs)

def dense_add_bias(data, weight=None, bias=None, units=None, **kwargs):
    """Wrapper of dense which automatically creates weights if not given.

    Parameters
    ----------
    data : relay.Expr
        The input expression.

    weight : relay.Expr
        The weight to conv2d.

    bias : relay.Expr
        The bias.

    kwargs : dict
        Additional arguments.

    Returns
    -------
    result : relay.Expr
        The result.
    """
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    if not bias:
        bias = relay.var(name + "_bias")
    data = relay.nn.dense(data, weight, units, **kwargs)
    data = relay.nn.bias_add(data, bias)
    return data
