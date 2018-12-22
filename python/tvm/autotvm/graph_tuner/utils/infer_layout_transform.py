# pylint: disable=too-many-locals
"""Built-in functions to infer layout transformation."""

def infer_layout_shape_avx(wkl, current_sch, target_sch, multi_input_node_shape=None):
    """Infer actual input and output shapes for layout transformation
    given a workload, input schedule and output schedule.

    This function is for Intel AVX schedule template. Re-implement it
    for different workload and schedule templates.

    Take a CNN as example, a layout transformation can happen
    in two cases:
        1. Between two convolution nodes. Data shape before and after
           layout transformation can be determined purely by workload
           and schedules.
        2. Before multi_input nodes. In this case, shape of the
           multi_input node is required as well.

    Parameters
    ----------
    wkl : tuple
        Input workload. If this is a multi_input node, workload
        should come from the leftmost input node.

    current_sch : ConfigEntity
        Schedule before the layout transformation.

    target_sch : ConfigEntity
        Schedule after the layout transformation.

    multi_input_node_shape : tuple of int, optional
        Shape of node data if layout transformation happens before
        an multi_input node.
        Note: this shape should be inferred with original data layout.

    Returns
    -------
    in_shape : tuple of int
        Input shape of layout transformation.

    out_shape : tuple of int
        Output shape of layout transformation.

    is_valid : boolean
        Whether this is a valid layout transformation.
        An invalid transformation usually happens for concatenate operator.
    """
    layout = wkl[6]
    if layout == "NCHW":
        batch_size, in_channel, height, width, _ = wkl[1]
        out_channel = wkl[2][0]
        if multi_input_node_shape:
            height = multi_input_node_shape[2]
            width = multi_input_node_shape[3]
    elif layout == "NHWC":
        batch_size, height, width, in_channel, _ = wkl[1]
        out_channel = wkl[2][3]
        if multi_input_node_shape:
            height = multi_input_node_shape[1]
            width = multi_input_node_shape[2]
    else:
        raise RuntimeError("Layout %s is not supported yet." % layout)

    oc_bn_c = current_sch["tile_oc"].val if hasattr(current_sch["tile_oc"], "val") \
        else current_sch["tile_oc"].size[-1]
    ic_bn_t = target_sch["tile_ic"].val if hasattr(target_sch["tile_ic"], "val") \
        else target_sch["tile_ic"].size[-1]
    oc_bn_t = target_sch["tile_oc"].val if hasattr(target_sch["tile_oc"], "val") \
        else target_sch["tile_oc"].size[-1]
    is_valid = True
    if multi_input_node_shape:
        if out_channel % oc_bn_t != 0:
            is_valid = False
        in_shape = (batch_size, out_channel // oc_bn_c, height, width, oc_bn_c)
        out_shape = (batch_size, out_channel // oc_bn_t, height, width, oc_bn_t)
    else:
        if in_channel % oc_bn_c != 0:
            is_valid = False
        in_shape = (batch_size, in_channel // oc_bn_c, height, width, oc_bn_c)
        out_shape = (batch_size, in_channel // ic_bn_t, height, width, ic_bn_t)
    return in_shape, out_shape, is_valid
