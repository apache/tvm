# pylint: disable=invalid-name
"""Helper functions and global data"""


RULE_OUT_NODE_NAMES = ["Tuple", "TupleGetItem", "batch_flatten", "transpose", "reshape",
                       "multibox_prior", "multibox_transform_loc", "where",
                       "non_max_suppression", "strided_slice"]
INVALID_LAYOUT_TIME = 1000000000.0
