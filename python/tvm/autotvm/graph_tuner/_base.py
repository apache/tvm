# pylint: disable=invalid-name
"""Helper functions and global data"""


ELEMLIKE_NODE_NAMES = ["elemwise", "concatenate", "broadcast"]
RULE_OUT_NODE_NAMES = ["flatten", "transpose", "reshape", "multibox_prior"]
INVALID_LAYOUT_TIME = 1000000000.0
