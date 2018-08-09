# pylint: disable=no-member
"""Layout transformation executor class"""
from nnvm import symbol as sym
from .base_tensor_executor import BaseTensorExecutor


class LayoutTransformExecutor(BaseTensorExecutor):
    """Executor class to benchmark layout transform operator.
    """
    def _get_op_symbol(self):
        """Get layout transform operator symbol.
        """
        return sym.__layout_transform__
