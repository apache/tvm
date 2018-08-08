from nnvm import symbol as sym
from .base_tensor_executor import BaseTensorExecutor


class LayoutTransformExecutor(BaseTensorExecutor):
    def _get_op_symbol(self):
        return sym.__layout_transform__
