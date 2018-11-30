"""The attributes node used for Relay operators"""

from ...attrs import Attrs
from ..base import register_relay_attr_node

@register_relay_attr_node
class Conv2DAttrs(Attrs):
    """Attribute of a Convolution Operator"""
    pass

@register_relay_attr_node
class GlobalPool2DAttrs(Attrs):
    """Attribute of a Global 2D Pooling Operator"""
    pass
