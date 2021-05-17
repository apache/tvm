from typing import *

from tvm import relay
from tvm.relay.transform.fp16_conversion import graph_colors

FP16OutDtype = NamedTuple("FP16OutDtype", [("accumulation_dtype", Optional[str]), ("output_dtype", str)])


class DefaultFP16TypeDefinition:
    """By default we assume every node wants the final output to be float16. 
    
    If the relay node supports out_dtype we try to accumulate the fp32 before
    casting beack.
    """

    def __call__(self, call_node: relay.Call) -> FP16OutDtype:
        if call_node.attrs is not None and hasattr(call_node.attrs, "out_dtype"):
            # Assume for now we always accumulate into fp32 if given the option
            return FP16OutDtype(accumulation_dtype="float32", output_dtype="float16")
        else:
            return FP16OutDtype(accumulation_dtype=None, output_dtype="float16")
