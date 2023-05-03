import tvm
import tvm.testing
from tvm import relax
from tvm.script import relax as R, tir as T
from tvm.script import ir as I
from tvm.relax.transform import LazyTransformParams
@I.ir_module
class Before:
    @T.prim_func
    def transform_layout_IOHW_to_OIHW(
        w1: T.Buffer((3, 16, 3, 3), "float32"), out: T.Buffer((16, 3, 3, 3), "float32")
    ):
        for ax0, ax1, ax2, ax3 in T.grid(16, 3, 3, 3):
            with T.block("layout_transform"):
                o, i, h, w = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(w1[i, o, h, w])
                T.writes(out[o, i, h, w])
                out[o, i, h, w] = w1[i, o, h, w]
                
    @R.function
    def main_transform_params(
        params: R.Tuple(
            R.Tensor((3, 16, 3, 3), dtype="float32"), R.Tensor((16, 16, 3, 3), dtype="float32")
        )
    ) -> R.Tuple(
        R.Tensor((16, 16, 3, 3), dtype="float32"), R.Tensor((16, 3, 3, 3), dtype="float32")
    ):
        cls = Before
        with R.dataflow():
            lv: R.Tensor((16, 16, 3, 3), dtype="float32") = params[1]
            lv1: R.Tensor((3, 16, 3, 3), dtype="float32") = params[0]
            lv2 = R.call_tir(
                cls.transform_layout_IOHW_to_OIHW,
                (lv1,),
                out_sinfo=R.Tensor((16, 3, 3, 3), dtype="float32"),
            )
            gv: R.Tuple(
                R.Tensor((16, 16, 3, 3), dtype="float32"),
                R.Tensor((16, 3, 3, 3), dtype="float32"),
            ) = (lv, lv2)
            R.output(gv)
        return gv
    
    
mod = LazyTransformParams()(Before)
print(mod.script())
