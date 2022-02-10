from onnx.backend.base import Backend

from onnx.backend.base import BackendRep

from tvm import relay
from tvm.driver.tvmc import TVMCModel
from tvm.driver import tvmc

class TVMBackendRep(BackendRep):

    def __init__(self, model, device, **kwargs):
        super(TVMBackendRep, self).__init__()

#        print("************** MODEL INITIALIZATION ******************")
#        print(model)

#        print(model.graph.input)

        self._model, self._params = relay.frontend.from_onnx(model)
#        print("**model**")
#        print(self._model)
#        print("**params**")
#        print(self._params)
        self._tvmc_model = TVMCModel(self._model, self._params)
        self._package = tvmc.compile(self._tvmc_model, target="llvm")
        self._inputs = []
        for i in model.graph.input:
            self._inputs.append(i.name)
#        print( self._inputs)
#        print("************** MODEL INITIALIZATION COMPLETE ******************")

    def run(self, inputs, **kwargs):
#        print("************** MODEL INPUTS ******************")
#        print(inputs)
#        print(**kwargs)
        i = dict(zip(self._inputs, inputs))
        result = tvmc.run(tvmc_package=self._package,
                device="cpu",
                inputs=i)
        results = []
        for output in result.outputs:
            results.append(result.get_output(output))
#        print("************** MODEL OUTPUTS ******************")
#        print(results)
        return results

class TVMBackend(Backend):

    @classmethod
    def prepare(cls,
                model,
                device='CPU',
                **kwargs
                ):
        super(TVMBackend, cls).prepare(model, device, **kwargs)
        return TVMBackendRep(model, device, **kwargs)

    @classmethod
    def run_node(cls,
                 node,
                 inputs,
                 device='CPU',
                 outputs_info=None,
                 **kwargs
                 ):
        '''Simple run one operator and return the results.
        Args:
            outputs_info: a list of tuples, which contains the element type and
            shape of each output. First element of the tuple is the dtype, and
            the second element is the shape. More use case can be found in
            https://github.com/onnx/onnx/blob/main/onnx/backend/test/runner/__init__.py
        '''
#        print("RUNNING NODE")
        return super(TVMBackend, cls).run_node(node, inputs, device, outputs_info, **kwargs)

    @classmethod
    def supports_device(cls, device) -> bool:
        """
        Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return True
