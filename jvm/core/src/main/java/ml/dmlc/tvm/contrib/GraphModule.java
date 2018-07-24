package ml.dmlc.tvm.contrib;

import ml.dmlc.tvm.Function;
import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.NDArray;
import ml.dmlc.tvm.TVMContext;

/**
 * Wrapper runtime module.
 * This is a thin wrapper of the underlying TVM module.
 * you can also directly call set_input, run, and get_output
 * of underlying module functions.
 */
public class GraphModule {
  private Module module;
  private TVMContext ctx;

  private Function fsetInput;
  private Function frun;
  private Function fgetOutput;
  private Function fgetInput;
  private Function fdebugGetOutput;
  private Function floadParams;

  GraphModule(Module module, TVMContext ctx) {
    this.module = module;
    this.ctx = ctx;
    fsetInput = module.getFunction("set_input");
    frun = module.getFunction("run");
    fgetInput = module.getFunction("get_input");
    fgetOutput = module.getFunction("get_output");
    try {
      fdebugGetOutput = module.getFunction("debug_get_output");
    } catch (IllegalArgumentException ignored) {
      // ignore
    }
    floadParams = module.getFunction("load_params");
  }

  /**
   * Release the GraphModule.
   * <p>
   * We highly recommend you to do this manually since the GC strategy is lazy.
   * </p>
   */
  public void release() {
    fsetInput.release();
    frun.release();
    fgetInput.release();
    fgetOutput.release();
    if (fdebugGetOutput != null) {
      fdebugGetOutput.release();
    }
    floadParams.release();
    module.release();
  }

  /**
   * Set inputs to the module.
   * @param key The input key.
   * @param value The input value
   * @return self.
   */
  public GraphModule setInput(String key, NDArray value) {
    NDArray input = value;
    if (!value.ctx().equals(ctx)) {
      input = NDArray.empty(value.shape(), ctx);
      value.copyTo(input);
    }
    fsetInput.pushArg(key).pushArg(input).invoke();
    return this;
  }

  /**
   * Set inputs to the module
   * @param key The input key.
   * @param value The input value.
   * @return self.
   */
  public GraphModule setInput(int key, NDArray value) {
    NDArray input = value;
    if (!value.ctx().equals(ctx)) {
      input = NDArray.empty(value.shape(), ctx);
      value.copyTo(input);
    }
    fsetInput.pushArg(key).pushArg(input).invoke();
    return this;
  }

  /**
   * Run forward execution of the graph.
   * @return self.
   */
  public GraphModule run() {
    frun.invoke();
    return this;
  }

  /**
   * Get index-th input to out.
   * @param index The input index.
   * @param out The output array container.
   * @return out.
   */
  public NDArray getInput(int index, NDArray out) {
    fgetInput.pushArg(index).pushArg(out).invoke();
    return out;
  }

  /**
   * Get index-th output to out.
   * @param index The output index.
   * @param out The output array container.
   * @return out.
   */
  public NDArray getOutput(int index, NDArray out) {
    fgetOutput.pushArg(index).pushArg(out).invoke();
    return out;
  }

  /**
   * Run graph up to node and get the output to out.
   * @param node The node name.
   * @param out The output array container.
   * @return out.
   */
  public NDArray debugGetOutput(String node, NDArray out) {
    if (fdebugGetOutput != null) {
      fdebugGetOutput.pushArg(node).pushArg(out).invoke();
    } else {
      throw new RuntimeException("Please compile runtime with USE_GRAPH_RUNTIME_DEBUG = 0");
    }
    return out;
  }

  /**
   * Run graph up to node and get the output to out.
   * @param node The node index.
   * @param out The output array container.
   * @return out.
   */
  public NDArray debugGetOutput(int node, NDArray out) {
    if (fdebugGetOutput != null) {
      fdebugGetOutput.pushArg(node).pushArg(out).invoke();
    } else {
      throw new RuntimeException("Please compile runtime with USE_GRAPH_RUNTIME_DEBUG = 0");
    }
    return out;
  }

  /**
   * Load parameters from serialized byte array of parameter dict.
   * @param params The serialized parameter.
   * @return self.
   */
  public GraphModule loadParams(byte[] params) {
    floadParams.pushArg(params).invoke();
    return this;
  }

  /**
   * Get internal module function.
   * @param key The key to the module.
   * @return The function.
   * @throws IllegalArgumentException if function does not exist.
   */
  public Function getFunction(String key) {
    return module.getFunction(key);
  }
}
