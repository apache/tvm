/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.tvm.contrib;

import ml.dmlc.tvm.Function;
import ml.dmlc.tvm.Module;
import ml.dmlc.tvm.TVMContext;
import ml.dmlc.tvm.TVMValue;
import ml.dmlc.tvm.rpc.RPC;
import ml.dmlc.tvm.rpc.RPCSession;
import ml.dmlc.tvm.rpc.TVMRemoteContext;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class GraphRuntime {
  /**
   * Create a runtime executor module given a graph and module.
   * @param graphJson The graph deployed in json format output by nnvm graph.
   * @param libmod The module of the corresponding function.
   * @param ctx The local or remote context to deploy the module.
   * @return Runtime graph module that can be used to execute the graph.
   */
  public static GraphModule create(String graphJson, Module libmod, TVMContext ctx) {
    Module graphModule = null;
    if (ctx.deviceType >= RPC.RPC_SESS_MASK) {
      if (!(ctx instanceof  TVMRemoteContext)) {
        throw new IllegalArgumentException(
            "Looks like you are using remote context with no RPCSession bind."
            + "Use session.context instead.");
      }
      RPCSession rpcSession = ((TVMRemoteContext) ctx).rpcSession;
      // check arguments
      if (!"rpc".equals(libmod.typeKey())) {
        throw new IllegalArgumentException("libmod.typeKey != rpc");
      }
      final int sessIndex = (int) ((Function) reflectionStaticCall(
          RPC.class, "getApi", "_SessTableIndex"))
          .pushArg(libmod).invoke().asLong();
      if (sessIndex != (Integer) reflectionGetField(rpcSession, "tblIndex")) {
        throw new IllegalArgumentException(String.format(
            "libmod SessTableIndex=%d mismatch rpcSession.tblIndex=%d",
            sessIndex, reflectionGetField(rpcSession, "tblIndex")));
      }

      Function rpcModuleHandle = (Function) reflectionStaticCall(
          RPC.class, "getApi","_ModuleHandle");
      if (rpcModuleHandle == null) {
        throw new RuntimeException("Cannot find global function tvm.rpc._ModuleHandle."
            + "Did you compile tvm_runtime with the correct version?");
      }

      Function fcreate = Function.getFunction("tvm.graph_runtime.remote_create");
      if (fcreate == null) {
        throw new RuntimeException("Cannot find global function tvm.graph_runtime.remote_create."
            + "Did you compile tvm_runtime with correct version?");
      }

      TVMValue hmod = rpcModuleHandle.pushArg(libmod).invoke();
      graphModule = fcreate.call(graphJson, hmod,
          ctx.deviceType % RPC.RPC_SESS_MASK, ctx.deviceId).asModule();
    } else {
      Function fcreate = Function.getFunction("tvm.graph_runtime.create");
      if (fcreate == null) {
        throw new RuntimeException("Cannot find global function tvm.graph_runtime.create."
            + "Did you compile tvm_runtime with correct version?");
      }
      graphModule = fcreate.pushArg(graphJson)
          .pushArg(libmod).pushArg(ctx.deviceType).pushArg(ctx.deviceId)
          .invoke().asModule();
    }

    return new GraphModule(graphModule, ctx);
  }

  private static Object reflectionGetField(Object obj, String fieldName) {
    try {
      Field field = obj.getClass().getDeclaredField(fieldName);
      field.setAccessible(true);
      return field.get(obj);
    } catch (NoSuchFieldException e) {
      throw new RuntimeException(e);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  private static Object reflectionStaticCall(Class<?> clazz, String methodName, Object ... args) {
    Class<?>[] types = new Class<?>[args.length];
    for (int i = 0; i < args.length; ++i) {
      types[i] = args[i].getClass();
    }
    try {
      Method method = clazz.getDeclaredMethod(methodName, types);
      method.setAccessible(true);
      return method.invoke(null, args);
    } catch (NoSuchMethodException e) {
      throw new RuntimeException(e);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InvocationTargetException e) {
      throw new RuntimeException(e);
    }
  }
}
