/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { SizeOf, ArgTypeCode } from "./ctypes";
import { assert, StringToUint8Array, Uint8ArrayToString } from "./support";
import { detectGPUDevice, GPUDeviceDetectOutput } from "./webgpu";
import * as compact from "./compact";
import * as runtime from "./runtime";
import { Disposable } from "./types";

enum RPCServerState {
  InitHeader,
  InitHeaderKey,
  InitServer,
  WaitForCallback,
  ReceivePacketHeader,
  ReceivePacketBody,
}

/** RPC magic header */
const RPC_MAGIC = 0xff271;

/**
 * An utility class to read from binary bytes.
 */
class ByteStreamReader {
  offset = 0;
  bytes: Uint8Array;

  constructor(bytes: Uint8Array) {
    this.bytes = bytes;
  }

  readU32(): number {
    const i = this.offset;
    const b = this.bytes;
    const val = b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24);
    this.offset += 4;
    return val;
  }

  readU64(): number {
    const val = this.readU32();
    this.offset += 4;
    return val;
  }

  readByteArray(): Uint8Array {
    const len = this.readU64();
    assert(this.offset + len <= this.bytes.byteLength);
    const ret = new Uint8Array(len);
    ret.set(this.bytes.slice(this.offset, this.offset + len));
    this.offset += len;
    return ret;
  }
}

/**
 * A websocket based RPC
 */
export class RPCServer {
  url: string;
  key: string;
  socket: WebSocket;
  state: RPCServerState = RPCServerState.InitHeader;
  logger: (msg: string) => void;
  getImports: () => Record<string, unknown>;
  private ndarrayCacheUrl: string;
  private ndarrayCacheDevice: string;
  private initProgressCallback?: runtime.InitProgressCallback;
  private asyncOnServerLoad?: (inst: runtime.Instance) => Promise<void>;
  private pendingSend: Promise<void> = Promise.resolve();
  private name: string;
  private inst?: runtime.Instance = undefined;
  private globalObjects: Array<Disposable> = [];
  private serverRecvData?: (header: Uint8Array, body: Uint8Array) => void;
  private currPacketHeader?: Uint8Array;
  private currPacketLength = 0;
  private remoteKeyLength = 0;
  private pendingBytes = 0;
  private buffredBytes = 0;
  private messageQueue: Array<Uint8Array> = [];

  constructor(
    url: string,
    key: string,
    getImports: () => Record<string, unknown>,
    logger: (msg: string) => void = console.log,
    ndarrayCacheUrl = "",
    ndarrayCacheDevice = "cpu",
    initProgressCallback: runtime.InitProgressCallback | undefined = undefined,
    asyncOnServerLoad: ((inst: runtime.Instance) => Promise<void>) | undefined = undefined,
  ) {
    this.url = url;
    this.key = key;
    this.name = "WebSocketRPCServer[" + this.key + "]: ";
    this.getImports = getImports;
    this.logger = logger;
    this.ndarrayCacheUrl = ndarrayCacheUrl;
    this.ndarrayCacheDevice = ndarrayCacheDevice;
    this.initProgressCallback = initProgressCallback;
    this.asyncOnServerLoad = asyncOnServerLoad;
    this.checkLittleEndian();
    this.socket = compact.createWebSocket(url);
    this.socket.binaryType = "arraybuffer";

    this.socket.addEventListener("open", (event: Event) => {
      return this.onOpen(event);
    });
    this.socket.addEventListener("message", (event: MessageEvent) => {
      return this.onMessage(event);
    });
    this.socket.addEventListener("close", (event: CloseEvent) => {
      return this.onClose(event);
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private onClose(_event: CloseEvent): void {
    if (this.inst !== undefined) {
      this.globalObjects.forEach(obj => {
        obj.dispose();
      });
      this.log(this.inst.runtimeStatsText());
      this.inst.dispose();
    }
    if (this.state === RPCServerState.ReceivePacketHeader) {
      this.log("Closing the server in clean state");
      this.log("Automatic reconnecting..");
      new RPCServer(
        this.url, this.key, this.getImports, this.logger,
        this.ndarrayCacheUrl, this.ndarrayCacheDevice,
        this.initProgressCallback, this.asyncOnServerLoad);
    } else {
      this.log("Closing the server, final state=" + this.state);
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private onOpen(_event: Event): void {
    // Send the headers
    let bkey = StringToUint8Array("server:" + this.key);
    bkey = bkey.slice(0, bkey.length - 1);
    const intbuf = new Int32Array(1);
    intbuf[0] = RPC_MAGIC;
    this.socket.send(intbuf);
    intbuf[0] = bkey.length;
    this.socket.send(intbuf);
    this.socket.send(bkey);
    this.log("connected...");
    // request bytes: magic + keylen
    this.requestBytes(SizeOf.I32 + SizeOf.I32);
    this.state = RPCServerState.InitHeader;
  }

  /** Handler for raw message. */
  private onMessage(event: MessageEvent): void {
    const buffer = event.data;
    this.buffredBytes += buffer.byteLength;
    this.messageQueue.push(new Uint8Array(buffer));
    this.processEvents();
  }
  /** Process ready events. */
  private processEvents(): void {
    while (this.buffredBytes >= this.pendingBytes && this.pendingBytes != 0) {
      this.onDataReady();
    }
  }
  /** State machine to handle each request */
  private onDataReady(): void {
    switch (this.state) {
      case RPCServerState.InitHeader: {
        this.handleInitHeader();
        break;
      }
      case RPCServerState.InitHeaderKey: {
        this.handleInitHeaderKey();
        break;
      }
      case RPCServerState.ReceivePacketHeader: {
        this.currPacketHeader = this.readFromBuffer(SizeOf.I64);
        const reader = new ByteStreamReader(this.currPacketHeader);
        this.currPacketLength = reader.readU64();
        assert(this.pendingBytes === 0);
        this.requestBytes(this.currPacketLength);
        this.state = RPCServerState.ReceivePacketBody;
        break;
      }
      case RPCServerState.ReceivePacketBody: {
        const body = this.readFromBuffer(this.currPacketLength);
        assert(this.pendingBytes === 0);
        assert(this.currPacketHeader !== undefined);
        this.onPacketReady(this.currPacketHeader, body);
        break;
      }
      case RPCServerState.WaitForCallback: {
        assert(this.pendingBytes === 0);
        break;
      }
      default: {
        throw new Error("Cannot handle state " + this.state);
      }
    }
  }

  private onPacketReady(header: Uint8Array, body: Uint8Array): void {
    if (this.inst === undefined) {
      // initialize server.
      const reader = new ByteStreamReader(body);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const code = reader.readU32();
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const ver = Uint8ArrayToString(reader.readByteArray());
      const nargs = reader.readU32();
      const tcodes = [];
      const args = [];
      for (let i = 0; i < nargs; ++i) {
        tcodes.push(reader.readU32());
      }

      for (let i = 0; i < nargs; ++i) {
        const tcode = tcodes[i];
        if (tcode === ArgTypeCode.TVMStr) {
          const str = Uint8ArrayToString(reader.readByteArray());
          args.push(str);
        } else if (tcode === ArgTypeCode.TVMBytes) {
          args.push(reader.readByteArray());
        } else {
          throw new Error("cannot support type code " + tcode);
        }
      }
      this.onInitServer(args, header, body);
    } else {
      assert(this.serverRecvData !== undefined);
      this.serverRecvData(header, body);
      this.requestBytes(SizeOf.I64);
      this.state = RPCServerState.ReceivePacketHeader;
    }
  }

  /** Event handler during server initialization. */
  private onInitServer(
    args: Array<unknown>,
    header: Uint8Array,
    body: Uint8Array
  ): void {
    // start the server
    assert(args[0] === "rpc.WasmSession");
    assert(this.pendingBytes === 0);

    const asyncInitServer = async (): Promise<void> => {
      assert(args[1] instanceof Uint8Array);
      const inst = await runtime.instantiate(
        args[1].buffer,
        this.getImports(),
        this.logger
      );

      try {
        const output: GPUDeviceDetectOutput | undefined = await detectGPUDevice();
        if (output !== undefined) {
          const label = "WebGPU: "+ output.adapterInfo.description;
          this.log("Initialize GPU device: " + label);
          inst.initWebGPU(output.device);
        } else {
          this.log("Cannot find WebGPU device in the env");
        }
      } catch (err) {
        this.log("Cannnot initialize WebGPU, " + err.toString());
      }

      this.inst = inst;
      // begin scope to allow handling of objects
      this.inst.beginScope();
      if (this.initProgressCallback !== undefined) {
        this.inst.registerInitProgressCallback(this.initProgressCallback);
      }

      if (this.ndarrayCacheUrl.length != 0) {
        if (this.ndarrayCacheDevice === "cpu") {
          await this.inst.fetchNDArrayCache(this.ndarrayCacheUrl, this.inst.cpu());
        } else {
          assert(this.ndarrayCacheDevice === "webgpu");
          await this.inst.fetchNDArrayCache(this.ndarrayCacheUrl, this.inst.webgpu());
        }
      }

      assert(this.inst !== undefined);
      if (this.asyncOnServerLoad !== undefined) {
        await this.asyncOnServerLoad(this.inst);
      }
      const fcreate = this.inst.getGlobalFunc("rpc.CreateEventDrivenServer");
      const messageHandler = fcreate(
        (cbytes: Uint8Array): runtime.Scalar => {
          assert(this.inst !== undefined);
          if (this.socket.readyState === 1) {
            // WebSocket will automatically close the socket
            // if we burst send data that exceeds its internal buffer
            // wait a bit before we send next one.
            const sendDataWithCongestionControl = async (): Promise<void> => {
              const packetSize = 4 << 10;
              const maxBufferAmount = 4 * packetSize;
              const waitTimeMs = 20;
              for (
                let offset = 0;
                offset < cbytes.length;
                offset += packetSize
              ) {
                const end = Math.min(offset + packetSize, cbytes.length);
                while (this.socket.bufferedAmount >= maxBufferAmount) {
                  await new Promise((r) => setTimeout(r, waitTimeMs));
                }
                this.socket.send(cbytes.slice(offset, end));
              }
            };
            // Chain up the pending send so that the async send is always in-order.
            this.pendingSend = this.pendingSend.then(
              sendDataWithCongestionControl
            );
            // Directly return since the data are "sent" from the caller's pov.
            return this.inst.scalar(cbytes.length, "int32");
          } else {
            return this.inst.scalar(0, "int32");
          }
        },
        this.name,
        this.key
      );
      // message handler should persist across RPC runs
      this.globalObjects.push(
        this.inst.detachFromCurrentScope(messageHandler)
      );
      const writeFlag = this.inst.scalar(3, "int32");

      this.serverRecvData = (header: Uint8Array, body: Uint8Array): void => {
        if (messageHandler(header, writeFlag) === 0) {
          this.socket.close();
        }
        if (messageHandler(body, writeFlag) === 0) {
          this.socket.close();
        }
      };

      // Forward the same init sequence to the wasm RPC.
      // The RPC will look for "rpc.wasmSession"
      // and we will redirect it to the correct local session.
      // register the callback to redirect the session to local.
      const flocal = this.inst.getGlobalFunc("wasm.LocalSession");
      const localSession = flocal();
      assert(localSession instanceof runtime.Module);

      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      this.inst.registerFunc(
        "rpc.WasmSession",
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        (_args: unknown): runtime.Module => {
          return localSession;
        }
      );
      messageHandler(header, writeFlag);
      messageHandler(body, writeFlag);

      this.log("Finish initializing the Wasm Server..");
      this.requestBytes(SizeOf.I64);
      this.state = RPCServerState.ReceivePacketHeader;
      // call process events in case there are bufferred data.
      this.processEvents();
      // recycle all values.
      this.inst.endScope();
    };

    this.state = RPCServerState.WaitForCallback;
    asyncInitServer();
  }

  private log(msg: string): void {
    this.logger(this.name + msg);
  }

  private handleInitHeader(): void {
    const reader = new ByteStreamReader(this.readFromBuffer(SizeOf.I32 * 2));
    const magic = reader.readU32();
    if (magic === RPC_MAGIC + 1) {
      throw new Error("key: " + this.key + " has already been used in proxy");
    } else if (magic === RPC_MAGIC + 2) {
      throw new Error("RPCProxy do not have matching client key " + this.key);
    }
    assert(magic === RPC_MAGIC, this.url + " is not an RPC Proxy");
    this.remoteKeyLength = reader.readU32();
    assert(this.pendingBytes === 0);
    this.requestBytes(this.remoteKeyLength);
    this.state = RPCServerState.InitHeaderKey;
  }

  private handleInitHeaderKey(): void {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const remoteKey = Uint8ArrayToString(
      this.readFromBuffer(this.remoteKeyLength)
    );
    assert(this.pendingBytes === 0);
    this.requestBytes(SizeOf.I64);
    this.state = RPCServerState.ReceivePacketHeader;
  }

  private checkLittleEndian(): void {
    const a = new ArrayBuffer(4);
    const b = new Uint8Array(a);
    const c = new Uint32Array(a);
    b[0] = 0x11;
    b[1] = 0x22;
    b[2] = 0x33;
    b[3] = 0x44;
    assert(c[0] === 0x44332211, "RPCServer little endian to work");
  }

  private requestBytes(nbytes: number): void {
    this.pendingBytes += nbytes;
  }

  private readFromBuffer(nbytes: number): Uint8Array {
    const ret = new Uint8Array(nbytes);
    let ptr = 0;
    while (ptr < nbytes) {
      assert(this.messageQueue.length != 0);
      const nleft = nbytes - ptr;
      if (this.messageQueue[0].byteLength <= nleft) {
        const buffer = this.messageQueue.shift() as Uint8Array;
        ret.set(buffer, ptr);
        ptr += buffer.byteLength;
      } else {
        const buffer = this.messageQueue[0];
        ret.set(buffer.slice(0, nleft), ptr);
        this.messageQueue[0] = buffer.slice(nleft, buffer.byteLength);
        ptr += nleft;
      }
    }
    this.buffredBytes -= nbytes;
    this.pendingBytes -= nbytes;
    return ret;
  }
}
