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
const fs = require("fs");
const path = require("path");
const tvmjs = require("../../dist/tvmjs.bundle");

const wasmSource = fs.readFileSync(
  path.join(__dirname, "../../dist/wasm/tvmjs_runtime.wasm"),
);

function createInstance() {
  return new tvmjs.Instance(
    new WebAssembly.Module(wasmSource),
    tvmjs.createPolyfillWASI(),
  );
}

function encodeBF16(bits) {
  const bytes = new Uint8Array(bits.length * 2);
  for (let i = 0; i < bits.length; ++i) {
    bytes[2 * i] = bits[i] & 0xff;
    bytes[2 * i + 1] = bits[i] >>> 8;
  }
  return bytes;
}

function decodedF32Bytes(bits) {
  const bytes = new Uint8Array(bits.length * 4);
  for (let i = 0; i < bits.length; ++i) {
    bytes[4 * i + 2] = bits[i] & 0xff;
    bytes[4 * i + 3] = bits[i] >>> 8;
  }
  return bytes;
}

function createArtifactCache(manifest, shard) {
  return {
    hasAllKeys: async () => true,
    addToCache: async () => {},
    deleteInCache: async () => {},
    fetchWithCache: async (_url, storeType) => {
      return storeType === "json" ? manifest : shard;
    },
  };
}

let tvm;

beforeAll(() => {
  tvm = createInstance();
});

afterAll(() => {
  tvm.dispose();
});

test("in-place BF16 expansion preserves exact floating-point bits", () => {
  const decode = tvm.ctx.arrayDecodeBF16ToF32Inplace;
  const bits = [
    0x0000,
    0x8000,
    0x3f80,
    0xc020,
    0x0001,
    0x7f80,
    0xff80,
    0x7fc1,
  ];
  const encoded = encodeBF16(bits);

  expect(decode).toBeDefined();
  tvm.withNewScope(() => {
    const tensor = tvm.empty([bits.length], "float32");
    tvm.memory.storeRawBytes(tensor.getCPUDataAddress(), encoded);
    decode(tensor, new tvmjs.Scalar(encoded.byteLength, "int64"));
    expect(Array.from(tensor.toRawBytes())).toEqual(
      Array.from(decodedF32Bytes(bits)),
    );
  });
});

test.each([
  [[], [0x3f80]],
  [[0], []],
  [[1], [0x8000]],
  [[3], [0x3f80, 0xc000, 0x7fc1]],
  [[2, 2], [0x0001, 0x3f00, 0x7f80, 0xff80]],
])("in-place BF16 expansion supports shape %j", (shape, bits) => {
  const encoded = encodeBF16(bits);

  tvm.withNewScope(() => {
    const tensor = tvm.empty(shape, "float32");
    tvm.memory.storeRawBytes(tensor.getCPUDataAddress(), encoded);
    tvm.ctx.arrayDecodeBF16ToF32Inplace(
      tensor,
      new tvmjs.Scalar(encoded.byteLength, "int64"),
    );
    expect(Array.from(tensor.toRawBytes())).toEqual(
      Array.from(decodedF32Bytes(bits)),
    );
  });
});

test("both BF16 decoders honor an unaligned tensor byte offset", () => {
  const bits = [0x3f80, 0xc000, 0x7fc1];
  const encoded = encodeBF16(bits);
  const expected = decodedF32Bytes(bits);

  tvm.withNewScope(() => {
    for (const inplace of [true, false]) {
      const base = tvm.empty([expected.byteLength + 2], "uint8");
      base.copyFromRawBytes(
        new Uint8Array(expected.byteLength + 2).fill(0xa5),
      );
      const view = tvm.ctx.tensorCreateView(
        base,
        tvm.ctx.makeShapeTuple(new tvmjs.Scalar(bits.length, "int")),
        "float32",
        new tvmjs.Scalar(1, "int64"),
      );
      if (inplace) {
        tvm.memory.storeRawBytes(view.getCPUDataAddress(), encoded);
        tvm.ctx.arrayDecodeBF16ToF32Inplace(
          view,
          new tvmjs.Scalar(encoded.byteLength, "int64"),
        );
      } else {
        tvm.ctx.arrayDecodeStorage(
          view,
          encoded,
          "f32-to-bf16",
          "float32",
        );
      }

      const result = base.toRawBytes();
      expect(result[0]).toBe(0xa5);
      expect(Array.from(result.subarray(1, 1 + expected.byteLength))).toEqual(
        Array.from(expected),
      );
      expect(result[result.length - 1]).toBe(0xa5);
    }
  });
});

test("in-place BF16 expansion rejects invalid tensor contracts", () => {
  const error = jest.spyOn(console, "error").mockImplementation(() => {});
  const originalExitCode = process.exitCode;
  const tvm = createInstance();

  try {
    tvm.withNewScope(() => {
      const tensor = tvm.empty([2], "float32");
      const encoded = encodeBF16([0x3f80, 0x4000]);
      tvm.memory.storeRawBytes(tensor.getCPUDataAddress(), encoded);
      expect(() => tvm.ctx.arrayDecodeBF16ToF32Inplace(
        tensor,
        new tvmjs.Scalar(encoded.byteLength - 1, "int64"),
      )).toThrow();
      expect(() => tvm.ctx.arrayDecodeBF16ToF32Inplace(
        tensor,
        new tvmjs.Scalar(encoded.byteLength + 1, "int64"),
      )).toThrow();
      expect(() => tvm.ctx.arrayDecodeBF16ToF32Inplace(
        tensor,
        new tvmjs.Scalar(-1, "int64"),
      )).toThrow();

      const wrongDtype = tvm.empty([2], "int32");
      expect(() => tvm.ctx.arrayDecodeBF16ToF32Inplace(
        wrongDtype,
        new tvmjs.Scalar(encoded.byteLength, "int64"),
      )).toThrow();

      expect(() => tvm.ctx.arrayDecodeStorage(
        tensor,
        encoded.subarray(0, encoded.byteLength - 1),
        "f32-to-bf16",
        "float32",
      )).toThrow();
      expect(() => tvm.ctx.arrayDecodeStorage(
        tensor,
        new Uint8Array(encoded.byteLength * 2),
        "f32-to-bf16",
        "float32",
      )).toThrow();
      expect(() => tvm.ctx.arrayDecodeStorage(
        wrongDtype,
        encoded,
        "f32-to-bf16",
        "float32",
      )).toThrow();
    });
  } finally {
    tvm.dispose();
    error.mockRestore();
    // Emscripten marks expected native contract failures as process failures.
    process.exitCode = originalExitCode;
  }
});

test("tensor cache uses in-place expansion only for packed BF16 records", async () => {
  const packedBits = [0x3f80, 0xc000];
  const packed = encodeBF16(packedBits);
  const shard = Uint8Array.from([1, 2, 3, 4, ...packed]).buffer;
  const manifest = {
    metadata: {},
    records: [{
      dataPath: "params.bin",
      format: "raw-shard",
      nbytes: shard.byteLength,
      records: [
        {
          name: "test.decode.raw",
          shape: [4],
          dtype: "uint8",
          format: "raw",
          byteOffset: 0,
          nbytes: 4,
        },
        {
          name: "test.decode.packed_bf16",
          shape: [2],
          dtype: "float32",
          format: "f32-to-bf16",
          byteOffset: 4,
          nbytes: packed.byteLength,
        },
      ],
    }],
  };
  const originalInplace = tvm.ctx.arrayDecodeBF16ToF32Inplace;
  const originalStorage = tvm.ctx.arrayDecodeStorage;
  const inplace = jest.fn((...args) => originalInplace(...args));
  const storage = jest.fn((...args) => originalStorage(...args));
  tvm.ctx.arrayDecodeBF16ToF32Inplace = inplace;
  tvm.ctx.arrayDecodeStorage = storage;

  try {
    await tvm.fetchTensorCache(
      "https://example.test/model/",
      tvm.cpu(),
      { artifactCache: createArtifactCache(manifest, shard) },
    );

    expect(inplace).toHaveBeenCalledTimes(1);
    expect(storage).toHaveBeenCalledTimes(1);
    tvm.withNewScope(() => {
      expect(Array.from(tvm.tensorCacheGet("test.decode.raw").toRawBytes()))
        .toEqual([1, 2, 3, 4]);
      expect(Array.from(
        tvm.tensorCacheGet("test.decode.packed_bf16").toRawBytes(),
      )).toEqual(Array.from(decodedF32Bytes(packedBits)));
    });
  } finally {
    tvm.ctx.arrayDecodeBF16ToF32Inplace = originalInplace;
    tvm.ctx.arrayDecodeStorage = originalStorage;
    tvm.tensorCacheClear();
  }
});

test("tensor cache falls back when in-place expansion is unavailable", async () => {
  const bits = [0x3f80, 0xc000];
  const packed = encodeBF16(bits);
  const manifest = {
    metadata: {},
    records: [{
      dataPath: "params.bin",
      format: "raw-shard",
      nbytes: packed.byteLength,
      records: [{
        name: "test.decode.fallback",
        shape: [2],
        dtype: "float32",
        format: "f32-to-bf16",
        byteOffset: 0,
        nbytes: packed.byteLength,
      }],
    }],
  };
  const originalInplace = tvm.ctx.arrayDecodeBF16ToF32Inplace;
  const originalStorage = tvm.ctx.arrayDecodeStorage;
  const storage = jest.fn((...args) => originalStorage(...args));
  tvm.ctx.arrayDecodeBF16ToF32Inplace = undefined;
  tvm.ctx.arrayDecodeStorage = storage;

  try {
    await tvm.fetchTensorCache(
      "https://example.test/model/",
      tvm.cpu(),
      { artifactCache: createArtifactCache(manifest, packed.buffer) },
    );
    expect(storage).toHaveBeenCalledTimes(1);
    tvm.withNewScope(() => {
      expect(Array.from(
        tvm.tensorCacheGet("test.decode.fallback").toRawBytes(),
      )).toEqual(Array.from(decodedF32Bytes(bits)));
    });
  } finally {
    tvm.ctx.arrayDecodeBF16ToF32Inplace = originalInplace;
    tvm.ctx.arrayDecodeStorage = originalStorage;
    tvm.tensorCacheClear();
  }
});

test("tensor cache disposes its CPU tensor when in-place expansion fails", async () => {
  const packed = encodeBF16([0x3f80, 0xc000]);
  const manifest = {
    metadata: {},
    records: [{
      dataPath: "params.bin",
      format: "raw-shard",
      nbytes: packed.byteLength,
      records: [{
        name: "test.decode.failure",
        shape: [2],
        dtype: "float32",
        format: "f32-to-bf16",
        byteOffset: 0,
        nbytes: packed.byteLength,
      }],
    }],
  };
  const originalLogger = tvm.env.logger;
  const originalInplace = tvm.ctx.arrayDecodeBF16ToF32Inplace;
  const empty = jest.spyOn(tvm, "empty");
  tvm.env.logger = () => {};
  tvm.ctx.arrayDecodeBF16ToF32Inplace = jest.fn(() => {
    throw new Error("in-place decode failed");
  });

  try {
    await expect(tvm.fetchTensorCache(
      "https://example.test/model/",
      tvm.cpu(),
      { artifactCache: createArtifactCache(manifest, packed.buffer) },
    )).rejects.toThrow("in-place decode failed");
    expect(empty).toHaveBeenCalledTimes(1);
    expect(empty.mock.results[0].value.getHandle(false)).toBe(0);
  } finally {
    empty.mockRestore();
    tvm.ctx.arrayDecodeBF16ToF32Inplace = originalInplace;
    tvm.env.logger = originalLogger;
  }
});

test.each([
  [[2], 8, "shape requires 4"],
  [[1.5], 3, "Invalid tensor dimension"],
  [[Number.MAX_SAFE_INTEGER, 2], 0, "safe integer range"],
])(
  "tensor cache rejects malformed packed BF16 metadata for shape %j",
  async (shape, nbytes, message) => {
    const originalLogger = tvm.env.logger;
    tvm.env.logger = () => {};
    const empty = jest.spyOn(tvm, "empty");
    const shard = new ArrayBuffer(nbytes);
    const slice = jest.spyOn(shard, "slice");
    const manifest = {
      metadata: {},
      records: [{
        dataPath: "params.bin",
        format: "raw-shard",
        nbytes,
        records: [{
          name: "test.decode.invalid",
          shape,
          dtype: "float32",
          format: "f32-to-bf16",
          byteOffset: 0,
          nbytes,
        }],
      }],
    };

    try {
      await expect(tvm.fetchTensorCache(
        "https://example.test/model/",
        tvm.cpu(),
        { artifactCache: createArtifactCache(manifest, shard) },
      )).rejects.toThrow(message);
      expect(empty).not.toHaveBeenCalled();
      expect(slice).not.toHaveBeenCalled();
    } finally {
      slice.mockRestore();
      empty.mockRestore();
      tvm.env.logger = originalLogger;
    }
  },
);
