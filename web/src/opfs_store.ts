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

export type OPFSAccessMode = "async" | "sync" | "auto";

type OPFSEffectiveAccessMode = "async" | "sync";
type OPFSSyncAccessHandleMode = "read-only" | "readwrite";

interface OPFSWritableFileStream extends WritableStream<Uint8Array> {
  write(value: Blob | BufferSource | string): Promise<void>;
  close(): Promise<void>;
}

interface OPFSFileHandle {
  getFile(): Promise<Blob>;
  createWritable(): Promise<OPFSWritableFileStream>;
  createSyncAccessHandle?: (options?: {
    mode?: OPFSSyncAccessHandleMode;
  }) => Promise<OPFSSyncAccessHandle>;
}

interface OPFSDirectoryHandle {
  getDirectoryHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OPFSDirectoryHandle>;
  getFileHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OPFSFileHandle>;
  removeEntry(name: string): Promise<void>;
}

interface OPFSStorageManager {
  getDirectory?: () => Promise<OPFSDirectoryHandle>;
}

interface OPFSStoreMetadata {
  url: string;
  contentType?: string;
}

interface OPFSStoredEntry {
  payloadHandle: OPFSFileHandle;
  metadata: OPFSStoreMetadata;
}

interface OPFSSyncAccessHandle {
  getSize(): number;
  read(buffer: BufferSource, options?: { at?: number }): number;
  write(buffer: BufferSource, options?: { at?: number }): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
}

type OPFSGlobalScope = typeof globalThis & {
  DedicatedWorkerGlobalScope?: new () => object;
  FileSystemFileHandle?: {
    prototype?: {
      createSyncAccessHandle?: unknown;
    };
  };
};

const HASH_ALGORITHM = "SHA-256";
const OPFS_STORE_ROOT_DIRECTORY = "tvmjs-opfs-store";

export class OPFSStore {
  private readonly scope: string;
  private readonly requestedAccessMode: OPFSAccessMode;
  private accessMode: OPFSEffectiveAccessMode;
  private directoryPromise?: Promise<OPFSDirectoryHandle>;

  constructor(scope: string, accessMode: OPFSAccessMode = "async") {
    this.scope = scope;
    this.requestedAccessMode = accessMode;
    this.accessMode = OPFSStore.resolveAccessMode(accessMode);
  }

  static isAvailable(): boolean {
    const storage = OPFSStore.getStorageManager();
    return storage !== undefined && typeof storage.getDirectory === "function";
  }

  private static resolveAccessMode(
    accessMode: OPFSAccessMode,
  ): OPFSEffectiveAccessMode {
    if (accessMode !== "auto") {
      return accessMode;
    }
    return OPFSStore.isDedicatedWorkerWithSyncAccessHandle()
      ? "sync"
      : "async";
  }

  async has(url: string): Promise<boolean> {
    return (await this.getExistingEntry(url)) !== undefined;
  }

  async read(url: string): Promise<Response | undefined> {
    const entry = await this.getExistingEntry(url);
    if (entry === undefined) {
      return undefined;
    }
    if (this.accessMode === "async") {
      const blob = await entry.payloadHandle.getFile();
      return new Response(blob, this.getResponseInit(entry.metadata));
    }
    const payload = await this.readPayload(entry.payloadHandle);
    return new Response(payload, this.getResponseInit(entry.metadata));
  }

  async readArrayBuffer(url: string): Promise<ArrayBuffer | undefined> {
    const entry = await this.getExistingEntry(url);
    if (entry === undefined) {
      return undefined;
    }
    return this.readPayload(entry.payloadHandle);
  }

  async write(url: string, response: Response): Promise<void> {
    const directory = await this.getScopedDirectory();
    const baseName = await this.hashUrl(url);
    const payloadHandle = await directory.getFileHandle(`${baseName}.bin`, {
      create: true,
    });
    const metadataHandle = await directory.getFileHandle(
      `${baseName}.meta.json`,
      { create: true },
    );
    const metadata: OPFSStoreMetadata = {
      url,
      contentType: response.headers.get("content-type") ?? undefined,
    };
    await this.writePayload(payloadHandle, response);
    await this.writeMetadata(metadataHandle, metadata);
  }

  async remove(url: string): Promise<void> {
    const directory = await this.getScopedDirectory();
    const baseName = await this.hashUrl(url);
    await this.removeEntryIfExists(directory, `${baseName}.bin`);
    await this.removeEntryIfExists(directory, `${baseName}.meta.json`);
  }

  private async getExistingEntry(
    url: string,
  ): Promise<OPFSStoredEntry | undefined> {
    const directory = await this.getScopedDirectory();
    const baseName = await this.hashUrl(url);
    const payloadHandle = await this.getFileHandleIfExists(
      directory,
      `${baseName}.bin`,
      false,
    );
    if (payloadHandle === undefined) {
      return undefined;
    }
    const metadataHandle = await this.getFileHandleIfExists(
      directory,
      `${baseName}.meta.json`,
      false,
    );
    if (metadataHandle === undefined) {
      return undefined;
    }
    const metadata = await this.readMetadata(metadataHandle);
    if (metadata === undefined) {
      return undefined;
    }
    if (metadata.url !== url) {
      throw new Error("OPFSStore: metadata URL does not match key URL.");
    }
    return { payloadHandle, metadata };
  }

  private static getStorageManager(): OPFSStorageManager | undefined {
    if (typeof navigator === "undefined") {
      return undefined;
    }
    return navigator.storage as unknown as OPFSStorageManager;
  }

  private static isDedicatedWorkerWithSyncAccessHandle(): boolean {
    const scope = globalThis as OPFSGlobalScope;
    return (
      typeof scope.DedicatedWorkerGlobalScope === "function" &&
      globalThis instanceof scope.DedicatedWorkerGlobalScope &&
      typeof scope.FileSystemFileHandle?.prototype?.createSyncAccessHandle ===
        "function"
    );
  }

  private async getScopedDirectory(): Promise<OPFSDirectoryHandle> {
    if (this.directoryPromise !== undefined) {
      return this.directoryPromise;
    }
    // Cache scoped directory handle to avoid repeated tree traversal
    this.directoryPromise = (async () => {
      const storage = OPFSStore.getStorageManager();
      if (storage === undefined || typeof storage.getDirectory !== "function") {
        throw new Error("OPFSStore: OPFS API unavailable.");
      }
      let directory = await storage.getDirectory();
      directory = await directory.getDirectoryHandle(OPFS_STORE_ROOT_DIRECTORY, {
        create: true,
      });
      const scopeParts = this.scope.split("/").filter((part) => part.length > 0);
      for (const part of scopeParts) {
        directory = await directory.getDirectoryHandle(
          encodeURIComponent(part),
          { create: true },
        );
      }
      return directory;
    })();
    return this.directoryPromise;
  }

  private async readMetadata(
    fileHandle: OPFSFileHandle,
  ): Promise<OPFSStoreMetadata | undefined> {
    try {
      const text = await (await fileHandle.getFile()).text();
      const parsed = JSON.parse(text);
      if (
        parsed === undefined ||
        parsed === null ||
        typeof parsed !== "object" ||
        typeof parsed.url !== "string"
      ) {
        throw new Error("OPFSStore: invalid metadata format.");
      }
      const metadata: OPFSStoreMetadata = {
        url: parsed.url,
      };
      if (typeof parsed.contentType === "string") {
        metadata.contentType = parsed.contentType;
      }
      return metadata;
    } catch (err) {
      if (this.isNotFoundError(err)) {
        // Treat metadata disappearance between lookup and read as a cache miss
        return undefined;
      }
      throw err;
    }
  }

  private getResponseInit(metadata: OPFSStoreMetadata): ResponseInit | undefined {
    return metadata.contentType !== undefined
      ? { headers: { "content-type": metadata.contentType } }
      : undefined;
  }

  private async writeMetadata(
    handle: OPFSFileHandle,
    metadata: OPFSStoreMetadata,
  ): Promise<void> {
    const writable = await handle.createWritable();
    await writable.write(new TextEncoder().encode(JSON.stringify(metadata)));
    await writable.close();
  }

  private async readPayload(handle: OPFSFileHandle): Promise<ArrayBuffer> {
    const syncHandle = await this.openSyncAccessHandle(handle, "read-only");
    return syncHandle !== undefined
      ? this.readPayloadWithSyncHandle(syncHandle)
      : (await handle.getFile()).arrayBuffer();
  }

  private async writePayload(
    handle: OPFSFileHandle,
    response: Response,
  ): Promise<void> {
    const syncHandle = await this.openSyncAccessHandle(handle, "readwrite");
    if (syncHandle !== undefined) {
      await this.writePayloadWithSyncHandle(syncHandle, response);
      return;
    }
    await this.writePayloadWithWritable(handle, response);
  }

  private async writePayloadWithWritable(
    handle: OPFSFileHandle,
    response: Response,
  ): Promise<void> {
    const writable = await handle.createWritable();
    if (response.body !== null) {
      await response.body.pipeTo(writable);
    } else {
      await writable.write(await response.arrayBuffer());
      await writable.close();
    }
  }

  private readPayloadWithSyncHandle(
    syncHandle: OPFSSyncAccessHandle,
  ): ArrayBuffer {
    try {
      const size = syncHandle.getSize();
      const payload = new ArrayBuffer(size);
      syncHandle.read(new Uint8Array(payload), { at: 0 });
      return payload;
    } finally {
      syncHandle.close();
    }
  }

  private async writePayloadWithSyncHandle(
    syncHandle: OPFSSyncAccessHandle,
    response: Response,
  ): Promise<void> {
    try {
      syncHandle.truncate(0);
      let offset = 0;
      if (response.body !== null) {
        const reader = response.body.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            syncHandle.write(value, { at: offset });
            offset += value.byteLength;
          }
        } finally {
          reader.releaseLock();
        }
      } else {
        const payload = await response.arrayBuffer();
        syncHandle.write(new Uint8Array(payload), { at: 0 });
      }
      syncHandle.flush();
    } finally {
      syncHandle.close();
    }
  }

  private async openSyncAccessHandle(
    handle: OPFSFileHandle,
    mode: OPFSSyncAccessHandleMode,
  ): Promise<OPFSSyncAccessHandle | undefined> {
    if (this.accessMode === "async") {
      return undefined;
    }
    if (typeof handle.createSyncAccessHandle !== "function") {
      throw this.createSyncUnavailableError();
    }
    try {
      return await handle.createSyncAccessHandle({ mode });
    } catch (err) {
      const isLockContention =
        err &&
        typeof err === "object" &&
        "name" in err &&
        (err as { name?: unknown }).name === "NoModificationAllowedError";
      if (this.requestedAccessMode === "auto" && isLockContention) {
        return undefined;
      }
      throw err;
    }
  }

  private async getFileHandleIfExists(
    directory: OPFSDirectoryHandle,
    filename: string,
    create: boolean,
  ): Promise<OPFSFileHandle | undefined> {
    try {
      return await directory.getFileHandle(filename, { create });
    } catch (err) {
      if (this.isNotFoundError(err)) {
        // NotFound maps to cache miss semantics
        return undefined;
      }
      throw err;
    }
  }

  private async removeEntryIfExists(
    directory: OPFSDirectoryHandle,
    filename: string,
  ): Promise<void> {
    try {
      await directory.removeEntry(filename);
    } catch (err) {
      if (this.isNotFoundError(err)) {
        // Delete is intentionally idempotent for missing entries
        return;
      }
      throw err;
    }
  }

  private async hashUrl(url: string): Promise<string> {
    const textEncoder = new TextEncoder();
    const input = textEncoder.encode(url);
    if (
      typeof crypto === "undefined" ||
      crypto.subtle === undefined ||
      typeof crypto.subtle.digest !== "function"
    ) {
      throw new Error("OPFSStore: crypto.subtle.digest is unavailable.");
    }
    const digest = await crypto.subtle.digest(HASH_ALGORITHM, input);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }

  private isNotFoundError(err: unknown): boolean {
    if (err && typeof err === "object" && "name" in err) {
      const name = (err as { name?: unknown }).name;
      return name === "NotFoundError";
    }
    return false;
  }

  private createSyncUnavailableError(): Error {
    const err = new Error(
      "OPFSStore: createSyncAccessHandle unavailable; sync OPFS access requires a supported dedicated worker context.",
    );
    err.name = "NotSupportedError";
    return err;
  }
}
